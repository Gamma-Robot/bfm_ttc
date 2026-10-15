from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def _tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _constant(name, values):
    return helper.make_node("Constant", [], [name + "_output_0"], name=name,
                            value=numpy_helper.from_array(np.asarray(values, np.int64)))


def _source_pair(tmp_path, *, low_latency=False):
    rng = np.random.default_rng(47)
    nodes = []
    orientation_range = (584, 644) if low_latency else (601, 661)
    for name, start, end in [("command", 4, 584), ("orientation", *orientation_range)]:
        nodes.extend([_constant(name + "_start", [start]), _constant(name + "_end", [end]),
                      _constant(name + "_axis", [1])])
        nodes.append(helper.make_node("Slice", ["obs_dict", name + "_start_output_0", name + "_end_output_0", name + "_axis_output_0"], [name]))
        shape = [1, 10, 58 if name == "command" else 6]
        nodes.extend([_constant(name + "_shape", shape), helper.make_node("Reshape", [name, name + "_shape_output_0"], [name + "_frames"])])
    nodes.append(helper.make_node("Concat", ["command_frames", "orientation_frames"], ["/Reshape_10_output_0"], axis=2))
    nodes.extend([
        _constant("/g1/Constant", [1, 640]),
        helper.make_node("Reshape", ["/Reshape_10_output_0", "/g1/Constant_output_0"], ["/g1/Reshape_output_0"], name="/g1/Reshape"),
        helper.make_node("MatMul", ["/g1/Reshape_output_0", "encoder_weight"], ["encoded"], name="/g1/test"),
        _constant("/g1/Constant_1", [1, 2, 32]),
        helper.make_node("Reshape", ["encoded", "/g1/Constant_1_output_0"], ["quantized"], name="/g1/Reshape_1"),
        helper.make_node("Cast", ["quantized"], ["/quantizer/Cast_output_0"], to=TensorProto.FLOAT, name="/quantizer/Cast"),
        _constant("flatten", [1, 64]),
        helper.make_node("Reshape", ["/quantizer/Cast_output_0", "flatten_output_0"], ["encoded_tokens"]),
    ])
    weight = numpy_helper.from_array(rng.normal(0, .01, (640, 64)).astype(np.float32), "encoder_weight")
    encoder_width = 1247 if low_latency else 1762
    encoder = helper.make_model(helper.make_graph(nodes, "encoder", [_tensor("obs_dict", [1, encoder_width])], [_tensor("encoded_tokens", [1, 64])], [weight]), opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    nodes = [_constant("axis", [1]), helper.make_node("Unsqueeze", ["obs_dict", "axis_output_0"], ["/Concat_output_0"]),
             helper.make_node("MatMul", ["/Concat_output_0", "onnx::MatMul_136"], ["decoded"], name="/g1_dyn/module/module.0/MatMul"),
             _constant("/Constant_10", [2]), _constant("/Constant_11", [0]), _constant("/Constant_12", [29]), _constant("/Constant_13", [1]),
             helper.make_node("Slice", ["decoded", "/Constant_11_output_0", "/Constant_12_output_0", "/Constant_10_output_0", "/Constant_13_output_0"], ["/Slice_2_output_0"], name="/Slice_2"),
             _constant("/Constant_14", [1]), helper.make_node("Squeeze", ["/Slice_2_output_0", "/Constant_14_output_0"], ["action"], name="/Squeeze")]
    weight = numpy_helper.from_array(rng.normal(0, .01, (994, 29)).astype(np.float32), "onnx::MatMul_136")
    decoder = helper.make_model(helper.make_graph(nodes, "decoder", [_tensor("obs_dict", [1, 994])], [_tensor("action", [1, 29])], [weight]), opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    paths = [tmp_path / "encoder.onnx", tmp_path / "decoder.onnx"]
    for path, model in zip(paths, [encoder, decoder]):
        model.graph.node[-1].doc_string = "/private/source/export.py"
        onnx.save(model, path)
    return paths


@pytest.mark.parametrize("low_latency", [False, True])
def test_composition_matches_public_pair_without_source_metadata(tmp_path, monkeypatch, low_latency):
    from bfm_adapters.families.sonic import source_export

    encoder, decoder = _source_pair(tmp_path, low_latency=low_latency)
    identity_name = "_LOW_LATENCY_SOURCE_SHA256" if low_latency else "_SOURCE_SHA256"
    monkeypatch.setattr(source_export, identity_name, tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in [encoder, decoder]))
    export = source_export.export_sonic_low_latency if low_latency else source_export.export_sonic_release
    destination = tmp_path / "release.onnx"
    export(encoder, decoder, destination)
    assert b"/private/" not in destination.read_bytes()
    repeated = tmp_path / "repeated.onnx"
    export(encoder, decoder, repeated)
    assert repeated.read_bytes() == destination.read_bytes()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(p), sess_options=options, providers=["CPUExecutionProvider"]) for p in [encoder, decoder, destination]]
    rng = np.random.default_rng(18)
    for _ in range(4):
        command = rng.normal(size=640).astype(np.float32)
        proprioception = rng.normal(size=930).astype(np.float32)
        universal = np.zeros((1, 1247 if low_latency else 1762), np.float32)
        universal[:, 4:584] = command[:580]
        if low_latency:
            universal[:, 584:644] = command[580:]
        else:
            universal[:, 601:661] = command[580:]
        token = sessions[0].run(None, {"obs_dict": universal})[0]
        action = sessions[1].run(None, {"obs_dict": np.concatenate([token, proprioception[None]], axis=1)})[0]
        actual_action, actual_token = sessions[2].run(None, {"g1_input": command, "proprioception": proprioception})
        np.testing.assert_array_equal(actual_token, token[0])
        np.testing.assert_array_equal(actual_action, action[0])


@pytest.mark.parametrize("low_latency", [False, True])
def test_rejects_unverified_source_before_writing(tmp_path, low_latency):
    from bfm_adapters.families.sonic import source_export

    encoder, decoder = _source_pair(tmp_path, low_latency=low_latency)
    export = source_export.export_sonic_low_latency if low_latency else source_export.export_sonic_release
    destination = tmp_path / "release.onnx"
    with pytest.raises(ValueError, match="identity"):
        export(encoder, decoder, destination)
    assert not destination.exists()


@pytest.mark.parametrize("low_latency", [False, True])
def test_rejects_wrong_source_interface(tmp_path, monkeypatch, low_latency):
    from bfm_adapters.families.sonic import source_export

    encoder, decoder = _source_pair(tmp_path, low_latency=low_latency)
    model = onnx.load(encoder)
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1751
    onnx.save(model, encoder)
    identity_name = "_LOW_LATENCY_SOURCE_SHA256" if low_latency else "_SOURCE_SHA256"
    monkeypatch.setattr(source_export, identity_name, tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in [encoder, decoder]))
    export = source_export.export_sonic_low_latency if low_latency else source_export.export_sonic_release
    with pytest.raises(ValueError, match="interface"):
        export(encoder, decoder, tmp_path / "release.onnx")


@pytest.mark.parametrize("low_latency", [False, True])
def test_cli_accepts_explicit_source_pair(tmp_path, monkeypatch, low_latency):
    from bfm_adapters.families.sonic import source_export

    encoder, decoder = _source_pair(tmp_path, low_latency=low_latency)
    identity_name = "_LOW_LATENCY_SOURCE_SHA256" if low_latency else "_SOURCE_SHA256"
    monkeypatch.setattr(source_export, identity_name, tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in [encoder, decoder]))
    destination = tmp_path / "release.onnx"
    args = ["source_export", str(encoder), str(decoder), str(destination)]
    if low_latency:
        args.extend(["--variant", "low-latency"])
    monkeypatch.setattr(sys, "argv", args)
    source_export.main()
    onnx.checker.check_model(onnx.load(destination))
