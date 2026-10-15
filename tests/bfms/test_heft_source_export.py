from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def _source(tmp_path: Path) -> Path:
    outputs = {
        "policy_1": [1, 1729], "priv_pred": [1, 256],
        "_actor_inp": [1, 1985], "_actor_feature": [1, 512],
        "action": [1, 29], "scale": [1, 29],
        "action_alias_6": [1, 29], "action_log_prob": [1],
    }
    initializers = [numpy_helper.from_array(
        np.arange(1729, dtype=np.float32).reshape(1, -1) / 1729,
        "normalization_offset",
    )]
    nodes = [helper.make_node("Sub", ["policy", "normalization_offset"], ["policy_1"])]
    for name, shape in outputs.items():
        if name == "policy_1":
            continue
        initializers.append(numpy_helper.from_array(np.zeros(shape, np.float32), name + "_value"))
        nodes.append(helper.make_node("Identity", [name + "_value"], [name]))
    model = helper.make_model(helper.make_graph(
        nodes, "heft_source", [helper.make_tensor_value_info("policy", TensorProto.FLOAT, [1, 1729])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in outputs.items()],
        initializers,
    ), opset_imports=[helper.make_opsetid("", 20)], ir_version=10)
    path = tmp_path / "policy.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="policy.onnx.data", size_threshold=0)
    return path


def test_export_preserves_external_tensor_values_and_semantic_input_order(tmp_path: Path) -> None:
    from bfm_adapters.families.heft.source_export import export_heft_policy

    source = _source(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "prepared" / "policy.onnx"
    export_heft_policy(source, destination)
    assert source.read_bytes() == original
    exported = onnx.load(destination, load_external_data=False)
    assert all(t.data_location != TensorProto.EXTERNAL for t in exported.graph.initializer)
    source_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    policy = np.arange(1729, dtype=np.float32).reshape(1, -1)
    values = dict(zip(
        ["context", "motion_command", "target_motion", "proprioception"],
        np.split(policy, [1, 115, 921], axis=1),
    ))
    for expected, actual in zip(source_session.run(None, {"policy": policy}), session.run(None, values)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("malformation", ["input_name", "input_width", "input_type", "extra_input", "action_width", "missing_output"])
def test_export_rejects_wrong_source_interface(tmp_path: Path, malformation: str) -> None:
    from bfm_adapters.families.heft.source_export import export_heft_policy

    source = _source(tmp_path)
    model = onnx.load(source)
    if malformation == "input_name":
        model.graph.input[0].name = "obs"
    elif malformation == "input_width":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1728
    elif malformation == "input_type":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif malformation == "extra_input":
        model.graph.input.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1]))
    elif malformation == "action_width":
        next(x for x in model.graph.output if x.name == "action").type.tensor_type.shape.dim[1].dim_value = 28
    else:
        del model.graph.output[0]
    onnx.save(model, source)
    destination = tmp_path / "converted.onnx"
    with pytest.raises(ValueError, match="HEFT"):
        export_heft_policy(source, destination)
    assert not destination.exists()


def test_export_requires_external_data(tmp_path: Path) -> None:
    from bfm_adapters.families.heft.source_export import export_heft_policy

    source = _source(tmp_path)
    (tmp_path / "policy.onnx.data").unlink()
    with pytest.raises((OSError, RuntimeError, onnx.checker.ValidationError)):
        export_heft_policy(source, tmp_path / "converted.onnx")


def test_export_module_cli(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "converted.onnx"
    result = subprocess.run(
        [sys.executable, "-m", "bfm_adapters.families.heft.source_export", str(source), str(destination)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    onnx.checker.check_model(onnx.load(destination))


def test_export_removes_nested_exporter_paths_without_changing_weights(tmp_path: Path) -> None:
    from bfm_adapters.families.heft.source_export import export_heft_policy
    source = _source(tmp_path)
    model = onnx.load(source)
    model.doc_string = '/home/private-author/export.py'
    model.graph.node[0].metadata_props.add(key='stack_trace', value='/home/private-author/train.py')
    model.graph.initializer[0].doc_string = '/home/private-author/model.py'
    expected = [tensor.SerializeToString() for tensor in model.graph.initializer[1:]]
    onnx.save(model, source)
    destination = tmp_path / 'clean.onnx'
    export_heft_policy(source, destination)
    clean = onnx.load(destination)
    assert b'/home/private-author' not in clean.SerializeToString()
    assert [tensor.SerializeToString() for tensor in clean.graph.initializer[1:]] == expected
