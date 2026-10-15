from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from bfm_adapters.families.sonic import preparation as sonic_preparation  # noqa: F401
from bfm_adapters.preparation import apply_transform
from command_optimization.artifacts import load_lock, resolve


def test_sonic_low_latency_lock_declares_source_and_batch_recipe() -> None:
    entries = load_lock()
    source = resolve("sonic-low-latency-g1-policy", lock=entries)
    prepared = resolve("sonic-low-latency-g1-policy.batched", lock=entries)
    assert source.path == "artifacts/store/sonic/low_latency/g1/policy.onnx"
    assert prepared.source_id == source.id
    assert prepared.transform == "sonic_low_latency_g1_batch"
    assert prepared.transform_version == 1


def _constant(name: str, output: str, values: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=numpy_helper.from_array(np.asarray(values, dtype=np.int64)),
    )


def _raw_shaped_model() -> onnx.ModelProto:
    nodes = [
        _constant("/Constant", "/Constant_output_0", [0]),
        helper.make_node(
            "Unsqueeze",
            ["g1_input", "/Constant_output_0"],
            ["/Unsqueeze_output_0"],
            name="/Unsqueeze",
        ),
        _constant("/Constant_1", "/Constant_1_output_0", [0]),
        helper.make_node(
            "Unsqueeze",
            ["proprioception", "/Constant_1_output_0"],
            ["/Unsqueeze_1_output_0"],
            name="/Unsqueeze_1",
        ),
        _constant("/Constant_2", "/Constant_2_output_0", [0]),
        helper.make_node(
            "Unsqueeze",
            ["/Unsqueeze_output_0", "/Constant_2_output_0"],
            ["/Unsqueeze_2_output_0"],
            name="/Unsqueeze_2",
        ),
        _constant("/Constant_3", "/Constant_3_output_0", [0]),
        helper.make_node(
            "Unsqueeze",
            ["/Unsqueeze_1_output_0", "/Constant_3_output_0"],
            ["/Unsqueeze_3_output_0"],
            name="/Unsqueeze_3",
        ),
        _constant("/Constant_8", "/Constant_8_output_0", [1, 1, 10, 58]),
        _constant("/Constant_13", "/Constant_13_output_0", [1, 1, 10, 6]),
        _constant("/g1/Constant", "/g1/Constant_output_0", [1, 640]),
        _constant("/g1/Constant_1", "/g1/Constant_1_output_0", [1, 2, 32]),
        _constant(
            "/quantizer/Constant",
            "/quantizer/Constant_output_0",
            [1, 2, 1, 32],
        ),
        _constant(
            "/quantizer/Constant_5",
            "/quantizer/Constant_5_output_0",
            [1, 2, 32],
        ),
        _constant("/Constant_24", "/Constant_24_output_0", [0]),
        helper.make_node(
            "Squeeze",
            ["action_source", "/Constant_24_output_0"],
            ["/Squeeze_output_0"],
            name="/Squeeze",
        ),
        _constant("/Constant_25", "/Constant_25_output_0", [0]),
        helper.make_node(
            "Squeeze",
            ["/Squeeze_output_0", "/Constant_25_output_0"],
            ["action"],
            name="/Squeeze_1",
        ),
        helper.make_node(
            "Reshape",
            ["token_source", "sonic_token_output_shape"],
            ["token"],
            name="token_output",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "sonic-test-shape",
        [
            helper.make_tensor_value_info("g1_input", TensorProto.FLOAT, [640]),
            helper.make_tensor_value_info(
                "proprioception", TensorProto.FLOAT, [930]
            ),
        ],
        [
            helper.make_tensor_value_info("action", TensorProto.FLOAT, [29]),
            helper.make_tensor_value_info("token", TensorProto.FLOAT, [64]),
        ],
        initializer=[
            numpy_helper.from_array(
                np.zeros((1, 1, 29), dtype=np.float32), "action_source"
            ),
            numpy_helper.from_array(
                np.zeros(64, dtype=np.float32), "token_source"
            ),
            numpy_helper.from_array(
                np.asarray([64], dtype=np.int64), "sonic_token_output_shape"
            ),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def test_sonic_release_transform_promotes_real_graph_contract() -> None:
    model = _raw_shaped_model()

    apply_transform("sonic_release_g1_batch", 1, model)

    onnx.checker.check_model(model)
    assert {value.name: _shape(value) for value in model.graph.input} == {
        "g1_input": ["batch", 640],
        "proprioception": ["batch", 930],
    }
    assert {value.name: _shape(value) for value in model.graph.output} == {
        "action": ["batch", 29],
        "token": ["batch", 64],
    }
    nodes = {node.name: node for node in model.graph.node}
    assert nodes["/Unsqueeze_2"].input[0] == "g1_input"
    assert nodes["/Unsqueeze_3"].input[0] == "proprioception"
    assert nodes["/Squeeze"].output[0] == "action"
    assert "/Squeeze_1" not in nodes
    token_shape = next(
        value
        for value in model.graph.initializer
        if value.name == "sonic_token_output_shape"
    )
    assert numpy_helper.to_array(token_shape).tolist() == [-1, 64]


def test_sonic_release_transform_rejects_graph_drift() -> None:
    model = _raw_shaped_model()
    model.graph.input[0].name = "renamed"

    with pytest.raises(ValueError, match="inputs changed"):
        apply_transform("sonic_release_g1_batch", 1, model)


_LL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "store"
    / "sonic"
    / "low_latency"
    / "g1"
    / "policy.onnx"
)


def test_sonic_low_latency_transform_promotes_real_graph_contract() -> None:
    if not _LL_SOURCE.is_file():
        pytest.skip("SONIC low-latency source artifact is not installed")
    model = onnx.load(str(_LL_SOURCE))
    assert model.producer_name == "bfm-ttc-source-export"

    apply_transform("sonic_low_latency_g1_batch", 1, model)

    onnx.checker.check_model(model)
    assert {value.name: _shape(value) for value in model.graph.input} == {
        "g1_input": ["batch", 640],
        "proprioception": ["batch", 930],
    }
    assert {value.name: _shape(value) for value in model.graph.output} == {
        "action": ["batch", 29],
        "token": ["batch", 64],
    }
    encoder = next(
        node for node in model.graph.node if list(node.output)[:1] == ["encoder_obs_dict"]
    )
    decoder = next(
        node for node in model.graph.node if list(node.output)[:1] == ["decoder_obs_dict"]
    )
    action = next(
        node for node in model.graph.node if list(node.output)[:1] == ["action"]
    )
    assert list(encoder.input) == [
        "sonic_ll_mode_selector",
        "g1_input",
        "sonic_ll_zero_teleop",
        "sonic_ll_zero_smpl",
    ]
    assert list(decoder.input) == ["encoder_encoded_tokens", "proprioception"]
    assert action.name == "decoder_/Squeeze"
    token_shape = next(
        value
        for value in model.graph.initializer
        if value.name == "sonic_token_output_shape"
    )
    assert numpy_helper.to_array(token_shape).tolist() == [-1, 64]


def test_sonic_low_latency_transform_rejects_graph_drift() -> None:
    if not _LL_SOURCE.is_file():
        pytest.skip("SONIC low-latency source artifact is not installed")
    model = onnx.load(str(_LL_SOURCE))
    model.graph.name = "renamed"

    with pytest.raises(ValueError, match="graph name changed"):
        apply_transform("sonic_low_latency_g1_batch", 1, model)
