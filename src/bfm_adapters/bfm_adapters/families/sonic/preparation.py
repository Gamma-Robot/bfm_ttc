"""Offline batch transform for the released SONIC G1 policy graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bfm_adapters.preparation import register_transform


def _shape(value_info: Any) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _set_shape(value_info: Any, shape: Sequence[int | str]) -> None:
    import onnx

    tensor = value_info.type.tensor_type
    tensor.elem_type = onnx.TensorProto.FLOAT
    del tensor.shape.dim[:]
    for item in shape:
        dim = tensor.shape.dim.add()
        if isinstance(item, int):
            dim.dim_value = item
        else:
            dim.dim_param = item


def _replace_constant(
    node: Any,
    *,
    expected: Sequence[int],
    replacement: Sequence[int],
) -> None:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    if node.op_type != "Constant":
        raise ValueError(f"SONIC node {node.name!r} is not Constant")
    attribute = next(
        (
            item
            for item in node.attribute
            if item.type == onnx.AttributeProto.TENSOR
        ),
        None,
    )
    if attribute is None:
        raise ValueError(f"SONIC Constant {node.name!r} has no tensor value")
    actual = numpy_helper.to_array(attribute.t)
    if actual.tolist() != list(expected):
        raise ValueError(
            f"SONIC Constant {node.name!r} changed: "
            f"expected={list(expected)}, actual={actual.tolist()}"
        )
    attribute.t.CopyFrom(
        numpy_helper.from_array(
            np.asarray(replacement, dtype=actual.dtype),
            attribute.t.name,
        )
    )


@register_transform("sonic_release_g1_batch", 1)
def sonic_release_g1_batch(model: Any) -> None:
    """Promote the released, rank-one G1 graph to a true batch graph.

    This is intentionally tied to the released graph's named nodes. Any graph
    drift fails loudly instead of producing a plausible but wrong policy.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    nodes = {node.name: node for node in model.graph.node}
    if len(nodes) != len(model.graph.node):
        raise ValueError("SONIC released G1 graph has duplicate node names")

    inputs = {value.name: value for value in model.graph.input}
    expected_inputs = {"g1_input": [640], "proprioception": [930]}
    if set(inputs) != set(expected_inputs):
        raise ValueError(
            "SONIC released G1 inputs changed: "
            f"expected={sorted(expected_inputs)}, actual={sorted(inputs)}"
        )
    for name, expected in expected_inputs.items():
        if _shape(inputs[name]) != expected:
            raise ValueError(
                f"SONIC released G1 input {name!r} changed: "
                f"expected={expected}, actual={_shape(inputs[name])}"
            )
        _set_shape(inputs[name], ["batch", *expected])

    rewires = {
        "/Unsqueeze_2": ("/Unsqueeze_output_0", "g1_input"),
        "/Unsqueeze_3": ("/Unsqueeze_1_output_0", "proprioception"),
    }
    for name, (expected, replacement) in rewires.items():
        node = nodes.get(name)
        if node is None or node.op_type != "Unsqueeze" or not node.input:
            raise ValueError(f"SONIC released G1 graph is missing {name!r}")
        if node.input[0] != expected:
            raise ValueError(
                f"SONIC released G1 node {name!r} changed: "
                f"expected input {expected!r}, got {node.input[0]!r}"
            )
        node.input[0] = replacement

    reshape_constants = {
        "/Constant_8": ([1, 1, 10, 58], [-1, 1, 10, 58]),
        "/Constant_13": ([1, 1, 10, 6], [-1, 1, 10, 6]),
        "/g1/Constant": ([1, 640], [-1, 640]),
        "/g1/Constant_1": ([1, 2, 32], [-1, 2, 32]),
        "/quantizer/Constant": ([1, 2, 1, 32], [-1, 2, 1, 32]),
        "/quantizer/Constant_5": ([1, 2, 32], [-1, 2, 32]),
    }
    for name, (expected, replacement) in reshape_constants.items():
        try:
            node = nodes[name]
        except KeyError as error:
            raise ValueError(
                f"SONIC released G1 graph is missing {name!r}"
            ) from error
        _replace_constant(node, expected=expected, replacement=replacement)

    first_squeeze = nodes.get("/Squeeze")
    final_squeeze = nodes.get("/Squeeze_1")
    if (
        first_squeeze is None
        or first_squeeze.op_type != "Squeeze"
        or list(first_squeeze.output) != ["/Squeeze_output_0"]
        or final_squeeze is None
        or final_squeeze.op_type != "Squeeze"
        or list(final_squeeze.input[:1]) != ["/Squeeze_output_0"]
        or list(final_squeeze.output) != ["action"]
    ):
        raise ValueError("SONIC released G1 action output graph changed")
    first_squeeze.output[0] = "action"

    remove = {
        "/Constant",
        "/Unsqueeze",
        "/Constant_1",
        "/Unsqueeze_1",
        "/Constant_25",
        "/Squeeze_1",
    }
    missing = remove.difference(nodes)
    if missing:
        raise ValueError(
            "SONIC released G1 graph is missing removable nodes: "
            f"{sorted(missing)}"
        )
    retained = [node for node in model.graph.node if node.name not in remove]
    del model.graph.node[:]
    model.graph.node.extend(retained)

    initializers = {value.name: value for value in model.graph.initializer}
    try:
        token_shape = initializers["sonic_token_output_shape"]
    except KeyError as error:
        raise ValueError(
            "SONIC released G1 graph has no token output shape"
        ) from error
    token_shape_value = numpy_helper.to_array(token_shape)
    if token_shape_value.tolist() != [64]:
        raise ValueError(
            "SONIC released G1 token output shape changed: "
            f"{token_shape_value.tolist()}"
        )
    token_shape.CopyFrom(
        numpy_helper.from_array(
            np.asarray([-1, 64], dtype=token_shape_value.dtype),
            token_shape.name,
        )
    )

    outputs = {value.name: value for value in model.graph.output}
    if set(outputs) != {"action", "token"}:
        raise ValueError(
            "SONIC released G1 outputs changed: "
            f"{sorted(outputs)}"
        )
    _set_shape(outputs["action"], ["batch", 29])
    _set_shape(outputs["token"], ["batch", 64])

    io_names = {*inputs, *outputs}
    value_info = [
        value for value in model.graph.value_info if value.name not in io_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)


_LL_RESHAPE_CONSTANTS = {
    "encoder_/Constant_20": ([1, 1, 10, 58], [-1, 1, 10, 58]),
    "encoder_/Constant_25": ([1, 1, 6], [-1, 1, 6]),
    "encoder_/Constant_30": ([1, 1, 10, 6], [-1, 1, 10, 6]),
    "encoder_/Constant_35": ([1, 1, 240], [-1, 1, 240]),
    "encoder_/Constant_40": ([1, 1, 9], [-1, 1, 9]),
    "encoder_/Constant_45": ([1, 1, 12], [-1, 1, 12]),
    "encoder_/Constant_50": ([1, 1, 4, 72], [-1, 1, 4, 72]),
    "encoder_/Constant_55": ([1, 1, 4, 6], [-1, 1, 4, 6]),
    "encoder_/Constant_60": ([1, 1, 4, 6], [-1, 1, 4, 6]),
    "encoder_/g1/Constant": ([1, 640], [-1, 640]),
    "encoder_/g1/Constant_1": ([1, 2, 32], [-1, 2, 32]),
    "encoder_/quantizer/Constant": ([1, 2, 1, 32], [-1, 2, 1, 32]),
    "encoder_/quantizer/Constant_5": ([1, 2, 32], [-1, 2, 32]),
    "encoder_/teleop/Constant": ([1, 2, 32], [-1, 2, 32]),
    "encoder_/quantizer_1/Constant": ([1, 2, 1, 32], [-1, 2, 1, 32]),
    "encoder_/quantizer_1/Constant_5": ([1, 2, 32], [-1, 2, 32]),
    "encoder_/smpl/Constant": ([1, 336], [-1, 336]),
    "encoder_/smpl/Constant_1": ([1, 2, 32], [-1, 2, 32]),
    "encoder_/quantizer_2/Constant": ([1, 2, 1, 32], [-1, 2, 1, 32]),
    "encoder_/quantizer_2/Constant_5": ([1, 2, 32], [-1, 2, 32]),
}


def _named_nodes(model: Any) -> dict[str, Any]:
    named: dict[str, Any] = {}
    for node in model.graph.node:
        if not node.name:
            continue
        if node.name in named:
            raise ValueError(
                f"SONIC low-latency G1 graph has duplicate node name {node.name!r}"
            )
        named[node.name] = node
    return named


def _node_by_output(model: Any, name: str) -> Any:
    matches = [node for node in model.graph.node if list(node.output)[:1] == [name]]
    if len(matches) != 1:
        raise ValueError(
            f"SONIC low-latency G1 graph must contain one producer of {name!r}"
        )
    return matches[0]


def _initializer(model: Any, name: str) -> Any:
    matches = [value for value in model.graph.initializer if value.name == name]
    if len(matches) != 1:
        raise ValueError(f"SONIC low-latency G1 graph is missing initializer {name!r}")
    return matches[0]


def _require_vector(initializer: Any, name: str, expected: list[int] | list[float]) -> None:
    from onnx import numpy_helper

    actual = numpy_helper.to_array(initializer)
    if actual.tolist() != expected:
        raise ValueError(
            f"SONIC low-latency G1 initializer {name!r} changed: "
            f"expected={expected}, actual={actual.tolist()}"
        )


@register_transform("sonic_low_latency_g1_batch", 1)
def sonic_low_latency_g1_batch(model: Any) -> None:
    """Promote the fused low-latency G1 graph to a true batch graph.

    Wrappers are identified by output names because five fuse nodes are unnamed.
    Rank-1 fusion pads are Expanded to the runtime batch before Concat.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if model.graph.name != "sonic_low_latency_g1_policy":
        raise ValueError(
            f"SONIC low-latency G1 graph name changed: {model.graph.name!r}"
        )
    if model.producer_name != "bfm-ttc-source-export":
        raise ValueError(
            "SONIC low-latency G1 producer changed: "
            f"{model.producer_name!r}"
        )
    if len(model.graph.node) != 303:
        raise ValueError(
            "SONIC low-latency G1 node count changed: "
            f"expected=303, actual={len(model.graph.node)}"
        )

    inputs = {value.name: value for value in model.graph.input}
    expected_inputs = {"g1_input": [640], "proprioception": [930]}
    if set(inputs) != set(expected_inputs):
        raise ValueError(
            "SONIC low-latency G1 inputs changed: "
            f"expected={sorted(expected_inputs)}, actual={sorted(inputs)}"
        )
    for name, expected in expected_inputs.items():
        if _shape(inputs[name]) != expected:
            raise ValueError(
                f"SONIC low-latency G1 input {name!r} changed: "
                f"expected={expected}, actual={_shape(inputs[name])}"
            )
        _set_shape(inputs[name], ["batch", *expected])

    outputs = {value.name: value for value in model.graph.output}
    if set(outputs) != {"action", "token"}:
        raise ValueError(
            "SONIC low-latency G1 outputs changed: "
            f"{sorted(outputs)}"
        )
    if _shape(outputs["action"]) != [29] or _shape(outputs["token"]) != [64]:
        raise ValueError("SONIC low-latency G1 output ranks changed")

    g1_unsqueeze = _node_by_output(model, "mode_input_batched")
    proprio_unsqueeze = _node_by_output(model, "proprioception_batched")
    encoder_concat = _node_by_output(model, "encoder_obs_dict")
    decoder_concat = _node_by_output(model, "decoder_obs_dict")
    action_squeeze = _node_by_output(model, "action")
    if (
        g1_unsqueeze.op_type != "Unsqueeze"
        or list(g1_unsqueeze.input) != ["g1_input", "batch_axis"]
        or proprio_unsqueeze.op_type != "Unsqueeze"
        or list(proprio_unsqueeze.input) != ["proprioception", "batch_axis"]
        or encoder_concat.op_type != "Concat"
        or list(encoder_concat.input)
        != ["mode_selector", "mode_input_batched", "zero_teleop", "zero_smpl"]
        or decoder_concat.op_type != "Concat"
        or list(decoder_concat.input)
        != ["encoder_encoded_tokens", "proprioception_batched"]
        or action_squeeze.op_type != "Squeeze"
        or list(action_squeeze.input) != ["decoder_action", "batch_axis"]
    ):
        raise ValueError("SONIC low-latency G1 fuse wrappers changed")

    named = _named_nodes(model)
    decoder_squeeze = named.get("decoder_/Squeeze")
    squeeze_axes = named.get("decoder_/Constant_14")
    token = named.get("ExposeSonicToken")
    if (
        decoder_squeeze is None
        or decoder_squeeze.op_type != "Squeeze"
        or list(decoder_squeeze.output) != ["decoder_action"]
        or squeeze_axes is None
        or token is None
        or token.op_type != "Reshape"
        or list(token.input)
        != ["encoder_encoded_tokens", "sonic_token_output_shape"]
        or list(token.output) != ["token"]
    ):
        raise ValueError("SONIC low-latency G1 action/token graph changed")
    _replace_constant(squeeze_axes, expected=[1], replacement=[1])

    batch_axis = _initializer(model, "batch_axis")
    mode_selector = _initializer(model, "mode_selector")
    zero_teleop = _initializer(model, "zero_teleop")
    zero_smpl = _initializer(model, "zero_smpl")
    token_shape = _initializer(model, "sonic_token_output_shape")
    _require_vector(batch_axis, "batch_axis", [0])
    mode_value = numpy_helper.to_array(mode_selector)
    if mode_value.shape != (1, 4) or not np.array_equal(mode_value, np.zeros((1, 4))):
        raise ValueError("SONIC low-latency G1 mode_selector changed")
    teleop_value = numpy_helper.to_array(zero_teleop)
    smpl_value = numpy_helper.to_array(zero_smpl)
    if teleop_value.shape != (1, 267) or np.any(teleop_value):
        raise ValueError("SONIC low-latency G1 zero_teleop changed")
    if smpl_value.shape != (1, 336) or np.any(smpl_value):
        raise ValueError("SONIC low-latency G1 zero_smpl changed")
    token_shape_value = numpy_helper.to_array(token_shape)
    if token_shape_value.tolist() != [64]:
        raise ValueError(
            "SONIC low-latency G1 token output shape changed: "
            f"{token_shape_value.tolist()}"
        )

    for name, (expected, replacement) in _LL_RESHAPE_CONSTANTS.items():
        try:
            node = named[name]
        except KeyError as error:
            raise ValueError(
                f"SONIC low-latency G1 graph is missing {name!r}"
            ) from error
        _replace_constant(node, expected=expected, replacement=replacement)

    token_shape.CopyFrom(
        numpy_helper.from_array(
            np.asarray([-1, 64], dtype=token_shape_value.dtype),
            token_shape.name,
        )
    )

    decoder_squeeze.output[0] = "action"
    decoder_concat.input[1] = "proprioception"
    encoder_concat.input[:] = [
        "sonic_ll_mode_selector",
        "g1_input",
        "sonic_ll_zero_teleop",
        "sonic_ll_zero_smpl",
    ]

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([4], np.int64), "sonic_ll_mode_width"),
            numpy_helper.from_array(
                np.asarray([267], np.int64), "sonic_ll_teleop_width"
            ),
            numpy_helper.from_array(
                np.asarray([336], np.int64), "sonic_ll_smpl_width"
            ),
        ]
    )
    expand_nodes = [
        helper.make_node("Shape", ["g1_input"], ["sonic_ll_g1_shape"], "sonic_ll_g1_shape"),
        helper.make_node(
            "Gather",
            ["sonic_ll_g1_shape", "batch_axis"],
            ["sonic_ll_batch"],
            "sonic_ll_batch",
        ),
        helper.make_node(
            "Concat",
            ["sonic_ll_batch", "sonic_ll_mode_width"],
            ["sonic_ll_mode_shape"],
            "sonic_ll_mode_shape",
            axis=0,
        ),
        helper.make_node(
            "Concat",
            ["sonic_ll_batch", "sonic_ll_teleop_width"],
            ["sonic_ll_teleop_shape"],
            "sonic_ll_teleop_shape",
            axis=0,
        ),
        helper.make_node(
            "Concat",
            ["sonic_ll_batch", "sonic_ll_smpl_width"],
            ["sonic_ll_smpl_shape"],
            "sonic_ll_smpl_shape",
            axis=0,
        ),
        helper.make_node(
            "Expand",
            ["mode_selector", "sonic_ll_mode_shape"],
            ["sonic_ll_mode_selector"],
            "sonic_ll_expand_mode",
        ),
        helper.make_node(
            "Expand",
            ["zero_teleop", "sonic_ll_teleop_shape"],
            ["sonic_ll_zero_teleop"],
            "sonic_ll_expand_teleop",
        ),
        helper.make_node(
            "Expand",
            ["zero_smpl", "sonic_ll_smpl_shape"],
            ["sonic_ll_zero_smpl"],
            "sonic_ll_expand_smpl",
        ),
    ]
    remove = {id(g1_unsqueeze), id(proprio_unsqueeze), id(action_squeeze)}
    retained = [node for node in model.graph.node if id(node) not in remove]
    del model.graph.node[:]
    model.graph.node.extend(expand_nodes)
    model.graph.node.extend(retained)

    _set_shape(outputs["action"], ["batch", 29])
    _set_shape(outputs["token"], ["batch", 64])
    io_names = {*inputs, *outputs}
    value_info = [
        value for value in model.graph.value_info if value.name not in io_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)


__all__ = ["sonic_release_g1_batch", "sonic_low_latency_g1_batch"]
