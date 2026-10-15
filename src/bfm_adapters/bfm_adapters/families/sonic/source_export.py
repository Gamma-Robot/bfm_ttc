"""Reconstruct native-G1 SONIC policies from pinned official ONNX pairs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# nvidia/GEAR-SONIC, revision 0af0cb7568257b04a8569b9c76f326e707b122fc.
_SOURCE_SHA256 = (
    "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
)
# nvidia/GEAR-SONIC, revision 183ed69145708878ee68cc8791848ee66f233d24,
# low_latency/model_{encoder,decoder}.onnx.
_LOW_LATENCY_SOURCE_SHA256 = (
    "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b",
    "c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab",
)


def _tensor(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _clear_doc_strings(message) -> None:
    if "doc_string" in message.DESCRIPTOR.fields_by_name:
        message.ClearField("doc_string")
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.is_repeated else [value]:
                _clear_doc_strings(child)


def _check_interface(model: onnx.ModelProto, width: int, output: str, output_width: int) -> None:
    expected = [_tensor("obs_dict", [1, width]), _tensor(output, [1, output_width])]
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("SONIC source interface must have one input and one output")
    for actual, reference in zip([model.graph.input[0], model.graph.output[0]], expected):
        if actual.name != reference.name or actual.type != reference.type:
            raise ValueError("SONIC source interface does not match the expected variant")


def export_sonic_release(encoder_path: Path, decoder_path: Path, destination: Path) -> None:
    """Select the original release's G1 encoder, retain quantization, and decode.

    The named wrapper topology remains compatible with sonic_release_g1_batch.
    Source weights and operators are reused, and export stack traces are removed.
    Low-latency and SONIC v1.1 inputs are deliberately rejected by source identity.
    """
    models = []
    for path, expected in zip([encoder_path, decoder_path], _SOURCE_SHA256):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SONIC release source identity mismatch: {path}")
        models.append(onnx.load_model_from_string(data))
    encoder, decoder = models
    _check_interface(encoder, 1762, "encoded_tokens", 64)
    _check_interface(decoder, 994, "action", 29)

    nodes: list[onnx.NodeProto] = []

    def node(name: str, kind: str, inputs: list[str], output: str | None = None, **attributes) -> str:
        output = output or name + "_output_0"
        nodes.append(helper.make_node(kind, inputs, [output], name=name, **attributes))
        return output

    def constant(index: int, values: list[int]) -> str:
        name = "/Constant" + (f"_{index}" if index else "")
        return node(name, "Constant", [], value=numpy_helper.from_array(np.asarray(values, np.int64)))

    g1 = node("/Unsqueeze", "Unsqueeze", ["g1_input", constant(0, [0])])
    proprio = node("/Unsqueeze_1", "Unsqueeze", ["proprioception", constant(1, [0])])
    g1 = node("/Unsqueeze_2", "Unsqueeze", [g1, constant(2, [1])])
    proprio = node("/Unsqueeze_3", "Unsqueeze", [proprio, constant(3, [1])])
    # Public universal input uses command at [4:584] and orientation at [601:661].
    # The semantic input removes inactive fields; interleave per-frame features
    # exactly as the selected G1 encoder does before its 640-wide flatten.
    axis, start, end, step = [constant(i, v) for i, v in zip(range(4, 8), [[2], [0], [580], [1]])]
    command = node("/Slice", "Slice", [g1, start, end, axis, step])
    command = node("/Reshape", "Reshape", [command, constant(8, [1, 1, 10, 58])])
    axis, start, end, step = [constant(i, v) for i, v in zip(range(9, 13), [[2], [580], [640], [1]])]
    orientation = node("/Slice_1", "Slice", [g1, start, end, axis, step])
    orientation = node("/Reshape_1", "Reshape", [orientation, constant(13, [1, 1, 10, 6])])
    frames = node("/Concat", "Concat", [command, orientation], axis=-1)
    node("/Reshape_2", "Reshape", [frames, constant(14, [-1, 10, 64])])

    selected = [deepcopy(n) for n in encoder.graph.node if n.name.startswith(("/g1/", "/quantizer/"))]
    for item in selected:
        for i, name in enumerate(item.input):
            if name == "/Reshape_10_output_0":
                item.input[i] = "/Reshape_2_output_0"
    nodes.extend(selected)
    quantized = node("/Unsqueeze_4", "Unsqueeze", ["/quantizer/Cast_output_0", constant(15, [1])])
    shape = node("/Shape", "Shape", [quantized])
    axis, start, end = [constant(i, v) for i, v in zip(range(16, 19), [[0], [0], [-2]])]
    prefix = node("/Slice_2", "Slice", [shape, start, end, axis])
    shape = node("/Concat_1", "Concat", [prefix, constant(19, [-1])], axis=0)
    token = node("/Reshape_3", "Reshape", [quantized, shape])
    node("/Concat_2", "Concat", [token, proprio], axis=-1)

    renames = {"/Concat_output_0": "/Concat_2_output_0", "action": "/Squeeze_output_0"}
    for old, new in [(f"/Constant_{i}", f"/Constant_{i + 10}") for i in range(10, 15)] + [("/Slice_2", "/Slice_3")]:
        renames[old] = new
        renames[old + "_output_0"] = new + "_output_0"
    renames.update({f"onnx::MatMul_{136 + i}": f"onnx::MatMul_{289 + i}" for i in range(7)})
    start = next(i for i, n in enumerate(decoder.graph.node) if n.name == "/g1_dyn/module/module.0/MatMul")
    for original in decoder.graph.node[start:]:
        item = deepcopy(original)
        item.name = renames.get(item.name, item.name)
        for names in [item.input, item.output]:
            for i, name in enumerate(names):
                names[i] = renames.get(name, name)
        nodes.append(item)
    node("/Squeeze_1", "Squeeze", ["/Squeeze_output_0", constant(25, [0])], "action")
    node("ExposeSonicToken", "Reshape", [token, "sonic_token_output_shape"], "token")

    used = {name for item in nodes for name in item.input}
    weights = []
    for model in [encoder, decoder]:
        for original in model.graph.initializer:
            item = deepcopy(original)
            if model is decoder:
                item.name = renames.get(item.name, item.name)
            if item.name in used:
                weights.append(item)
    weights.append(helper.make_tensor("sonic_token_output_shape", TensorProto.INT64, [1], [64]))
    graph = helper.make_graph(nodes, "main_graph", [_tensor("g1_input", [640]), _tensor("proprioception", [930])],
                              [_tensor("action", [29]), _tensor("token", [64])], weights)
    model = helper.make_model(graph, producer_name="bfm-ttc-source-export", ir_version=7,
                              opset_imports=[helper.make_opsetid("", 13)])
    _clear_doc_strings(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def export_sonic_low_latency(encoder_path: Path, decoder_path: Path, destination: Path) -> None:
    """Compose the official low-latency pair with G1 mode and inactive zero pads.

    The universal encoder and decoder retain their operators and weights. The
    wrapper exposes the native family interface and the quantized 64-wide token.
    """
    from onnx.compose import add_prefix

    models = []
    for path, expected in zip([encoder_path, decoder_path], _LOW_LATENCY_SOURCE_SHA256):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SONIC low-latency source identity mismatch: {path}")
        models.append(onnx.load_model_from_string(data))
    encoder, decoder = models
    _check_interface(encoder, 1247, "encoded_tokens", 64)
    _check_interface(decoder, 994, "action", 29)
    encoder = add_prefix(encoder, "encoder_")
    decoder = add_prefix(decoder, "decoder_")
    nodes = [
        helper.make_node("Unsqueeze", ["g1_input", "batch_axis"], ["mode_input_batched"]),
        helper.make_node("Unsqueeze", ["proprioception", "batch_axis"], ["proprioception_batched"]),
        helper.make_node("Concat", ["mode_selector", "mode_input_batched", "zero_teleop", "zero_smpl"], ["encoder_obs_dict"], axis=1),
        *encoder.graph.node,
        helper.make_node("Concat", ["encoder_encoded_tokens", "proprioception_batched"], ["decoder_obs_dict"], axis=1),
        *decoder.graph.node,
        helper.make_node("Squeeze", ["decoder_action", "batch_axis"], ["action"]),
        helper.make_node("Reshape", ["encoder_encoded_tokens", "sonic_token_output_shape"], ["token"], name="ExposeSonicToken"),
    ]
    weights = [*encoder.graph.initializer, *decoder.graph.initializer]
    for name, value in [
        ("batch_axis", np.asarray([0], np.int64)),
        ("mode_selector", np.zeros((1, 4), np.float32)),
        ("zero_teleop", np.zeros((1, 267), np.float32)),
        ("zero_smpl", np.zeros((1, 336), np.float32)),
        ("sonic_token_output_shape", np.asarray([64], np.int64)),
    ]:
        weights.append(numpy_helper.from_array(value, name))
    graph = helper.make_graph(nodes, "sonic_low_latency_g1_policy",
                              [_tensor("g1_input", [640]), _tensor("proprioception", [930])],
                              [_tensor("action", [29]), _tensor("token", [64])], weights)
    model = helper.make_model(graph, producer_name="bfm-ttc-source-export", ir_version=7,
                              opset_imports=[helper.make_opsetid("", 13)])
    _clear_doc_strings(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("decoder", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", choices=["release", "low-latency"], default="release")
    args = parser.parse_args()
    export = export_sonic_low_latency if args.variant == "low-latency" else export_sonic_release
    export(args.encoder, args.decoder, args.destination)


if __name__ == "__main__":
    main()
