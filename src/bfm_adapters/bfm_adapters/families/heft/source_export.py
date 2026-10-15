"""Expose semantic inputs on the official HEFT G1 PMG deployment graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


_INPUTS = {"context": 1, "motion_command": 114, "target_motion": 806, "proprioception": 808}
_OUTPUTS = {
    "policy_1": [1, 1729], "priv_pred": [1, 256],
    "_actor_inp": [1, 1985], "_actor_feature": [1, 512],
    "action": [1, 29], "scale": [1, 29],
    "action_alias_6": [1, 29], "action_log_prob": [1],
}


def _is_float_shape(value: onnx.ValueInfoProto, shape: list[int]) -> bool:
    tensor = value.type.tensor_type
    return tensor.elem_type == TensorProto.FLOAT and [
        dim.dim_value if dim.HasField("dim_value") else None
        for dim in tensor.shape.dim
    ] == shape


def _clear_export_metadata(message) -> None:
    """Remove exporter annotations, including nested local stack traces."""
    for field, value in list(message.ListFields()):
        if field.name in {"doc_string", "metadata_props"}:
            message.ClearField(field.name)
        elif field.type == field.TYPE_MESSAGE:
            for child in value if field.is_repeated else [value]:
                _clear_export_metadata(child)


def export_heft_policy(source: Path, destination: Path) -> None:
    """Load the source and adjacent external tensors, then save one ONNX file.

    This preserves normalization, weights, outputs and all existing operators.
    The runtime verifies the resulting graph against the artifact lock.
    """
    model = onnx.load(source, load_external_data=True)
    inputs = model.graph.input
    if len(inputs) != 1 or inputs[0].name != "policy" or not _is_float_shape(inputs[0], [1, 1729]):
        raise ValueError("HEFT source must have exactly one float32 policy[1,1729] input")
    outputs = model.graph.output
    if len(outputs) != len(_OUTPUTS) or any(
        value.name not in _OUTPUTS or not _is_float_shape(value, _OUTPUTS[value.name])
        for value in outputs
    ) or {value.name for value in outputs} != set(_OUTPUTS):
        raise ValueError("HEFT source does not match the G1 PMG output interface")
    onnx.checker.check_model(model)
    del inputs[:]
    inputs.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width])
                  for name, width in _INPUTS.items())
    model.graph.node.insert(0, helper.make_node(
        "Concat", list(_INPUTS), ["policy"], name="assemble_policy", axis=1,
    ))
    onnx.checker.check_model(model)
    _clear_export_metadata(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official policy.onnx with adjacent policy.onnx.data")
    parser.add_argument("destination", type=Path, help="Output single-file semantic-input ONNX")
    args = parser.parse_args()
    export_heft_policy(args.source, args.destination)


if __name__ == "__main__":
    main()
