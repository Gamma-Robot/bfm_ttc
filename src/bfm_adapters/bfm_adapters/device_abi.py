"""Shared validation for borrowed device bfm arrays and state views."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bfm_adapters.contracts import DevicePDCommand


_COMMAND_ARRAY_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)


def validate_device_layout(
    state_view: object,
    command_view: object,
    *,
    family: str,
    command_joint_names: tuple[str, ...],
    float_dtype: object,
    body_dtypes: Mapping[str, object],
) -> tuple[int, object]:
    """Validate one CUDA state/command binding and return its batch and device."""

    if not isinstance(command_view, DevicePDCommand):
        raise TypeError("command_view must be a DevicePDCommand")
    joint_position = getattr(state_view, "joint_position", None)
    try:
        batch_size = int(joint_position.shape[0])
        device = joint_position.device
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("state_view must expose a batched Warp joint array") from exc
    if batch_size < 1 or not getattr(device, "is_cuda", False):
        raise ValueError(f"{family} device bfm requires a non-empty CUDA batch")

    joints = len(tuple(getattr(state_view, "joint_names")))
    bodies = len(tuple(getattr(state_view, "body_names")))
    validate_warp_array(
        getattr(state_view, "joint_position"),
        (batch_size, joints),
        float_dtype,
        device,
        "joint_position",
    )
    validate_warp_array(
        getattr(state_view, "joint_velocity"),
        (batch_size, joints),
        float_dtype,
        device,
        "joint_velocity",
    )
    for name, dtype in body_dtypes.items():
        validate_warp_array(
            getattr(state_view, name),
            (batch_size, bodies),
            dtype,
            device,
            name,
        )

    if command_view.joint_names != command_joint_names:
        raise ValueError(
            f"command_view joint names differ from {family} command names"
        )
    command_shape = (batch_size, len(command_joint_names))
    for name in _COMMAND_ARRAY_FIELDS:
        validate_warp_array(
            getattr(command_view, name),
            command_shape,
            float_dtype,
            device,
            f"command_view.{name}",
        )
    return batch_size, device


def validate_bound_state_view(
    state_view: object, bound_state_view: object, *, family: str
) -> None:
    """Require the exact state view whose device buffers were bound at construction."""

    if state_view is not bound_state_view:
        raise ValueError(f"{family} device bfm is bound to one RobotStateView")


def validate_warp_array(
    value: object,
    shape: tuple[int, ...],
    dtype: object,
    device: object,
    name: str,
) -> None:
    """Validate the common borrowed Warp-array ABI without importing Warp."""

    if (
        getattr(value, "shape", None) != shape
        or getattr(value, "dtype", None) != dtype
        or getattr(value, "device", None) != device
        or not bool(getattr(value, "is_contiguous", False))
    ):
        raise ValueError(
            f"{name} must be a contiguous {dtype} Warp array with shape "
            f"{shape} on {device}"
        )


def normalize_bfm_state(
    state: object,
    template: Mapping[str, np.ndarray],
    *,
    family: str,
) -> dict[str, np.ndarray]:
    """Validate family-owned state and return contiguous arrays in template dtypes."""

    if not isinstance(state, dict):
        raise TypeError(f"{family} bfm state must be a dictionary")
    if set(state) != set(template):
        raise ValueError(
            f"{family} bfm state fields differ from the family contract"
        )
    values: dict[str, np.ndarray] = {}
    for name, reference in template.items():
        value = np.asarray(state[name])
        if value.shape != reference.shape:
            raise ValueError(
                f"{family} bfm state {name} has shape {value.shape}, "
                f"expected {reference.shape}"
            )
        values[name] = np.ascontiguousarray(value, dtype=reference.dtype)
    return values
