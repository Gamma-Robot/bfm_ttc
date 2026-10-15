from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from bfm_adapters.contracts import DevicePDCommand
from bfm_adapters.device_abi import (
    normalize_bfm_state,
    validate_bound_state_view,
    validate_device_layout,
)


_FLOAT32 = object()
_VEC3 = object()
_QUAT = object()
_COMMAND_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)


@dataclass(frozen=True)
class _Device:
    is_cuda: bool = True

    def __str__(self) -> str:
        return "cuda:test"


@dataclass
class _Array:
    shape: tuple[int, ...]
    dtype: object
    device: _Device
    is_contiguous: bool = True


def _binding() -> tuple[SimpleNamespace, DevicePDCommand, _Device]:
    device = _Device()
    state_view = SimpleNamespace(
        joint_names=("j0", "j1"),
        body_names=("root",),
        joint_position=_Array((3, 2), _FLOAT32, device),
        joint_velocity=_Array((3, 2), _FLOAT32, device),
        body_orientation_world=_Array((3, 1), _QUAT, device),
        body_angular_velocity_world=_Array((3, 1), _VEC3, device),
    )
    arrays = {
        name: _Array((3, 2), _FLOAT32, device) for name in _COMMAND_FIELDS
    }
    command = DevicePDCommand(joint_names=("j0", "j1"), **arrays)
    return state_view, command, device


def _validate(state_view: object, command_view: object) -> tuple[int, object]:
    return validate_device_layout(
        state_view,
        command_view,
        family="TEST",
        command_joint_names=("j0", "j1"),
        float_dtype=_FLOAT32,
        body_dtypes={
            "body_orientation_world": _QUAT,
            "body_angular_velocity_world": _VEC3,
        },
    )


def test_validate_device_layout_accepts_declared_state_and_command_abi() -> None:
    state_view, command, device = _binding()

    assert _validate(state_view, command) == (3, device)


def test_validate_device_layout_rejects_invalid_command_binding() -> None:
    state_view, command, _ = _binding()

    with pytest.raises(TypeError, match="DevicePDCommand"):
        _validate(state_view, object())
    with pytest.raises(ValueError, match="TEST command names"):
        _validate(state_view, replace(command, joint_names=("other",)))
    command.position_target.is_contiguous = False
    with pytest.raises(ValueError, match="command_view.position_target"):
        _validate(state_view, command)


def test_validate_device_layout_checks_only_declared_family_body_fields() -> None:
    state_view, command, _ = _binding()
    state_view.body_angular_velocity_world.dtype = _QUAT

    with pytest.raises(ValueError, match="body_angular_velocity_world"):
        _validate(state_view, command)
    state_view.body_angular_velocity_world.dtype = _VEC3
    state_view.unused_body_field = object()
    assert _validate(state_view, command)[0] == 3


def test_bound_view_and_bfm_state_validation_preserve_family_errors() -> None:
    state_view, _, _ = _binding()
    validate_bound_state_view(state_view, state_view, family="TEST")
    with pytest.raises(ValueError, match="TEST device bfm is bound"):
        validate_bound_state_view(object(), state_view, family="TEST")

    template = {"history": np.zeros((2, 3), dtype=np.float32)}
    source = np.asfortranarray(np.ones((2, 3), dtype=np.float64))
    normalized = normalize_bfm_state(
        {"history": source}, template, family="TEST"
    )
    assert normalized["history"].dtype == np.float32
    assert normalized["history"].flags.c_contiguous

    with pytest.raises(TypeError, match="TEST bfm state"):
        normalize_bfm_state(object(), template, family="TEST")
    with pytest.raises(ValueError, match="fields differ"):
        normalize_bfm_state({}, template, family="TEST")
    with pytest.raises(ValueError, match="has shape"):
        normalize_bfm_state(
            {"history": np.zeros((1, 3))}, template, family="TEST"
        )
