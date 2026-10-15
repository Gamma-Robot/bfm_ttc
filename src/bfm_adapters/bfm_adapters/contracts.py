"""BFM-neutral values exchanged with the command-optimization application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np

NamedTensors: TypeAlias = Mapping[str, np.ndarray]
BFMStateBatch: TypeAlias = object
MOTION_COMMAND_ARRAY_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)


def _names(values: tuple[str, ...], field: str) -> None:
    if not values or any(not name for name in values) or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique, non-empty names")


def _array(value: np.ndarray, shape: tuple[int, ...], field: str) -> None:
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{field} must be a floating-point numpy array")
    if value.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {value.shape}")


def _batch_size(value: np.ndarray, field: str) -> int:
    if not isinstance(value, np.ndarray) or value.ndim == 0:
        raise TypeError(f"{field} must be a batched numpy array")
    if value.shape[0] < 1:
        raise ValueError(f"{field} batch must be non-empty")
    return value.shape[0]


@dataclass(frozen=True, eq=False)
class RobotState:
    """Batched physical state; world-frame quaternions use ``(w, x, y, z)``."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_orientation_world: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        _names(self.joint_names, "joint_names")
        _names(self.body_names, "body_names")
        batch = _batch_size(self.joint_position, "joint_position")
        joints, bodies = len(self.joint_names), len(self.body_names)
        _array(self.joint_position, (batch, joints), "joint_position")
        _array(self.joint_velocity, (batch, joints), "joint_velocity")
        _array(self.body_position_world, (batch, bodies, 3), "body_position_world")
        _array(
            self.body_orientation_world,
            (batch, bodies, 4),
            "body_orientation_world",
        )
        _array(
            self.body_linear_velocity_world,
            (batch, bodies, 3),
            "body_linear_velocity_world",
        )
        _array(
            self.body_angular_velocity_world,
            (batch, bodies, 3),
            "body_angular_velocity_world",
        )


@dataclass(frozen=True, eq=False)
class MotionCommand:
    """Batched G1 targets at ordered Control-Step offsets."""

    query_offsets: tuple[int, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_orientation_world: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        if all(isinstance(offset, (int, np.integer)) for offset in self.query_offsets):
            object.__setattr__(
                self, "query_offsets", tuple(int(offset) for offset in self.query_offsets)
            )
        if (
            not self.query_offsets
            or len(set(self.query_offsets)) != len(self.query_offsets)
            or any(not isinstance(offset, int) for offset in self.query_offsets)
        ):
            raise ValueError("query_offsets must contain unique integer offsets")
        _names(self.joint_names, "joint_names")
        _names(self.body_names, "body_names")
        batch = _batch_size(self.joint_position, "joint_position")
        queries = len(self.query_offsets)
        joints, bodies = len(self.joint_names), len(self.body_names)
        _array(self.joint_position, (batch, queries, joints), "joint_position")
        _array(self.joint_velocity, (batch, queries, joints), "joint_velocity")
        _array(
            self.body_position_world,
            (batch, queries, bodies, 3),
            "body_position_world",
        )
        _array(
            self.body_orientation_world,
            (batch, queries, bodies, 4),
            "body_orientation_world",
        )
        _array(
            self.body_linear_velocity_world,
            (batch, queries, bodies, 3),
            "body_linear_velocity_world",
        )
        _array(
            self.body_angular_velocity_world,
            (batch, queries, bodies, 3),
            "body_angular_velocity_world",
        )


@dataclass(frozen=True, eq=False)
class PDCommand:
    """Batched command in canonical G1 joint order."""

    joint_names: tuple[str, ...]
    position_target: np.ndarray
    velocity_target: np.ndarray
    feedforward_torque: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    def __post_init__(self) -> None:
        _names(self.joint_names, "joint_names")
        batch = _batch_size(self.position_target, "position_target")
        shape = (batch, len(self.joint_names))
        for field in (
            "position_target",
            "velocity_target",
            "feedforward_torque",
            "proportional_gain",
            "derivative_gain",
        ):
            _array(getattr(self, field), shape, field)


@dataclass(frozen=True, eq=False)
class DevicePDCommand:
    """Borrowed device command buffers consumed before the next Control Step."""

    joint_names: tuple[str, ...]
    position_target: object
    velocity_target: object
    feedforward_torque: object
    proportional_gain: object
    derivative_gain: object

    def __post_init__(self) -> None:
        _names(self.joint_names, "joint_names")


@dataclass(frozen=True)
class TensorSpec:
    """One named policy tensor, excluding its leading rollout batch axis."""

    name: str
    dtype: str
    non_batch_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tensor name must be non-empty")
        try:
            np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid tensor dtype {self.dtype!r}") from exc
        object.__setattr__(self, "non_batch_shape", tuple(self.non_batch_shape))
        if any(not isinstance(size, int) or size < 1 for size in self.non_batch_shape):
            raise ValueError("non_batch_shape dimensions must be positive integers")
