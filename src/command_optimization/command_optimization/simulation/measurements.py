"""BFM-neutral per-Control-Step simulation measurements.

`MeasurementBuffer` is the CPU reference of the collection seam: only the
primitives declared by the active objective are allocated and copied. Device
collection arrives later behind the same ``ControlStepMeasurements`` currency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from bfm_adapters import RobotState


class MeasurementPrimitive(str, Enum):
    """One per-Control-Step quantity required by an objective."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    BODY_POSITION_WORLD = "body_position_world"
    BODY_ORIENTATION_WORLD = "body_orientation_world"


def _trailing(
    primitive: MeasurementPrimitive, joints: int, bodies: int
) -> tuple[int, ...]:
    return {
        MeasurementPrimitive.JOINT_POSITION: (joints,),
        MeasurementPrimitive.JOINT_VELOCITY: (joints,),
        MeasurementPrimitive.BODY_POSITION_WORLD: (bodies, 3),
        MeasurementPrimitive.BODY_ORIENTATION_WORLD: (bodies, 4),
    }[primitive]


def _names(values: tuple[str, ...], field: str) -> None:
    if not values or any(not name for name in values) or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique, non-empty names")


@dataclass(frozen=True, eq=False)
class ControlStepMeasurements:
    """Finite `(B, T, ...)` arrays for exactly the primitives a collection declared."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    arrays: Mapping[MeasurementPrimitive, np.ndarray]

    def __post_init__(self) -> None:
        _names(self.joint_names, "joint_names")
        _names(self.body_names, "body_names")
        if not self.arrays:
            raise ValueError("arrays must contain at least one primitive")
        normalized: dict[MeasurementPrimitive, np.ndarray] = {}
        lead: tuple[int, int] | None = None
        for key, value in self.arrays.items():
            primitive = MeasurementPrimitive(key)
            if not isinstance(value, np.ndarray) or not np.issubdtype(
                value.dtype, np.floating
            ):
                raise TypeError(f"{primitive.value} must be a floating-point numpy array")
            if value.ndim < 2:
                raise ValueError(f"{primitive.value} must be batched over (B, T)")
            if lead is None:
                lead = value.shape[:2]
                if lead[0] < 1 or lead[1] < 1:
                    raise ValueError(f"batch and num_steps must be >= 1, got {lead}")
            expected = (
                *lead,
                *_trailing(primitive, len(self.joint_names), len(self.body_names)),
            )
            if value.shape != expected:
                raise ValueError(
                    f"{primitive.value} must have shape {expected}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{primitive.value} must be finite")
            normalized[primitive] = value
        object.__setattr__(self, "arrays", MappingProxyType(normalized))

    @property
    def batch(self) -> int:
        return next(iter(self.arrays.values())).shape[0]

    @property
    def num_steps(self) -> int:
        return next(iter(self.arrays.values())).shape[1]


class MeasurementBuffer:
    """Preallocated `(B, T, ...)` storage for exactly the required primitives."""

    def __init__(
        self,
        requires: frozenset[MeasurementPrimitive],
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        batch: int,
        num_steps: int,
    ) -> None:
        if not requires:
            raise ValueError("requires must declare at least one primitive")
        self._joint_names = tuple(joint_names)
        self._body_names = tuple(body_names)
        _names(self._joint_names, "joint_names")
        _names(self._body_names, "body_names")
        if batch < 1 or num_steps < 1:
            raise ValueError(
                f"batch and num_steps must be >= 1, got ({batch}, {num_steps})"
            )
        self._batch = batch
        self._num_steps = num_steps
        joints, bodies = len(self._joint_names), len(self._body_names)
        self._arrays = {
            primitive: np.zeros((batch, num_steps, *_trailing(primitive, joints, bodies)))
            for primitive in (MeasurementPrimitive(p) for p in requires)
        }
        self._recorded = np.zeros(num_steps, dtype=bool)

    def record(self, step_index: int, state: RobotState) -> None:
        """Copy ``state`` into one Control Step index exactly once."""
        if not 0 <= step_index < self._num_steps:
            raise ValueError(
                f"step_index must be in [0, {self._num_steps}), got {step_index}"
            )
        if self._recorded[step_index]:
            raise ValueError(f"step_index {step_index} already recorded")
        if (
            state.joint_names != self._joint_names
            or state.body_names != self._body_names
        ):
            raise ValueError("state joint/body names must match the buffer's names")
        if state.joint_position.shape[0] != self._batch:
            raise ValueError(
                f"state batch must be {self._batch}, got {state.joint_position.shape[0]}"
            )
        for primitive, storage in self._arrays.items():
            storage[:, step_index] = getattr(state, primitive.value)
        self._recorded[step_index] = True

    def finalize(self) -> ControlStepMeasurements:
        """Package the completed collection; every step must have been recorded."""
        if not self._recorded.all():
            missing = np.flatnonzero(~self._recorded).tolist()
            raise ValueError(f"cannot finalize with unrecorded steps {missing}")
        return ControlStepMeasurements(self._joint_names, self._body_names, self._arrays)
