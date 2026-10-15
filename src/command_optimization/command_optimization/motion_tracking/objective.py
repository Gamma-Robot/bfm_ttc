"""Finite-horizon motion-tracking objective.

The immutable reference_motion motion_command defines the task. Candidate motion_command is the
optimizer's deformation of that task, and executed_trajectory measurements are produced by
closed-loop simulation. Keeping all three explicit prevents candidate tracking
from being mistaken for task fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import ClassVar, Mapping

import numpy as np

from bfm_adapters import (
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
)
from command_optimization.motion_tracking.metrics import _quaternion_geodesic
from command_optimization.simulation.measurements import MeasurementPrimitive, ControlStepMeasurements

def _readonly_vector(value: np.ndarray, field: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{field} must have shape (B,) with B >= 1, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class CostResult:
    """Immutable full-horizon costs and weighted per-term contributions."""

    total: np.ndarray
    terms: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        total = _readonly_vector(self.total, "total")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.terms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("term names must be non-empty strings")
            term = _readonly_vector(value, f"terms[{name!r}]")
            if term.shape != total.shape:
                raise ValueError(
                    f"terms[{name!r}] must have shape {total.shape}, got {term.shape}"
                )
            normalized[name] = term
        if not normalized:
            raise ValueError("terms must contain at least one contribution")
        term_total = np.sum(np.stack(tuple(normalized.values()), axis=0), axis=0)
        if not np.allclose(total, term_total, rtol=1e-12, atol=1e-12):
            raise ValueError("total must equal the sum of all term contributions")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "terms", MappingProxyType(normalized))


@dataclass(frozen=True)
class MotionTrackingObjective:
    """Weighted task fidelity, candidate deformation, and terminal fidelity.

    Running terms are summed over the fixed planning horizon. Terminal terms
    are evaluated once at its last Control Step. All returned costs are float64 and
    lower-is-better.
    """

    body_names: tuple[str, ...]
    tracking_joint_position_weight: float
    tracking_joint_velocity_weight: float
    tracking_body_position_weight: float
    tracking_body_orientation_weight: float
    deformation_joint_position_weight: float
    deformation_body_position_weight: float
    deformation_body_orientation_weight: float
    joint_speed_weight: float
    terminal_joint_position_weight: float
    terminal_body_position_weight: float

    requires: ClassVar[frozenset[MeasurementPrimitive]] = frozenset(
        MeasurementPrimitive
    )

    def __post_init__(self) -> None:
        if isinstance(self.body_names, str):
            raise ValueError("body_names must be a sequence of task body names")
        names = tuple(self.body_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("body_names must contain unique, non-empty task body names")
        object.__setattr__(self, "body_names", names)
        for field in fields(self):
            if field.name == "body_names":
                continue
            value = float(getattr(self, field.name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field.name} must be a finite weight >= 0, got {value}"
                )
            object.__setattr__(self, field.name, value)

    def body_indices(self, available_names: tuple[str, ...]) -> list[int]:
        """Resolve the task subset in a shared policy/measurement body layout."""
        return [available_names.index(name) for name in self.body_names]

    def evaluate(
        self,
        measurements: ControlStepMeasurements,
        reference_motion: MotionCommand,
        candidate: MotionCommand,
    ) -> CostResult:
        """Reduce one full rollout per candidate to a cost vector of shape ``(B,)``."""
        self._validate(measurements, reference_motion, candidate)
        bodies = self.body_indices(measurements.body_names)

        executed_trajectory_joint_position = np.asarray(
            measurements.arrays[MeasurementPrimitive.JOINT_POSITION],
            dtype=np.float64,
        )
        executed_trajectory_joint_velocity = np.asarray(
            measurements.arrays[MeasurementPrimitive.JOINT_VELOCITY],
            dtype=np.float64,
        )
        executed_trajectory_body_position = np.asarray(
            measurements.arrays[MeasurementPrimitive.BODY_POSITION_WORLD],
            dtype=np.float64,
        )
        executed_trajectory_body_orientation = np.asarray(
            measurements.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD],
            dtype=np.float64,
        )

        deformation = _deformation_terms(self, reference_motion, candidate)
        terms = {
            "tracking/joint_position": self.tracking_joint_position_weight
            * _joint_error(executed_trajectory_joint_position, reference_motion.joint_position).sum(axis=1),
            "tracking/joint_velocity": self.tracking_joint_velocity_weight
            * _joint_error(executed_trajectory_joint_velocity, reference_motion.joint_velocity).sum(axis=1),
            "tracking/body_position": self.tracking_body_position_weight
            * _body_position_error(
                executed_trajectory_body_position[:, :, bodies],
                reference_motion.body_position_world[:, :, bodies],
            ).sum(axis=1),
            "tracking/body_orientation": self.tracking_body_orientation_weight
            * _body_orientation_error(
                executed_trajectory_body_orientation[:, :, bodies],
                reference_motion.body_orientation_world[:, :, bodies],
            ).sum(axis=1),
            **deformation,
            "stability/joint_speed": self.joint_speed_weight
            * np.mean(executed_trajectory_joint_velocity**2, axis=-1).sum(axis=1),
            "terminal/joint_position": self.terminal_joint_position_weight
            * _joint_error(
                executed_trajectory_joint_position[:, -1], reference_motion.joint_position[:, -1]
            ),
            "terminal/body_position": self.terminal_body_position_weight
            * _body_position_error(
                executed_trajectory_body_position[:, -1, bodies],
                reference_motion.body_position_world[:, -1, bodies],
            ),
        }
        total = np.sum(np.stack(tuple(terms.values()), axis=0), axis=0)
        return CostResult(total=total, terms=terms)

    @staticmethod
    def _validate(
        measurements: ControlStepMeasurements,
        reference_motion: MotionCommand,
        candidate: MotionCommand,
    ) -> None:
        missing = MotionTrackingObjective.requires.difference(measurements.arrays)
        if missing:
            names = sorted(primitive.value for primitive in missing)
            raise ValueError(f"measurements are missing required primitives: {names}")
        for name, motion_command in (("reference_motion", reference_motion), ("candidate", candidate)):
            if (
                motion_command.joint_names != measurements.joint_names
                or motion_command.body_names != measurements.body_names
            ):
                raise ValueError(
                    f"{name} joint/body names must match the measurements"
                )
            if motion_command.query_offsets != tuple(range(measurements.num_steps)):
                raise ValueError(
                    f"{name} query_offsets must be dense "
                    f"(0..{measurements.num_steps - 1}), got "
                    f"{motion_command.query_offsets}"
                )
            for field in _COMMAND_ARRAYS:
                if not np.isfinite(getattr(motion_command, field)).all():
                    raise ValueError(f"{name} {field} must be finite")
        if reference_motion.joint_position.shape[0] != 1:
            raise ValueError(
                f"reference_motion batch must be 1, got {reference_motion.joint_position.shape[0]}"
            )
        if candidate.joint_position.shape[0] != measurements.batch:
            raise ValueError(
                f"candidate batch must be {measurements.batch}, "
                f"got {candidate.joint_position.shape[0]}"
            )


def _joint_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(
        (np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64))
        ** 2,
        axis=-1,
    )


def _deformation_terms(
    objective: MotionTrackingObjective,
    reference_motion: MotionCommand,
    candidate: MotionCommand,
) -> dict[str, np.ndarray]:
    """Return the executed_trajectory-state-independent objective contributions."""
    bodies = objective.body_indices(reference_motion.body_names)
    return {
        "deformation/joint_position": objective.deformation_joint_position_weight
        * _joint_error(candidate.joint_position, reference_motion.joint_position).sum(axis=1),
        "deformation/body_position": objective.deformation_body_position_weight
        * _body_position_error(
            candidate.body_position_world[:, :, bodies],
            reference_motion.body_position_world[:, :, bodies],
        ).sum(axis=1),
        "deformation/body_orientation": objective.deformation_body_orientation_weight
        * _body_orientation_error(
            candidate.body_orientation_world[:, :, bodies],
            reference_motion.body_orientation_world[:, :, bodies],
        ).sum(axis=1),
    }


def _body_position_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = (
        np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    return np.mean(np.sum(error**2, axis=-1), axis=-1)


def _body_orientation_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    angle = _quaternion_geodesic(actual, target)
    return np.mean(angle**2, axis=-1)
