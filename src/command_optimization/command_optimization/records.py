"""Immutable currencies for one baseline/optimized command-optimization experiment pair."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from bfm_adapters import (
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
)
from command_optimization.simulation.measurements import (
    MeasurementPrimitive,
    ControlStepMeasurements,
)


_CLAIM_PRIMITIVES = frozenset(MeasurementPrimitive)


def _owned_float(
    value: np.ndarray, field: str, *, ndim: int | None = None
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or not np.issubdtype(
        value.dtype, np.floating
    ):
        raise TypeError(f"{field} must be a floating-point numpy array")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{field} must have {ndim} dimensions, got {value.shape}")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite")
    result.flags.writeable = False
    return result


def _owned_motion_command(value: MotionCommand, field: str) -> MotionCommand:
    if not isinstance(value, MotionCommand):
        raise TypeError(f"{field} must be MotionCommand")
    arrays = {
        name: _owned_float(
            getattr(value, name),
            f"{field}.{name}",
        )
        for name in _COMMAND_ARRAYS
    }
    return MotionCommand(
        query_offsets=tuple(value.query_offsets),
        joint_names=tuple(value.joint_names),
        body_names=tuple(value.body_names),
        **arrays,
    )


def _owned_measurements(value: ControlStepMeasurements) -> ControlStepMeasurements:
    if not isinstance(value, ControlStepMeasurements):
        raise TypeError("executed_trajectory must be ControlStepMeasurements or None")
    arrays = {
        primitive: _owned_float(
            array,
            f"executed_trajectory.{primitive.value}",
        )
        for primitive, array in value.arrays.items()
    }
    return ControlStepMeasurements(
        joint_names=tuple(value.joint_names),
        body_names=tuple(value.body_names),
        arrays=arrays,
    )


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True, eq=False)
class ExecutionTrace:
    """One full reference_motion timeline and its physically executed_trajectory valid prefix.

    ``failure_step`` is the zero-based index of the first invalid Control Step.
    Executed measurements and physical states contain real samples only; a
    failed suffix is never padded with fabricated poses.
    """

    timeline: np.ndarray
    reference_motion: MotionCommand
    applied_command: MotionCommand
    executed_trajectory: ControlStepMeasurements | None
    qpos: np.ndarray
    qvel: np.ndarray
    valid: np.ndarray
    root_body_name: str
    failure_step: int | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        timeline = _owned_float(self.timeline, "timeline", ndim=1)
        if timeline.size < 1:
            raise ValueError("timeline must contain at least one Control Step")
        if timeline.size > 1 and not np.all(np.diff(timeline) > 0.0):
            raise ValueError("timeline must be strictly increasing")

        reference_motion = _owned_motion_command(self.reference_motion, "reference_motion")
        applied_command = _owned_motion_command(
            self.applied_command, "applied_command"
        )
        num_steps = timeline.size
        for name, motion_command in (
            ("reference_motion", reference_motion),
            ("applied_command", applied_command),
        ):
            if motion_command.joint_position.shape[0] != 1:
                raise ValueError(f"{name} must contain exactly one execution lane")
            if motion_command.query_offsets != tuple(range(num_steps)):
                raise ValueError(
                    f"{name} must use the trace's dense zero-based timeline"
                )
        if (
            applied_command.joint_names != reference_motion.joint_names
            or applied_command.body_names != reference_motion.body_names
        ):
            raise ValueError("applied_command metadata must match reference_motion")

        if not isinstance(self.valid, np.ndarray) or self.valid.dtype != np.bool_:
            raise TypeError("valid must be a boolean numpy array")
        if self.valid.shape != (num_steps,):
            raise ValueError(
                f"valid must have shape {(num_steps,)}, got {self.valid.shape}"
            )
        valid = np.array(self.valid, dtype=np.bool_, order="C", copy=True)
        valid_steps = int(np.count_nonzero(valid))
        expected_valid = np.arange(num_steps) < valid_steps
        if not np.array_equal(valid, expected_valid):
            raise ValueError("valid Control Steps must form one contiguous prefix")
        valid.flags.writeable = False

        executed_trajectory = None if self.executed_trajectory is None else _owned_measurements(self.executed_trajectory)
        if valid_steps == 0:
            if executed_trajectory is not None:
                raise ValueError("executed_trajectory must be None when no Control Step is valid")
        else:
            if executed_trajectory is None:
                raise ValueError(
                    "executed_trajectory measurements are required for valid Control Steps"
                )
            if executed_trajectory.batch != 1 or executed_trajectory.num_steps != valid_steps:
                raise ValueError(
                    "executed_trajectory must contain one lane and exactly the valid prefix"
                )
            if (
                executed_trajectory.joint_names != reference_motion.joint_names
                or executed_trajectory.body_names != reference_motion.body_names
            ):
                raise ValueError("executed_trajectory metadata must match reference_motion")
            if set(executed_trajectory.arrays) != _CLAIM_PRIMITIVES:
                raise ValueError(
                    "executed_trajectory must contain all motion-tracking claim primitives"
                )

        qpos = _owned_float(self.qpos, "qpos", ndim=2)
        qvel = _owned_float(self.qvel, "qvel", ndim=2)
        if qpos.shape[0] != valid_steps or qvel.shape[0] != valid_steps:
            raise ValueError("qpos and qvel must contain exactly the valid prefix")
        if qpos.shape[1] < 1 or qvel.shape[1] < 1:
            raise ValueError("qpos and qvel state dimensions must be non-empty")

        _nonempty(self.root_body_name, "root_body_name")
        if self.root_body_name not in reference_motion.body_names:
            raise ValueError("root_body_name must name one reference_motion body")

        if valid_steps == num_steps:
            if self.failure_step is not None or self.failure_reason is not None:
                raise ValueError("a complete trace cannot carry failure metadata")
        else:
            if not isinstance(self.failure_step, (int, np.integer)):
                raise ValueError("an incomplete trace requires failure_step")
            if int(self.failure_step) != valid_steps:
                raise ValueError("failure_step must be the first invalid Control Step")
            if self.failure_reason is None:
                raise ValueError("an incomplete trace requires failure_reason")
            _nonempty(self.failure_reason, "failure_reason")
            object.__setattr__(self, "failure_step", int(self.failure_step))

        object.__setattr__(self, "timeline", timeline)
        object.__setattr__(self, "reference_motion", reference_motion)
        object.__setattr__(self, "applied_command", applied_command)
        object.__setattr__(self, "executed_trajectory", executed_trajectory)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "qvel", qvel)
        object.__setattr__(self, "valid", valid)

    @property
    def num_steps(self) -> int:
        return int(self.timeline.size)

    @property
    def valid_steps(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def success(self) -> bool:
        return self.failure_step is None


@dataclass(frozen=True, eq=False)
class PlanningDiagnostics:
    """Persisted evidence for one configured Planning Update.

    ``update_wall_seconds`` is the complete wall time of the Planning Update
    measured with a monotonic clock; it is feasibility evidence, not a
    real-time claim.
    """

    update_step: int
    selected_source: Literal["best_observed", "proposal"]
    selected_candidate_id: str
    selected_cost: float
    best_observed_candidate_id: str
    best_observed_cost: float
    proposal_candidate_id: str
    proposal_cost: float | None
    cost_quantiles: np.ndarray
    candidate_deformation: Mapping[str, float]
    candidate_evaluations: int
    scenario_count: int
    scenario_digest: str | None
    logical_rollouts: int
    physical_rollouts: int
    penalized_candidate_evaluations: int
    invalid_scenario_rollouts: int
    update_wall_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_step, (bool, np.bool_))
            or not isinstance(self.update_step, (int, np.integer))
            or self.update_step < 0
        ):
            raise ValueError("update_step must be a non-negative integer")
        object.__setattr__(self, "update_step", int(self.update_step))
        if self.selected_source not in ("best_observed", "proposal"):
            raise ValueError("selected_source must be best_observed or proposal")
        for field in (
            "selected_candidate_id",
            "best_observed_candidate_id",
            "proposal_candidate_id",
        ):
            _nonempty(getattr(self, field), field)
        for field in ("selected_cost", "best_observed_cost"):
            value = float(getattr(self, field))
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, value)
        if self.proposal_cost is not None:
            proposal_cost = float(self.proposal_cost)
            if not np.isfinite(proposal_cost):
                raise ValueError("proposal_cost must be finite when present")
            object.__setattr__(self, "proposal_cost", proposal_cost)

        if self.selected_source == "best_observed":
            expected_id = self.best_observed_candidate_id
            expected_cost = self.best_observed_cost
        else:
            if self.proposal_cost is None:
                raise ValueError("a selected proposal must have an evaluated cost")
            expected_id = self.proposal_candidate_id
            expected_cost = self.proposal_cost
        if (
            self.selected_candidate_id != expected_id
            or self.selected_cost != expected_cost
        ):
            raise ValueError("selected identity/cost must match selected_source")

        quantiles = _owned_float(self.cost_quantiles, "cost_quantiles", ndim=2)
        if quantiles.shape[0] < 1 or quantiles.shape[1] != 5:
            raise ValueError("cost_quantiles must have shape (evaluations, 5)")
        if not np.all(np.diff(quantiles, axis=1) >= 0.0):
            raise ValueError("each cost quantile row must be nondecreasing")

        if not isinstance(self.candidate_deformation, Mapping):
            raise TypeError("candidate_deformation must be a mapping")
        deformation: dict[str, float] = {}
        for name, raw_value in self.candidate_deformation.items():
            _nonempty(name, "candidate_deformation key")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("candidate deformation values must be finite and >= 0")
            deformation[name] = value
        if not deformation:
            raise ValueError("candidate_deformation must contain at least one measure")

        for field in ("candidate_evaluations", "scenario_count", "logical_rollouts"):
            value = getattr(self, field)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{field} must be a positive integer")
        if self.logical_rollouts != self.candidate_evaluations * self.scenario_count:
            raise ValueError(
                "logical_rollouts must equal candidate_evaluations * scenario_count"
            )
        if self.scenario_digest is not None:
            _nonempty(self.scenario_digest, "scenario_digest")
            if not self.scenario_digest.startswith("sha256:"):
                raise ValueError("scenario_digest must be a sha256 identity")
        if (
            isinstance(self.physical_rollouts, (bool, np.bool_))
            or not isinstance(self.physical_rollouts, (int, np.integer))
            or self.physical_rollouts < self.logical_rollouts
        ):
            raise ValueError("physical_rollouts must be an integer >= logical_rollouts")
        for field, upper in (
            ("penalized_candidate_evaluations", self.candidate_evaluations),
            ("invalid_scenario_rollouts", self.logical_rollouts),
        ):
            value = getattr(self, field)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value <= upper
            ):
                raise ValueError(f"{field} must be an integer in [0, {upper}]")
        wall_seconds = float(self.update_wall_seconds)
        if not np.isfinite(wall_seconds) or wall_seconds <= 0.0:
            raise ValueError("update_wall_seconds must be finite and positive")
        object.__setattr__(self, "update_wall_seconds", wall_seconds)
        object.__setattr__(
            self, "candidate_evaluations", int(self.candidate_evaluations)
        )
        object.__setattr__(self, "scenario_count", int(self.scenario_count))
        object.__setattr__(self, "logical_rollouts", int(self.logical_rollouts))
        object.__setattr__(self, "physical_rollouts", int(self.physical_rollouts))
        object.__setattr__(
            self,
            "penalized_candidate_evaluations",
            int(self.penalized_candidate_evaluations),
        )
        object.__setattr__(
            self, "invalid_scenario_rollouts", int(self.invalid_scenario_rollouts)
        )
        object.__setattr__(self, "cost_quantiles", quantiles)
        object.__setattr__(
            self, "candidate_deformation", MappingProxyType(deformation)
        )


@dataclass(frozen=True, eq=False)
class ExecutionRunArtifact:
    """One execution plus immutable identities needed for a paired claim.

    ``initial_physical_state_digest`` identifies only the reset qpos/qvel
    content and is reused for both resets. Opaque bfm state remains outside
    this identity and is never serialized by the application.
    """

    kind: Literal["baseline", "optimized"]
    reference_id: str
    reference_digest: str
    bfm_id: str
    checkpoint_digest: str
    robot_model_digest: str
    initial_physical_state_digest: str
    control_period: float
    trace: ExecutionTrace
    planning: tuple[PlanningDiagnostics, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("baseline", "optimized"):
            raise ValueError("kind must be baseline or optimized")
        for field in (
            "reference_id",
            "reference_digest",
            "bfm_id",
            "checkpoint_digest",
            "robot_model_digest",
            "initial_physical_state_digest",
        ):
            _nonempty(getattr(self, field), field)
        period = float(self.control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be finite and positive")
        if self.trace.num_steps > 1 and not np.allclose(
            np.diff(self.trace.timeline),
            period,
            rtol=0.0,
            atol=max(1e-12, period * 1e-9),
        ):
            raise ValueError("trace timeline spacing must equal control_period")
        planning = tuple(self.planning)
        if self.kind == "baseline" and planning:
            raise ValueError("baseline runs cannot contain planning diagnostics")
        update_steps = tuple(record.update_step for record in planning)
        if any(step >= self.trace.num_steps for step in update_steps):
            raise ValueError("Planning Update steps must lie on the run timeline")
        if any(right <= left for left, right in zip(update_steps, update_steps[1:])):
            raise ValueError("Planning Update steps must be strictly increasing")
        object.__setattr__(self, "control_period", period)
        object.__setattr__(self, "planning", planning)


@dataclass(frozen=True)
class PairedRunArtifact:
    """Raw baseline and optimized records; claim metrics are derived from this."""

    baseline: ExecutionRunArtifact
    optimized: ExecutionRunArtifact

    def __post_init__(self) -> None:
        if self.baseline.kind != "baseline":
            raise ValueError("baseline must have kind='baseline'")
        if self.optimized.kind != "optimized":
            raise ValueError("optimized must have kind='optimized'")
