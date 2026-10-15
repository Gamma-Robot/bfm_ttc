"""Synchronous simulation-in-the-loop receding-horizon execution."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sampling_optimizers import ConvergenceCriteria, IterationLimit, SamplingOptimizer
from bfm_adapters import (
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
)
from command_optimization.motion_tracking.command import (
    motion_command_window,
    reference_right_context,
    take_motion_command,
)
from command_optimization.motion_tracking.metrics import _quaternion_geodesic
from command_optimization.motion_tracking.search_space import QposKnotsSpace
from command_optimization.planning.evaluation import EvaluationLedger, RolloutEvaluator
from command_optimization.planning.update import PlanningUpdateResult, run_planning_update
from command_optimization.records import ExecutionTrace, PlanningDiagnostics
from command_optimization.simulation.command_execution import CommandExecutor, ExecutionResult
from command_optimization.simulation.measurements import MeasurementPrimitive, ControlStepMeasurements
from command_optimization.simulation.execution_state import ExecutionState


SearchSpaceFactory = Callable[[MotionCommand], QposKnotsSpace]
EvaluatorFactory = Callable[
    [QposKnotsSpace, MotionCommand, MotionCommand | None, ExecutionState],
    RolloutEvaluator,
]


@dataclass(frozen=True)
class SilLoopResult:
    """Minimal optimized-run payload consumed by ``ExecutionRunArtifact``."""

    trace: ExecutionTrace
    planning: tuple[PlanningDiagnostics, ...]


def run_command_optimization(
    *,
    reference_future: MotionCommand,
    initial_execution_state: ExecutionState,
    optimizer: SamplingOptimizer,
    search_space_factory: SearchSpaceFactory,
    evaluator_factory: EvaluatorFactory,
    executor: CommandExecutor,
    horizon_steps: int,
    replanning_interval_steps: int,
    control_period: float,
    seed: int,
    iterations: IterationLimit,
    root_body_name: str,
    body_names: tuple[str, ...],
    execution_steps: int | None = None,
    future_context_steps: int = 0,
    evaluate_proposal: bool = True,
) -> SilLoopResult:
    """Plan against the supplied reference_motion future and execute its declared prefix.

    ``reference_future`` is batch-one motion_command aligned with executed_trajectory post-step
    samples. For a target containing the reset pose at frame zero, callers pass
    ``target[1:]`` reindexed to a dense zero-based query grid. Every Planning
    Update receives a fixed-size edge-padded window, fresh search-space and
    evaluator instances, and a fresh optimizer initialization. The first update
    uses the optimizer's zero default; later updates warm-start from the previous
    selected latent shifted by the number of executed chunk steps.

    ``execution_steps`` limits executed/evaluated Control Steps while later frames in
    ``reference_future`` remain immutable planning and bfm context. When it
    is omitted, the complete supplied future is executed. ``future_context_steps``
    appends immutable reference_motion frames only at the bfm observation boundary.
    Candidate rollout and objective horizons remain exactly
    ``horizon_steps`` Control Steps.

    ``body_names`` selects task bodies for recorded deformation metrics.

    The returned candidate is the best evaluated choice before the configured
    iteration limit stops; this function makes no global-optimality claim.
    """
    total_steps = _validate_inputs(
        reference_future=reference_future,
        initial_execution_state=initial_execution_state,
        horizon_steps=horizon_steps,
        replanning_interval_steps=replanning_interval_steps,
        control_period=control_period,
        seed=seed,
        iterations=iterations,
        evaluate_proposal=evaluate_proposal,
        root_body_name=root_body_name,
        execution_steps=execution_steps,
        future_context_steps=future_context_steps,
    )
    if (
        not body_names
        or len(set(body_names)) != len(body_names)
        or any(name not in reference_future.body_names for name in body_names)
    ):
        raise ValueError("body_names must be a non-empty unique subset of reference_motion bodies")
    body_indices = [reference_future.body_names.index(name) for name in body_names]
    period = float(control_period)
    horizon = int(horizon_steps)
    chunk_step_limit = int(replanning_interval_steps)
    context_steps = int(future_context_steps)

    execution_state = initial_execution_state
    cursor = 0
    update_index = 0
    previous_latent: np.ndarray | None = None
    previous_chunk_steps: int | None = None
    applied_command_segments: list[MotionCommand] = []
    measurement_segments: list[ControlStepMeasurements] = []
    qpos_segments: list[np.ndarray] = []
    qvel_segments: list[np.ndarray] = []
    persisted: list[PlanningDiagnostics] = []

    while cursor < total_steps:
        reference_window = motion_command_window(reference_future, cursor, horizon)
        right_context = reference_right_context(
            reference_future, cursor + horizon, context_steps
        )
        search_space = search_space_factory(reference_window)
        expected_shape = (optimizer.num_knots, optimizer.dim)
        if tuple(search_space.latent_shape) != expected_shape:
            raise ValueError(
                "search-space latent shape must match optimizer shape "
                f"{expected_shape}, got {search_space.latent_shape}"
            )
        evaluator = evaluator_factory(
            search_space, reference_window, right_context, execution_state
        )
        if previous_latent is None:
            initial_mean = None
        else:
            assert previous_chunk_steps is not None
            initial_mean = search_space.shift(
                previous_latent, previous_chunk_steps
            )
        # run_planning_update returns host-resident costs, so all work behind
        # them has synchronized before the clock stops.
        update_started = time.perf_counter()
        update = run_planning_update(
            optimizer,
            evaluator,
            seed=int(seed) + update_index,
            iterations=iterations,
            initial_mean=initial_mean,
            evaluate_proposal=evaluate_proposal,
        )
        update_wall_seconds = time.perf_counter() - update_started
        if _all_candidate_evaluations_penalized(update.diagnostics.evaluations):
            raise RuntimeError(
                f"planning update at step {cursor} penalized every candidate evaluation"
            )

        executed_chunk_steps = min(chunk_step_limit, total_steps - cursor)
        execution = executor.execute(
            execution_state,
            update.selected_command,
            num_steps=executed_chunk_steps,
            right_context=right_context,
            requires=frozenset(MeasurementPrimitive),
        )
        _validate_execution(
            execution, execution_state, executed_chunk_steps, period,
            physics_substeps=executor.physics_substeps,
        )

        applied_command_segments.append(
            _motion_command_prefix(update.selected_command, executed_chunk_steps)
        )
        measurement_segments.append(execution.measurements)
        qpos_segments.append(np.asarray(execution.qpos)[0])
        qvel_segments.append(np.asarray(execution.qvel)[0])
        persisted.append(
            _planning_diagnostics(
                cursor, update, reference_window, body_indices, update_wall_seconds
            )
        )

        execution_state = execution.final_execution_state
        previous_latent = update.selected_latent
        previous_chunk_steps = executed_chunk_steps
        cursor += executed_chunk_steps
        update_index += 1

    applied_command = _concatenate_motion_command(applied_command_segments)
    executed_trajectory = _concatenate_measurements(measurement_segments)
    timeline = initial_execution_state.time + period * np.arange(
        1, total_steps + 1, dtype=np.float64
    )
    trace = ExecutionTrace(
        timeline=timeline,
        reference_motion=_motion_command_prefix(reference_future, total_steps),
        applied_command=applied_command,
        executed_trajectory=executed_trajectory,
        qpos=np.concatenate(qpos_segments, axis=0),
        qvel=np.concatenate(qvel_segments, axis=0),
        valid=np.ones(total_steps, dtype=np.bool_),
        root_body_name=root_body_name,
    )
    return SilLoopResult(trace=trace, planning=tuple(persisted))


def _validate_inputs(
    *,
    reference_future: MotionCommand,
    initial_execution_state: ExecutionState,
    horizon_steps: int,
    replanning_interval_steps: int,
    control_period: float,
    seed: int,
    iterations: IterationLimit,
    evaluate_proposal: bool,
    root_body_name: str,
    execution_steps: int | None,
    future_context_steps: int,
) -> int:
    if not isinstance(reference_future, MotionCommand):
        raise TypeError("reference_future must be MotionCommand")
    available_steps = len(reference_future.query_offsets)
    if available_steps < 1:
        raise ValueError("reference_future must contain at least one future frame")
    if reference_future.query_offsets != tuple(range(available_steps)):
        raise ValueError("reference_future must use a dense zero-based query grid")
    if reference_future.joint_position.shape[0] != 1:
        raise ValueError("reference_future must contain exactly one execution lane")
    for field in _COMMAND_ARRAYS:
        if not np.isfinite(getattr(reference_future, field)).all():
            raise ValueError(f"reference_future {field} must be finite")
    if initial_execution_state.qpos.shape[0] != 1:
        raise ValueError("initial_execution_state must contain exactly one execution lane")
    if (
        not isinstance(root_body_name, str)
        or root_body_name not in reference_future.body_names
    ):
        raise ValueError("root_body_name must name one reference_motion body")
    _positive_integer(horizon_steps, "horizon_steps", minimum=2)
    _positive_integer(replanning_interval_steps, "replanning_interval_steps")
    if replanning_interval_steps > horizon_steps:
        raise ValueError("replanning_interval_steps cannot exceed horizon_steps")
    period = float(control_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("control_period must be finite and positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(iterations, ConvergenceCriteria):
        _positive_integer(iterations, "iterations")
    _positive_integer(future_context_steps, "future_context_steps", minimum=0)
    if execution_steps is None:
        total_steps = available_steps
    else:
        _positive_integer(execution_steps, "execution_steps")
        total_steps = int(execution_steps)
        if total_steps > available_steps:
            raise ValueError("execution_steps cannot exceed the supplied reference_motion future")
    if not isinstance(evaluate_proposal, bool):
        raise TypeError("evaluate_proposal must be a bool")
    return total_steps


def _positive_integer(value: int, field: str, *, minimum: int = 1) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _motion_command_prefix(motion_command: MotionCommand, num_steps: int) -> MotionCommand:
    return take_motion_command(motion_command, np.arange(num_steps))


def _all_candidate_evaluations_penalized(
    evaluations: tuple[EvaluationLedger, ...],
) -> bool:
    candidates = sum(record.candidate_evaluations for record in evaluations)
    penalized = sum(record.penalized_candidate_evaluations for record in evaluations)
    return candidates > 0 and penalized == candidates


def _validate_execution(
    execution: ExecutionResult,
    before: ExecutionState,
    num_steps: int,
    control_period: float,
    *,
    physics_substeps: int,
) -> None:
    if (
        execution.measurements.batch != 1
        or execution.measurements.num_steps != num_steps
    ):
        raise RuntimeError("execution returned an incompatible measurement trace")
    if execution.qpos.ndim != 3 or execution.qpos.shape[:2] != (1, num_steps):
        raise RuntimeError("execution returned an incompatible qpos trace")
    if execution.qvel.ndim != 3 or execution.qvel.shape[:2] != (1, num_steps):
        raise RuntimeError("execution returned an incompatible qvel trace")
    after = execution.final_execution_state
    expected_time = before.time + num_steps * control_period
    clock_magnitude = np.float32(max(1.0, abs(after.time), abs(expected_time)))
    clock_tolerance = max(
        1e-4,
        0.5 * (num_steps * physics_substeps + 1) * float(np.spacing(clock_magnitude)),
    )
    time_matches = np.isclose(
        after.time,
        expected_time,
        rtol=0.0,
        # Each float32 physics increment and the initial clock conversion can
        # round by half an ULP; the integer step and motion indices stay exact.
        atol=clock_tolerance,
    )
    if (
        after.qpos.shape[0] != 1
        or after.control_step != before.control_step + num_steps
        or after.motion_command_index != before.motion_command_index + num_steps
        or not time_matches
    ):
        raise RuntimeError("execution did not advance the complete execution state")


def _planning_diagnostics(
    update_step: int,
    update: PlanningUpdateResult,
    reference_motion: MotionCommand,
    body_indices: list[int],
    update_wall_seconds: float,
) -> PlanningDiagnostics:
    diagnostics = update.diagnostics
    quantiles = np.asarray(
        [record.cost_quantiles for record in diagnostics.evaluations],
        dtype=np.float64,
    )
    return PlanningDiagnostics(
        update_step=update_step,
        selected_source=diagnostics.selected_source,
        selected_candidate_id=_candidate_id(update.selected_latent),
        selected_cost=update.selected_cost,
        best_observed_candidate_id=_candidate_id(
            diagnostics.best_observed_candidate
        ),
        best_observed_cost=diagnostics.best_observed_cost,
        proposal_candidate_id=_candidate_id(diagnostics.proposal_candidate),
        proposal_cost=diagnostics.proposal_cost,
        cost_quantiles=quantiles,
        candidate_deformation=_candidate_deformation(
            update.selected_command, reference_motion, body_indices
        ),
        candidate_evaluations=diagnostics.candidate_evaluations,
        scenario_count=diagnostics.scenario_count,
        scenario_digest=diagnostics.scenario_digest,
        logical_rollouts=diagnostics.logical_rollouts,
        physical_rollouts=diagnostics.physical_rollouts,
        penalized_candidate_evaluations=(
            diagnostics.penalized_candidate_evaluations
        ),
        invalid_scenario_rollouts=diagnostics.invalid_scenario_rollouts,
        update_wall_seconds=update_wall_seconds,
    )


def _candidate_id(candidate: np.ndarray) -> str:
    value = np.ascontiguousarray(candidate, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _candidate_deformation(
    candidate: MotionCommand, reference_motion: MotionCommand, body_indices: list[int]
) -> dict[str, float]:
    joint_delta = candidate.joint_position - reference_motion.joint_position
    body_delta = (
        candidate.body_position_world[:, :, body_indices]
        - reference_motion.body_position_world[:, :, body_indices]
    )
    angle = _quaternion_geodesic(
        candidate.body_orientation_world[:, :, body_indices],
        reference_motion.body_orientation_world[:, :, body_indices],
    )
    return {
        "joint_position_rmse_rad": float(np.sqrt(np.mean(joint_delta**2))),
        "body_position_rmse_m": float(
            np.sqrt(np.mean(np.sum(body_delta**2, axis=-1)))
        ),
        "body_orientation_rmse_rad": float(np.sqrt(np.mean(angle**2))),
    }


def _concatenate_motion_command(segments: list[MotionCommand]) -> MotionCommand:
    first = segments[0]
    return MotionCommand(
        query_offsets=tuple(range(sum(len(segment.query_offsets) for segment in segments))),
        joint_names=first.joint_names,
        body_names=first.body_names,
        **{
            name: np.concatenate([getattr(segment, name) for segment in segments], axis=1)
            for name in _COMMAND_ARRAYS
        },
    )


def _concatenate_measurements(
    segments: list[ControlStepMeasurements],
) -> ControlStepMeasurements:
    first = segments[0]
    return ControlStepMeasurements(
        joint_names=first.joint_names,
        body_names=first.body_names,
        arrays={
            primitive: np.concatenate(
                [segment.arrays[primitive] for segment in segments], axis=1
            )
            for primitive in MeasurementPrimitive
        },
    )
