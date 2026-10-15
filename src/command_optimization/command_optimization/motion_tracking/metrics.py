"""Execution-clip tracking metrics derived from paired execution records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bfm_adapters import MotionCommand
from command_optimization.records import ExecutionRunArtifact, ExecutionTrace, PairedRunArtifact
from command_optimization.simulation.measurements import MeasurementPrimitive

if TYPE_CHECKING:
    from command_optimization.motion_tracking.objective import MotionTrackingObjective


@dataclass(frozen=True)
class ExecutionClipTrackingMetrics:
    """Tracking errors and weighted running cost for one complete Execution Clip.

    Joint position is scalar RMSE over Control Steps and joints (radians). Body/root
    position is mean Euclidean distance (metres), and orientation is mean
    sign-invariant quaternion geodesic angle (radians). Body metrics include
    the task's declared bodies; root metrics are also reported separately.

    ``running_cost_sum`` sums the weighted running objective over executed
    Control Steps, excluding local terminal costs. ``running_cost_mean`` divides
    that sum by ``total_steps`` (cost per Control Step, not per second).

    Error and cost values are ``None`` after an early failure: valid-prefix
    measurements are not mislabeled as a complete Execution Clip result.
    """

    success: bool
    valid_steps: int
    total_steps: int
    failure_step: int | None
    failure_reason: str | None
    running_cost_sum: float | None
    running_cost_mean: float | None
    joint_position_rmse_rad: float | None
    body_position_mean_error_m: float | None
    body_orientation_mean_error_rad: float | None
    root_position_mean_error_m: float | None
    root_orientation_mean_error_rad: float | None


@dataclass(frozen=True)
class PairedTrackingMetrics:
    """Baseline and optimized metrics computed under validated shared identity."""

    baseline: ExecutionClipTrackingMetrics
    optimized: ExecutionClipTrackingMetrics


def compute_paired_metrics(
    pair: PairedRunArtifact, *, objective: MotionTrackingObjective
) -> PairedTrackingMetrics:
    """Validate paired-run parity, then compute execution errors and running cost."""
    # The objective reuses the quaternion helper below; import after initialization.
    from command_optimization.motion_tracking.objective import MotionTrackingObjective

    if not isinstance(pair, PairedRunArtifact):
        raise TypeError("pair must be PairedRunArtifact")
    if not isinstance(objective, MotionTrackingObjective):
        raise TypeError("objective must be MotionTrackingObjective")
    body_names = objective.body_names
    _validate_pair(pair.baseline, pair.optimized)
    bodies = [pair.baseline.trace.reference_motion.body_names.index(name) for name in body_names]
    return PairedTrackingMetrics(
        baseline=_compute_run_metrics(pair.baseline.trace, bodies, objective),
        optimized=_compute_run_metrics(pair.optimized.trace, bodies, objective),
    )


def _compute_run_metrics(
    trace: ExecutionTrace, bodies: list[int], objective: MotionTrackingObjective
) -> ExecutionClipTrackingMetrics:
    common = dict(
        success=trace.success,
        valid_steps=trace.valid_steps,
        total_steps=trace.num_steps,
        failure_step=trace.failure_step,
        failure_reason=trace.failure_reason,
    )
    if not trace.success:
        return ExecutionClipTrackingMetrics(
            **common,
            running_cost_sum=None,
            running_cost_mean=None,
            joint_position_rmse_rad=None,
            body_position_mean_error_m=None,
            body_orientation_mean_error_rad=None,
            root_position_mean_error_m=None,
            root_orientation_mean_error_rad=None,
        )

    executed_trajectory = trace.executed_trajectory
    if executed_trajectory is None:  # Protected by ExecutionTrace; keeps this seam explicit.
        raise ValueError("a successful trace must contain executed_trajectory measurements")
    reference_motion = trace.reference_motion
    actual_joint = executed_trajectory.arrays[MeasurementPrimitive.JOINT_POSITION][0]
    actual_body_position = executed_trajectory.arrays[
        MeasurementPrimitive.BODY_POSITION_WORLD
    ][0]
    actual_body_orientation = executed_trajectory.arrays[
        MeasurementPrimitive.BODY_ORIENTATION_WORLD
    ][0]
    target_joint = reference_motion.joint_position[0]
    target_body_position = reference_motion.body_position_world[0]
    target_body_orientation = reference_motion.body_orientation_world[0]
    root = reference_motion.body_names.index(trace.root_body_name)

    # Re-evaluate actual post-step execution and the command that was applied.
    # Selected planning costs include overlapping, partly unexecuted horizons
    # and local terminal penalties, so they cannot supply this metric.
    cost = objective.evaluate(executed_trajectory, reference_motion, trace.applied_command)
    running_cost_sum = float(np.sum(
        [value[0] for name, value in cost.terms.items() if not name.startswith("terminal/")],
        dtype=np.float64,
    ))
    if not np.isfinite(running_cost_sum) or running_cost_sum < 0.0:
        raise ValueError("execution running cost must be finite and non-negative")

    return ExecutionClipTrackingMetrics(
        **common,
        running_cost_sum=running_cost_sum,
        running_cost_mean=running_cost_sum / trace.num_steps,
        joint_position_rmse_rad=float(
            np.sqrt(np.mean((actual_joint - target_joint) ** 2))
        ),
        body_position_mean_error_m=float(
            np.mean(
                np.linalg.norm(
                    actual_body_position[:, bodies] - target_body_position[:, bodies],
                    axis=-1,
                )
            )
        ),
        body_orientation_mean_error_rad=float(
            np.mean(
                _quaternion_geodesic(
                    actual_body_orientation[:, bodies], target_body_orientation[:, bodies]
                )
            )
        ),
        root_position_mean_error_m=float(
            np.mean(
                np.linalg.norm(
                    actual_body_position[:, root] - target_body_position[:, root],
                    axis=-1,
                )
            )
        ),
        root_orientation_mean_error_rad=float(
            np.mean(
                _quaternion_geodesic(
                    actual_body_orientation[:, root],
                    target_body_orientation[:, root],
                )
            )
        ),
    )


def _quaternion_geodesic(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    actual_norm = np.linalg.norm(actual, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    if (
        np.any(actual_norm <= 1e-12)
        or np.any(target_norm <= 1e-12)
        or not np.isfinite(actual_norm).all()
        or not np.isfinite(target_norm).all()
    ):
        raise ValueError("body orientations must contain finite, non-zero quaternions")
    dots = np.sum((actual / actual_norm) * (target / target_norm), axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))


def _validate_pair(
    baseline: ExecutionRunArtifact, optimized: ExecutionRunArtifact
) -> None:
    for field in (
        "reference_id",
        "reference_digest",
        "bfm_id",
        "checkpoint_digest",
        "robot_model_digest",
        "initial_physical_state_digest",
        "control_period",
    ):
        if getattr(baseline, field) != getattr(optimized, field):
            raise ValueError(f"paired runs must share {field}")

    left = baseline.trace
    right = optimized.trace
    if not np.array_equal(left.timeline, right.timeline):
        raise ValueError("paired runs must share the reference_motion timeline")
    if left.root_body_name != right.root_body_name:
        raise ValueError("paired runs must share root_body_name")
    if not _same_motion_command(left.reference_motion, right.reference_motion):
        raise ValueError("paired runs must share the immutable reference_motion target")


def _same_motion_command(left: MotionCommand, right: MotionCommand) -> bool:
    if (
        left.query_offsets != right.query_offsets
        or left.joint_names != right.joint_names
        or left.body_names != right.body_names
    ):
        return False
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in (
            "joint_position",
            "joint_velocity",
            "body_position_world",
            "body_orientation_world",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
        )
    )
