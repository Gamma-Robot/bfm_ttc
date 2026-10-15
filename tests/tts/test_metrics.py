from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bfm_adapters import MotionCommand
from command_optimization.motion_tracking.metrics import compute_paired_metrics
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.records import ExecutionRunArtifact, ExecutionTrace, PairedRunArtifact
from command_optimization.simulation.measurements import (
    MeasurementPrimitive,
    ControlStepMeasurements,
)


JOINTS = ("hip", "knee")
BODIES = ("pelvis", "foot")
NUM_STEPS = 2


def _objective(body_names=BODIES, **overrides):
    values = dict(
        body_names=body_names,
        tracking_joint_position_weight=1.0,
        tracking_joint_velocity_weight=0.0,
        tracking_body_position_weight=0.0,
        tracking_body_orientation_weight=0.0,
        deformation_joint_position_weight=0.1,
        deformation_body_position_weight=0.0,
        deformation_body_orientation_weight=0.0,
        joint_speed_weight=0.0,
        terminal_joint_position_weight=1.0e12,
        terminal_body_position_weight=1.0e12,
    )
    values.update(overrides)
    return MotionTrackingObjective(**values)


def _motion_command(*, candidate_joint: float = 0.0, num_steps: int = NUM_STEPS) -> MotionCommand:
    orientation = np.zeros((1, num_steps, len(BODIES), 4))
    orientation[..., 0] = 1.0
    body_vectors = np.zeros((1, num_steps, len(BODIES), 3))
    joint = np.full((1, num_steps, len(JOINTS)), candidate_joint)
    return MotionCommand(
        query_offsets=tuple(range(num_steps)),
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body_vectors.copy(),
        body_orientation_world=orientation,
        body_linear_velocity_world=body_vectors.copy(),
        body_angular_velocity_world=body_vectors.copy(),
    )


def _measurements(
    num_steps: int = NUM_STEPS,
    *,
    joint: float = 0.0,
    body_x: float = 0.0,
    orientation_angle: float = 0.0,
) -> ControlStepMeasurements:
    orientation = np.zeros((1, num_steps, len(BODIES), 4))
    orientation[..., 0] = np.cos(orientation_angle / 2.0)
    orientation[..., 1] = np.sin(orientation_angle / 2.0)
    body_position = np.zeros((1, num_steps, len(BODIES), 3))
    body_position[..., 0] = body_x
    return ControlStepMeasurements(
        JOINTS,
        BODIES,
        {
            MeasurementPrimitive.JOINT_POSITION: np.full(
                (1, num_steps, len(JOINTS)), joint
            ),
            MeasurementPrimitive.JOINT_VELOCITY: np.zeros(
                (1, num_steps, len(JOINTS))
            ),
            MeasurementPrimitive.BODY_POSITION_WORLD: body_position,
            MeasurementPrimitive.BODY_ORIENTATION_WORLD: orientation,
        },
    )


def _trace(
    *,
    executed_trajectory: ControlStepMeasurements | None = None,
    candidate_joint: float = 9.0,
    reference_joint: float = 0.0,
    valid_steps: int = NUM_STEPS,
    timeline: np.ndarray | None = None,
    num_steps: int = NUM_STEPS,
) -> ExecutionTrace:
    failed = valid_steps < num_steps
    return ExecutionTrace(
        timeline=(
            np.arange(num_steps, dtype=float) * 0.1 if timeline is None else timeline
        ),
        reference_motion=_motion_command(candidate_joint=reference_joint, num_steps=num_steps),
        applied_command=_motion_command(candidate_joint=candidate_joint, num_steps=num_steps),
        executed_trajectory=(
            _measurements(valid_steps) if executed_trajectory is None and valid_steps else executed_trajectory
        ),
        qpos=np.zeros((valid_steps, 4)),
        qvel=np.zeros((valid_steps, 3)),
        valid=np.arange(num_steps) < valid_steps,
        root_body_name="pelvis",
        failure_step=valid_steps if failed else None,
        failure_reason="fell" if failed else None,
    )


def _run(kind: str, *, trace: ExecutionTrace | None = None) -> ExecutionRunArtifact:
    return ExecutionRunArtifact(
        kind=kind,  # type: ignore[arg-type]
        reference_id="walk",
        reference_digest="target-sha",
        bfm_id="toy_bfm",
        checkpoint_digest="checkpoint-sha",
        robot_model_digest="model-sha",
        initial_physical_state_digest="physical-sha",
        control_period=0.1,
        trace=_trace() if trace is None else trace,
    )


def _pair(
    *,
    baseline: ExecutionRunArtifact | None = None,
    optimized: ExecutionRunArtifact | None = None,
) -> PairedRunArtifact:
    return PairedRunArtifact(
        _run("baseline") if baseline is None else baseline,
        _run("optimized") if optimized is None else optimized,
    )


@pytest.mark.parametrize(
    ("body_names", "position_error", "orientation_error"),
    ((BODIES, 2.0, 2 * np.pi / 3), (("foot",), 1.0, np.pi / 3)),
)
def test_execution_clip_metrics_use_reference_not_candidate_and_define_units(
    body_names: tuple[str, ...], position_error: float, orientation_error: float
) -> None:
    executed_trajectory = _measurements(joint=1.0, body_x=3.0, orientation_angle=np.pi)
    executed_trajectory.arrays[MeasurementPrimitive.BODY_POSITION_WORLD][:, :, 1, 0] = 1.0
    executed_trajectory.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD][:, :, 1] = (
        np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0, 0.0
    )
    baseline = _run("baseline", trace=_trace(executed_trajectory=executed_trajectory, candidate_joint=100.0))
    result = compute_paired_metrics(_pair(baseline=baseline), objective=_objective(body_names)).baseline

    assert result.success
    assert result.valid_steps == result.total_steps == NUM_STEPS
    assert result.joint_position_rmse_rad == pytest.approx(1.0)
    assert result.body_position_mean_error_m == pytest.approx(position_error)
    assert result.body_orientation_mean_error_rad == pytest.approx(orientation_error)
    assert result.root_position_mean_error_m == pytest.approx(3.0)
    assert result.root_orientation_mean_error_rad == pytest.approx(np.pi)


def test_quaternion_metrics_are_scale_and_sign_invariant() -> None:
    measurements = _measurements(orientation_angle=np.pi / 3)
    orientation = measurements.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD]
    orientation *= -7.0
    baseline = _run("baseline", trace=_trace(executed_trajectory=measurements))

    result = compute_paired_metrics(_pair(baseline=baseline), objective=_objective()).baseline

    assert result.body_orientation_mean_error_rad == pytest.approx(np.pi / 3)
    assert result.root_orientation_mean_error_rad == pytest.approx(np.pi / 3)


def test_early_failure_has_no_mislabeled_execution_clip_error() -> None:
    optimized = _run("optimized", trace=_trace(valid_steps=1))
    result = compute_paired_metrics(_pair(optimized=optimized), objective=_objective()).optimized

    assert not result.success
    assert result.failure_step == 1
    assert result.failure_reason == "fell"
    assert result.valid_steps == 1
    assert result.total_steps == 2
    assert result.running_cost_sum is None
    assert result.running_cost_mean is None
    assert result.joint_position_rmse_rad is None
    assert result.body_position_mean_error_m is None
    assert result.body_orientation_mean_error_rad is None
    assert result.root_position_mean_error_m is None
    assert result.root_orientation_mean_error_rad is None


@pytest.mark.parametrize(
    "field",
    (
        "reference_id",
        "reference_digest",
        "bfm_id",
        "checkpoint_digest",
        "robot_model_digest",
        "initial_physical_state_digest",
    ),
)
def test_metrics_reject_identity_mismatch(field: str) -> None:
    optimized = replace(_run("optimized"), **{field: "different"})

    with pytest.raises(ValueError, match=field):
        compute_paired_metrics(_pair(optimized=optimized), objective=_objective())


def test_metrics_reject_timeline_and_reference_target_mismatch() -> None:
    shifted_timeline = _trace(timeline=np.array([1.0, 1.1]))
    with pytest.raises(ValueError, match="timeline"):
        compute_paired_metrics(
            _pair(optimized=_run("optimized", trace=shifted_timeline)), objective=_objective()
        )

    changed_reference_motion = _trace(reference_joint=0.1)
    with pytest.raises(ValueError, match="reference_motion target"):
        compute_paired_metrics(
            _pair(optimized=_run("optimized", trace=changed_reference_motion)), objective=_objective()
        )


@pytest.mark.parametrize("num_steps", [2, 5])
def test_running_cost_uses_execution_and_applied_command_without_terminal(num_steps):
    baseline = _run("baseline", trace=_trace(
        num_steps=num_steps, valid_steps=num_steps,
        executed_trajectory=_measurements(num_steps, joint=1.0, body_x=3.0),
        candidate_joint=0.0,
    ))
    optimized = _run("optimized", trace=_trace(
        num_steps=num_steps, valid_steps=num_steps,
        executed_trajectory=_measurements(num_steps, joint=1.0, body_x=3.0),
        candidate_joint=2.0,
    ))
    pair = _pair(baseline=baseline, optimized=optimized)
    metrics = compute_paired_metrics(pair, objective=_objective())
    assert metrics.baseline.running_cost_sum == pytest.approx(num_steps * 1.0)
    assert metrics.baseline.running_cost_mean == pytest.approx(1.0)
    assert metrics.optimized.running_cost_sum == pytest.approx(num_steps * 1.4)
    assert metrics.optimized.running_cost_mean == pytest.approx(1.4)
    without_terminal = compute_paired_metrics(pair, objective=_objective(
        terminal_joint_position_weight=0.0, terminal_body_position_weight=0.0,
    ))
    assert metrics.optimized.running_cost_sum == without_terminal.optimized.running_cost_sum
    # Local horizon cost includes the huge terminal penalty; execution cost does not.
    local = _objective().evaluate(
        optimized.trace.executed_trajectory,
        optimized.trace.reference_motion,
        optimized.trace.applied_command,
    )
    assert local.total[0] > 1.0e12


def test_running_cost_uses_task_bodies_and_joint_speed_regularization():
    measurements = _measurements(body_x=100.0)
    measurements.arrays[MeasurementPrimitive.BODY_POSITION_WORLD][:, :, 1, 0] = 2.0
    measurements.arrays[MeasurementPrimitive.JOINT_VELOCITY][:] = 3.0
    run = _run("baseline", trace=_trace(executed_trajectory=measurements, candidate_joint=0.0))
    result = compute_paired_metrics(_pair(baseline=run), objective=_objective(
        body_names=("foot",), tracking_joint_position_weight=0.0,
        tracking_body_position_weight=1.0, joint_speed_weight=0.01,
    )).baseline
    assert result.running_cost_mean == pytest.approx(4.09)
    assert result.running_cost_sum == pytest.approx(8.18)
