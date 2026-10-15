from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from bfm_adapters import MotionCommand
from command_optimization.records import (
    ExecutionRunArtifact,
    ExecutionTrace,
    PairedRunArtifact,
    PlanningDiagnostics,
)
from command_optimization.simulation.measurements import (
    MeasurementPrimitive,
    ControlStepMeasurements,
)


JOINTS = ("hip", "knee")
BODIES = ("pelvis", "foot")


def _motion_command(num_steps: int = 3, *, joint: float = 0.0) -> MotionCommand:
    orientation = np.zeros((1, num_steps, len(BODIES), 4), dtype=np.float32)
    orientation[..., 0] = 1.0
    body_vectors = np.zeros((1, num_steps, len(BODIES), 3), dtype=np.float32)
    joint_position = np.full((1, num_steps, len(JOINTS)), joint, dtype=np.float32)
    return MotionCommand(
        query_offsets=tuple(range(num_steps)),
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=joint_position,
        joint_velocity=np.zeros_like(joint_position),
        body_position_world=body_vectors.copy(),
        body_orientation_world=orientation,
        body_linear_velocity_world=body_vectors.copy(),
        body_angular_velocity_world=body_vectors.copy(),
    )


def _measurements(num_steps: int) -> ControlStepMeasurements:
    orientation = np.zeros((1, num_steps, len(BODIES), 4), dtype=np.float32)
    orientation[..., 0] = 1.0
    return ControlStepMeasurements(
        JOINTS,
        BODIES,
        {
            MeasurementPrimitive.JOINT_POSITION: np.zeros(
                (1, num_steps, len(JOINTS)), dtype=np.float32
            ),
            MeasurementPrimitive.JOINT_VELOCITY: np.zeros(
                (1, num_steps, len(JOINTS)), dtype=np.float32
            ),
            MeasurementPrimitive.BODY_POSITION_WORLD: np.zeros(
                (1, num_steps, len(BODIES), 3), dtype=np.float32
            ),
            MeasurementPrimitive.BODY_ORIENTATION_WORLD: orientation,
        },
    )


def _trace(*, valid_steps: int = 3) -> ExecutionTrace:
    num_steps = 3
    failed = valid_steps < num_steps
    return ExecutionTrace(
        timeline=np.arange(num_steps, dtype=np.float64) * 0.1,
        reference_motion=_motion_command(num_steps),
        applied_command=_motion_command(num_steps, joint=0.25),
        executed_trajectory=None if valid_steps == 0 else _measurements(valid_steps),
        qpos=np.zeros((valid_steps, 4), dtype=np.float32),
        qvel=np.zeros((valid_steps, 3), dtype=np.float32),
        valid=np.arange(num_steps) < valid_steps,
        root_body_name="pelvis",
        failure_step=valid_steps if failed else None,
        failure_reason="fell" if failed else None,
    )


def _diagnostics() -> PlanningDiagnostics:
    return PlanningDiagnostics(
        update_step=0,
        selected_source="best_observed",
        selected_candidate_id="best-sha",
        selected_cost=1.0,
        best_observed_candidate_id="best-sha",
        best_observed_cost=1.0,
        proposal_candidate_id="proposal-sha",
        proposal_cost=2.0,
        cost_quantiles=np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        candidate_deformation={"joint_rmse_rad": 0.2},
        candidate_evaluations=8,
        scenario_count=1,
        scenario_digest=None,
        logical_rollouts=8,
        physical_rollouts=16,
        penalized_candidate_evaluations=0,
        invalid_scenario_rollouts=0,
        update_wall_seconds=0.125,
    )


def _run(kind: str) -> ExecutionRunArtifact:
    return ExecutionRunArtifact(
        kind=kind,  # type: ignore[arg-type]
        reference_id="walk",
        reference_digest="target-sha",
        bfm_id="heft_wbc",
        checkpoint_digest="policy-sha",
        robot_model_digest="model-sha",
        initial_physical_state_digest="physical-sha",
        control_period=0.1,
        trace=_trace(),
        planning=() if kind == "baseline" else (_diagnostics(),),
    )


def test_records_own_float64_read_only_values_without_aliasing() -> None:
    reference_motion = _motion_command()
    measurements = _measurements(3)
    timeline = np.arange(3, dtype=np.float32) * 0.1
    qpos = np.zeros((3, 4), dtype=np.float32)
    trace = ExecutionTrace(
        timeline=timeline,
        reference_motion=reference_motion,
        applied_command=_motion_command(joint=0.25),
        executed_trajectory=measurements,
        qpos=qpos,
        qvel=np.zeros((3, 3), dtype=np.float32),
        valid=np.ones(3, dtype=bool),
        root_body_name="pelvis",
    )

    timeline[:] = 9.0
    reference_motion.joint_position[:] = 9.0
    measurements.arrays[MeasurementPrimitive.JOINT_POSITION][:] = 9.0
    qpos[:] = 9.0

    assert trace.timeline.dtype == np.float64
    np.testing.assert_allclose(trace.timeline, (0.0, 0.1, 0.2))
    np.testing.assert_array_equal(trace.reference_motion.joint_position, 0.0)
    assert trace.executed_trajectory is not None
    np.testing.assert_array_equal(
        trace.executed_trajectory.arrays[MeasurementPrimitive.JOINT_POSITION], 0.0
    )
    np.testing.assert_array_equal(trace.qpos, 0.0)
    for array in (
        trace.timeline,
        trace.reference_motion.joint_position,
        trace.executed_trajectory.arrays[MeasurementPrimitive.JOINT_POSITION],
        trace.qpos,
        trace.valid,
    ):
        assert not array.flags.writeable


def test_trace_records_true_failure_without_fabricating_a_suffix() -> None:
    trace = _trace(valid_steps=1)

    assert not trace.success
    assert trace.failure_step == 1
    assert trace.failure_reason == "fell"
    assert trace.num_steps == 3
    assert trace.valid_steps == 1
    assert trace.qpos.shape[0] == 1
    assert trace.executed_trajectory is not None and trace.executed_trajectory.num_steps == 1

    with pytest.raises(ValueError, match="first invalid"):
        ExecutionTrace(
            timeline=np.arange(3, dtype=float) * 0.1,
            reference_motion=_motion_command(),
            applied_command=_motion_command(),
            executed_trajectory=_measurements(1),
            qpos=np.zeros((1, 4)),
            qvel=np.zeros((1, 3)),
            valid=np.array([True, False, False]),
            root_body_name="pelvis",
            failure_step=2,
            failure_reason="late label",
        )


def test_planning_diagnostics_keep_cost_currencies_distinct_and_immutable() -> None:
    quantiles = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    deformation = {"joint_rmse_rad": 0.2}
    diagnostics = PlanningDiagnostics(
        update_step=2,
        selected_source="proposal",
        selected_candidate_id="proposal-sha",
        selected_cost=0.8,
        best_observed_candidate_id="best-sha",
        best_observed_cost=1.0,
        proposal_candidate_id="proposal-sha",
        proposal_cost=0.8,
        cost_quantiles=quantiles,
        candidate_deformation=deformation,
        candidate_evaluations=9,
        scenario_count=3,
        scenario_digest="sha256:scenario",
        logical_rollouts=27,
        physical_rollouts=32,
        penalized_candidate_evaluations=2,
        invalid_scenario_rollouts=3,
        update_wall_seconds=0.5,
    )
    quantiles[:] = 99.0
    deformation["joint_rmse_rad"] = 99.0

    assert diagnostics.best_observed_candidate_id != diagnostics.proposal_candidate_id
    assert diagnostics.selected_cost == diagnostics.proposal_cost
    np.testing.assert_array_equal(diagnostics.cost_quantiles, [[1, 2, 3, 4, 5]])
    assert diagnostics.candidate_deformation["joint_rmse_rad"] == 0.2
    assert isinstance(diagnostics.candidate_deformation, MappingProxyType)
    assert not diagnostics.cost_quantiles.flags.writeable
    assert diagnostics.update_wall_seconds == 0.5
    assert diagnostics.scenario_digest == "sha256:scenario"
    with pytest.raises(TypeError):
        diagnostics.candidate_deformation["x"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("wall_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_planning_diagnostics_require_a_positive_measured_wall_time(
    wall_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="update_wall_seconds"):
        dataclasses.replace(_diagnostics(), update_wall_seconds=wall_seconds)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"logical_rollouts": 9}, "must equal"),
        ({"penalized_candidate_evaluations": -1}, "penalized_candidate"),
        ({"penalized_candidate_evaluations": 9}, "penalized_candidate"),
        ({"invalid_scenario_rollouts": -1}, "invalid_scenario"),
        ({"invalid_scenario_rollouts": 9}, "invalid_scenario"),
        ({"scenario_digest": ""}, "scenario_digest"),
        ({"scenario_digest": "not-a-digest"}, "sha256"),
    ],
)
def test_planning_diagnostics_reject_inconsistent_robust_ledgers(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(_diagnostics(), **changes)


def test_run_and_pair_roles_are_explicit() -> None:
    baseline = _run("baseline")
    optimized = _run("optimized")
    pair = PairedRunArtifact(baseline, optimized)

    assert pair.baseline.planning == ()
    assert pair.optimized.planning[0].logical_rollouts == 8
    with pytest.raises(ValueError, match="baseline"):
        PairedRunArtifact(optimized, optimized)
