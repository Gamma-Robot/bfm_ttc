from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sampling_optimizers import ConvergenceCriteria
from bfm_adapters import MotionCommand
from command_optimization.planning.evaluation import EvaluationLedger, WaveDiagnostics
from command_optimization.planning.update import PlanningUpdateDiagnostics, PlanningUpdateResult
from command_optimization.loop import _candidate_deformation, _validate_execution, run_command_optimization
from command_optimization.simulation.command_execution import ExecutionResult
from command_optimization.simulation.measurements import MeasurementPrimitive, ControlStepMeasurements
from command_optimization.simulation.execution_state import ExecutionState


JOINTS = ("joint",)
BODIES = ("pelvis",)


def _motion_command(values: np.ndarray) -> MotionCommand:
    joint = np.asarray(values, dtype=np.float64).reshape(1, -1, 1)
    num_steps = joint.shape[1]
    body_position = np.zeros((1, num_steps, 1, 3), dtype=np.float64)
    body_position[..., 0] = joint
    orientation = np.zeros((1, num_steps, 1, 4), dtype=np.float64)
    orientation[..., 0] = 1.0
    vectors = np.zeros_like(body_position)
    return MotionCommand(
        query_offsets=tuple(range(num_steps)),
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body_position,
        body_orientation_world=orientation,
        body_linear_velocity_world=vectors,
        body_angular_velocity_world=vectors.copy(),
    )


def _execution_state(*, control_step: int = 7, time: float = 0.5, cursor: int = 1) -> ExecutionState:
    return ExecutionState(
        qpos=np.zeros((1, 1)),
        qvel=np.zeros((1, 1)),
        control_step=control_step,
        time=time,
        motion_command_index=cursor,
        command=None,
        bfm_state={"steps": control_step},
    )


class _Space:
    latent_shape = (2, 1)

    def __init__(self, reference: MotionCommand, index: int) -> None:
        self.reference = reference
        self.index = index
        self.shift_calls: list[tuple[np.ndarray, int]] = []

    def shift(self, latent: np.ndarray, num_steps: int) -> np.ndarray:
        self.shift_calls.append((latent.copy(), num_steps))
        return latent + 10.0 + self.index


class _Executor:
    physics_substeps = 4

    def __init__(self) -> None:
        self.calls: list[tuple[ExecutionState, MotionCommand, int]] = []
        self.right_contexts: list[MotionCommand | None] = []

    def execute(self, execution_state, motion_command, *, num_steps, right_context=None, requires):
        assert requires == frozenset(MeasurementPrimitive)
        self.calls.append((execution_state, motion_command, num_steps))
        self.right_contexts.append(right_context)
        joint = motion_command.joint_position[:, :num_steps].copy()
        body = motion_command.body_position_world[:, :num_steps].copy()
        orientation = motion_command.body_orientation_world[:, :num_steps].copy()
        measurements = ControlStepMeasurements(
            JOINTS,
            BODIES,
            {
                MeasurementPrimitive.JOINT_POSITION: joint,
                MeasurementPrimitive.JOINT_VELOCITY: np.zeros_like(joint),
                MeasurementPrimitive.BODY_POSITION_WORLD: body,
                MeasurementPrimitive.BODY_ORIENTATION_WORLD: orientation,
            },
        )
        end = float(joint[0, -1, 0])
        return ExecutionResult(
            measurements=measurements,
            qpos=np.full((1, num_steps, 1), end),
            qvel=np.full((1, num_steps, 1), -end),
            final_execution_state=ExecutionState(
                qpos=np.full((1, 1), end),
                qvel=np.full((1, 1), -end),
                control_step=execution_state.control_step + num_steps,
                time=execution_state.time + 0.1 * num_steps,
                motion_command_index=execution_state.motion_command_index + num_steps,
                command=None,
                bfm_state={"steps": execution_state.control_step + num_steps},
                motion_command_history=execution_state.motion_command_history,
            ),
        )


def test_execution_time_validation_accepts_float32_drift_but_rejects_step_error() -> None:
    before = _execution_state()
    execution = _Executor().execute(
        before,
        _motion_command(np.arange(2.0)),
        num_steps=2,
        requires=frozenset(MeasurementPrimitive),
    )
    expected = before.time + 0.2
    drifted = replace(
        execution,
        final_execution_state=replace(execution.final_execution_state, time=expected + 3e-6),
    )

    _validate_execution(drifted, before, num_steps=2, control_period=0.1, physics_substeps=4)

    wrong = replace(
        execution,
        final_execution_state=replace(execution.final_execution_state, time=expected + 1e-3),
    )
    with pytest.raises(RuntimeError, match="did not advance"):
        _validate_execution(wrong, before, num_steps=2, control_period=0.1, physics_substeps=4)


@pytest.mark.parametrize(
    "physics_period,before_substeps,physics_substeps",
    [(0.005, 48_800, 4), (0.001, 8_000, 20), (0.0005, 16_000, 40)],
)
def test_execution_time_validation_accepts_accumulated_float32_clock_drift(
    physics_period: float, before_substeps: int, physics_substeps: int
) -> None:
    after_substeps = before_substeps + 20 * physics_substeps
    times = np.concatenate(
        (
            np.zeros(1, dtype=np.float32),
            np.cumsum(np.full(after_substeps, np.float32(physics_period)), dtype=np.float32),
        )
    )
    before = _execution_state(time=float(times[before_substeps]))
    execution = _Executor().execute(
        before,
        _motion_command(np.arange(20.0)),
        num_steps=20,
        requires=frozenset(MeasurementPrimitive),
    )
    execution = replace(
        execution,
        final_execution_state=replace(
            execution.final_execution_state,
            time=float(times[after_substeps]),
        ),
    )

    _validate_execution(
        execution, before, num_steps=20, control_period=0.02,
        physics_substeps=physics_substeps,
    )

    wrong = replace(
        execution,
        final_execution_state=replace(
            execution.final_execution_state,
            time=execution.final_execution_state.time + 1.0e-3,
        ),
    )
    with pytest.raises(RuntimeError, match="did not advance"):
        _validate_execution(
            wrong, before, num_steps=20, control_period=0.02,
            physics_substeps=physics_substeps,
        )


def _ledger(*, penalized: bool = False, scenario_count: int = 1) -> EvaluationLedger:
    wave = WaveDiagnostics(
        wave_index=0,
        candidate_start=0,
        candidate_evaluations=2,
        scenario_count=scenario_count,
        logical_rollouts=2 * scenario_count,
        physical_rollouts=2 * scenario_count,
        penalized_candidate_ids=(0, 1) if penalized else (),
        invalid_scenario_rollouts=2 * scenario_count if penalized else 0,
        failure="failed" if penalized else None,
        minimum_cost=1.0,
        median_cost=1.5,
        maximum_cost=2.0,
    )
    return EvaluationLedger(
        candidate_evaluations=2,
        scenario_count=scenario_count,
        scenario_digest=None if scenario_count == 1 else "sha256:scenario",
        logical_rollouts=2 * scenario_count,
        physical_rollouts=2 * scenario_count,
        waves=(wave,),
        cost_quantiles=(1.0, 1.25, 1.5, 1.75, 2.0),
    )


def _update(
    reference: MotionCommand,
    value: float,
    *,
    penalized: bool = False,
    scenario_count: int = 1,
):
    candidate = np.full((2, 1), value, dtype=np.float64)
    proposal = np.full((2, 1), value + 1.0, dtype=np.float64)
    motion_command = _motion_command(reference.joint_position[0, :, 0] + value)
    diagnostics = PlanningUpdateDiagnostics(
        iterations=(),
        evaluations=(
            _ledger(penalized=penalized, scenario_count=scenario_count),
        ),
        best_observed_candidate=candidate,
        best_observed_cost=1.0,
        proposal_candidate=proposal,
        proposal_cost=2.0,
        selected_source="best_observed",
    )
    return PlanningUpdateResult(candidate, motion_command, 1.0, diagnostics)


def test_planning_update_diagnostics_reject_mixed_scenario_identity() -> None:
    diagnostics = _update(_motion_command(np.arange(2.0)), 1.0).diagnostics
    changed = replace(_ledger(), scenario_digest="sha256:different")

    with pytest.raises(ValueError, match="one scenario count and digest"):
        replace(diagnostics, evaluations=(*diagnostics.evaluations, changed))


@pytest.mark.parametrize("iteration_limit", [3, ConvergenceCriteria(3, 0.0)])
def test_loop_slices_pads_warm_starts_and_stitches_post_step_trace(monkeypatch, iteration_limit) -> None:
    reference_future = _motion_command(np.arange(1.0, 6.0))
    spaces: list[_Space] = []
    evaluator_calls: list[
        tuple[object, MotionCommand, MotionCommand | None, ExecutionState]
    ] = []
    update_calls: list[dict[str, object]] = []
    executor = _Executor()

    def space_factory(reference: MotionCommand) -> _Space:
        space = _Space(reference, len(spaces))
        spaces.append(space)
        return space

    def evaluator_factory(space, reference_motion, right_context, execution_state):
        evaluator_calls.append((space, reference_motion, right_context, execution_state))
        return SimpleNamespace()

    def fake_update(optimizer, evaluator, **kwargs):
        del optimizer, evaluator
        update_calls.append(kwargs)
        return _update(spaces[-1].reference, float(len(update_calls)))

    monkeypatch.setattr("command_optimization.loop.run_planning_update", fake_update)
    optimizer = SimpleNamespace(num_knots=2, dim=1)
    result = run_command_optimization(
        reference_future=reference_future,
        initial_execution_state=_execution_state(),
        optimizer=optimizer,
        search_space_factory=space_factory,
        evaluator_factory=evaluator_factory,
        executor=executor,
        horizon_steps=4,
        replanning_interval_steps=2,
        control_period=0.1,
        seed=40,
        iterations=iteration_limit,
        root_body_name="pelvis",
        body_names=BODIES,
        future_context_steps=2,
    )

    assert all(call["iterations"] is iteration_limit for call in update_calls)
    assert [call["seed"] for call in update_calls] == [40, 41, 42]
    assert [call["initial_mean"] is None for call in update_calls] == [True, False, False]
    np.testing.assert_array_equal(update_calls[1]["initial_mean"], 12.0)
    np.testing.assert_array_equal(update_calls[2]["initial_mean"], 14.0)
    assert [call[1].joint_position[0, :, 0].tolist() for call in evaluator_calls] == [
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
    ]
    assert [call[3].control_step for call in evaluator_calls] == [7, 9, 11]
    assert [call[3].motion_command_index for call in evaluator_calls] == [1, 3, 5]
    assert [call[2] for call in executor.calls] == [2, 2, 1]
    for call in evaluator_calls:
        context = call[2]
        assert context is not None
        np.testing.assert_array_equal(context.joint_position[0, :, 0], [5.0, 5.0])
        assert not context.joint_position.flags.writeable
    for context in executor.right_contexts:
        assert context is not None
        np.testing.assert_array_equal(context.joint_position[0, :, 0], [5.0, 5.0])

    trace = result.trace
    np.testing.assert_allclose(trace.timeline, [0.6, 0.7, 0.8, 0.9, 1.0])
    np.testing.assert_array_equal(trace.reference_motion.joint_position[0, :, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        trace.applied_command.joint_position[0, :, 0], [2, 3, 5, 6, 8]
    )
    assert trace.valid_steps == 5
    assert trace.qpos.shape == (5, 1)
    assert trace.qvel.shape == (5, 1)
    assert trace.executed_trajectory is not None and trace.executed_trajectory.num_steps == 5
    assert [record.update_step for record in result.planning] == [0, 2, 4]
    assert all(record.selected_candidate_id.startswith("sha256:") for record in result.planning)
    assert all(record.logical_rollouts == 2 for record in result.planning)
    assert all(record.penalized_candidate_evaluations == 0 for record in result.planning)
    assert all(record.invalid_scenario_rollouts == 0 for record in result.planning)
    assert all(
        np.isfinite(record.update_wall_seconds) and record.update_wall_seconds > 0.0
        for record in result.planning
    )


def test_execution_prefix_keeps_later_reference_frames_as_planning_context(
    monkeypatch,
) -> None:
    reference_future = _motion_command(np.arange(1.0, 7.0))
    spaces: list[_Space] = []
    executor = _Executor()

    def space_factory(reference: MotionCommand) -> _Space:
        space = _Space(reference, len(spaces))
        spaces.append(space)
        return space

    monkeypatch.setattr(
        "command_optimization.loop.run_planning_update",
        lambda *_args, **_kwargs: _update(spaces[-1].reference, 1.0),
    )
    result = run_command_optimization(
        reference_future=reference_future,
        initial_execution_state=_execution_state(),
        optimizer=SimpleNamespace(num_knots=2, dim=1),
        search_space_factory=space_factory,
        evaluator_factory=lambda *_args: SimpleNamespace(),
        executor=executor,
        horizon_steps=4,
        replanning_interval_steps=2,
        control_period=0.1,
        seed=0,
        iterations=1,
        root_body_name="pelvis",
        body_names=BODIES,
        execution_steps=3,
        future_context_steps=1,
    )

    assert [space.reference.joint_position[0, :, 0].tolist() for space in spaces] == [
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert [call[2] for call in executor.calls] == [2, 1]
    np.testing.assert_array_equal(
        result.trace.reference_motion.joint_position[0, :, 0], [1.0, 2.0, 3.0]
    )
    assert result.trace.num_steps == 3


def test_all_penalized_update_is_rejected_before_physical_execution(monkeypatch) -> None:
    reference_motion = _motion_command(np.arange(1.0, 4.0))
    executor = _Executor()
    space = _Space(_motion_command(np.arange(1.0, 5.0)), 0)

    monkeypatch.setattr(
        "command_optimization.loop.run_planning_update",
        lambda *args, **kwargs: _update(
            space.reference, 1.0, penalized=True, scenario_count=4
        ),
    )
    with pytest.raises(RuntimeError, match="penalized every candidate"):
        run_command_optimization(
            reference_future=reference_motion,
            initial_execution_state=_execution_state(),
            optimizer=SimpleNamespace(num_knots=2, dim=1),
            search_space_factory=lambda reference: space,
            evaluator_factory=lambda *args: SimpleNamespace(),
            executor=executor,
            horizon_steps=4,
            replanning_interval_steps=2,
            control_period=0.1,
            seed=0,
            iterations=1,
            root_body_name="pelvis",
            body_names=BODIES,
        )

    assert executor.calls == []


def test_deformation_diagnostics_exclude_policy_only_bodies() -> None:
    single_body = _motion_command(np.zeros(2))
    reference_motion = replace(
        single_body,
        body_names=("policy_only", "pelvis"),
        **{
            name: np.repeat(getattr(single_body, name), 2, axis=2)
            for name in (
                "body_position_world",
                "body_orientation_world",
                "body_linear_velocity_world",
                "body_angular_velocity_world",
            )
        },
    )
    candidate = replace(
        reference_motion,
        joint_position=np.ones_like(reference_motion.joint_position),
        body_position_world=reference_motion.body_position_world.copy(),
        body_orientation_world=reference_motion.body_orientation_world.copy(),
    )
    candidate.body_position_world[:, :, 0] = 10.0
    candidate.body_orientation_world[:, :, 0] = [0.0, 1.0, 0.0, 0.0]
    reference_motion.body_orientation_world[:, :, 1] *= 0.9999999
    candidate.body_orientation_world[:, :, 1] *= -0.9999999

    metrics = _candidate_deformation(candidate, reference_motion, [1])

    assert metrics == {
        "joint_position_rmse_rad": 1.0,
        "body_position_rmse_m": 0.0,
        "body_orientation_rmse_rad": 0.0,
    }
    candidate.body_position_world[:, :, 1, 0] = 2.0
    candidate.body_orientation_world[:, :, 1] = [0.0, 0.0, 1.0, 0.0]
    changed = _candidate_deformation(candidate, reference_motion, [1])
    assert changed["body_position_rmse_m"] == pytest.approx(2.0)
    assert changed["body_orientation_rmse_rad"] == pytest.approx(np.pi)
