from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from bfm_adapters import MotionCommand, PDCommand
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.planning.evaluation import (
    EvaluationLedger,
    RolloutEvaluator,
    WaveDiagnostics,
)
from command_optimization.simulation.measurements import (
    ControlStepMeasurements,
    MeasurementPrimitive,
)
from command_optimization.simulation.randomization import ScenarioBatch
from command_optimization.simulation.execution_state import ExecutionState


JOINTS = ("joint",)
BODIES = ("pelvis",)
HORIZON = 2


class _Space:
    latent_shape = (HORIZON, 1)

    def decode(self, latents: np.ndarray) -> MotionCommand:
        batch = latents.shape[0]
        joint = np.array(latents, dtype=np.float64, copy=True)
        body_position = np.zeros((batch, HORIZON, 1, 3))
        body_orientation = np.zeros((batch, HORIZON, 1, 4))
        body_orientation[..., 0] = 1.0
        body_velocity = np.zeros_like(body_position)
        return MotionCommand(
            query_offsets=tuple(range(HORIZON)),
            joint_names=JOINTS,
            body_names=BODIES,
            joint_position=joint,
            joint_velocity=np.zeros_like(joint),
            body_position_world=body_position,
            body_orientation_world=body_orientation,
            body_linear_velocity_world=body_velocity,
            body_angular_velocity_world=body_velocity.copy(),
        )


class _Adapter:
    def __init__(self) -> None:
        self.expansions: list[int] = []

    def expand_state(self, state: object, batch_size: int) -> object:
        assert state == ("one-lane",)
        self.expansions.append(batch_size)
        return tuple(("lane", index) for index in range(batch_size))


class _Executor:
    def __init__(self, batch: int, *, fail_at: float | None = None) -> None:
        self.batch = batch
        self.fail_at = fail_at
        self.motion_command_values: list[np.ndarray] = []
        self.execution_states: list[ExecutionState] = []
        self.right_contexts: list[MotionCommand | None] = []
        self.executed_steps: list[int] = []
        self.reduced_calls = 0

    def rollout(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        *,
        num_steps: int,
        right_context: MotionCommand | None,
        requires: frozenset[MeasurementPrimitive],
    ) -> object:
        value = motion_command.joint_position[:, :, 0]
        self.motion_command_values.append(value.copy())
        self.execution_states.append(execution_state)
        self.right_contexts.append(right_context)
        self.executed_steps.append(num_steps)
        assert num_steps == HORIZON
        assert execution_state.qpos.shape[0] == self.batch
        assert len(execution_state.bfm_state) == self.batch
        if execution_state.command is not None:
            assert execution_state.command.position_target.shape == (self.batch, 1)
        if self.fail_at is not None and value[0, 0] == self.fail_at:
            raise RuntimeError("synthetic rollout failure")

        body_position = np.zeros((self.batch, HORIZON, 1, 3))
        body_orientation = np.zeros((self.batch, HORIZON, 1, 4))
        body_orientation[..., 0] = 1.0
        arrays = {
            MeasurementPrimitive.JOINT_POSITION: motion_command.joint_position.copy(),
            MeasurementPrimitive.JOINT_VELOCITY: np.zeros_like(
                motion_command.joint_position
            ),
            MeasurementPrimitive.BODY_POSITION_WORLD: body_position,
            MeasurementPrimitive.BODY_ORIENTATION_WORLD: body_orientation,
        }
        return ControlStepMeasurements(JOINTS, BODIES, arrays)

    def rollout_reduced(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        reducer: object,
        *,
        num_steps: int,
        right_context: MotionCommand | None,
    ) -> np.ndarray:
        del motion_command, num_steps, right_context
        self.execution_states.append(execution_state)
        self.reduced_calls += 1
        return reducer.finalize()  # type: ignore[attr-defined,no-any-return]


class _Factory:
    def __init__(self, *, fail_at: float | None = None) -> None:
        self.fail_at = fail_at
        self.capacities: list[int] = []
        self.executor: _Executor | None = None

    def __call__(self, capacity: int) -> _Executor:
        self.capacities.append(capacity)
        self.executor = _Executor(capacity, fail_at=self.fail_at)
        return self.executor


class _Reducer:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.costs: np.ndarray | None = None

    def prepare(
        self, reference_motion: MotionCommand, candidate: MotionCommand
    ) -> None:
        self.prepare_calls += 1
        self.costs = np.mean(
            (candidate.joint_position - reference_motion.joint_position) ** 2, axis=-1
        ).sum(axis=1)

    def record(self, step: int, state_view: object) -> None:
        del step, state_view

    def finalize(self) -> np.ndarray:
        assert self.costs is not None
        return self.costs.copy()


class _ReducerFactory:
    def __init__(self) -> None:
        self.capacities: list[int] = []
        self.reducer: _Reducer | None = None

    def __call__(self, capacity: int) -> _Reducer:
        self.capacities.append(capacity)
        self.reducer = _Reducer()
        return self.reducer


def _motion_command(batch: int = 1) -> MotionCommand:
    return _Space().decode(np.zeros((batch, HORIZON, 1), dtype=np.float64))


def _constant_motion_command(value: float, *, batch: int = 1) -> MotionCommand:
    latent = np.full((batch, HORIZON, 1), value, dtype=np.float64)
    return _Space().decode(latent)


def _history(value: float, *, batch: int = 1) -> MotionCommand:
    motion_command = _constant_motion_command(value, batch=batch)
    return MotionCommand(
        query_offsets=(-HORIZON, -1),
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        joint_position=motion_command.joint_position,
        joint_velocity=motion_command.joint_velocity,
        body_position_world=motion_command.body_position_world,
        body_orientation_world=motion_command.body_orientation_world,
        body_linear_velocity_world=motion_command.body_linear_velocity_world,
        body_angular_velocity_world=motion_command.body_angular_velocity_world,
    )


def _objective() -> MotionTrackingObjective:
    return MotionTrackingObjective(
        body_names=BODIES,
        tracking_joint_position_weight=1.0,
        tracking_joint_velocity_weight=0.0,
        tracking_body_position_weight=0.0,
        tracking_body_orientation_weight=0.0,
        deformation_joint_position_weight=0.0,
        deformation_body_position_weight=0.0,
        deformation_body_orientation_weight=0.0,
        joint_speed_weight=0.0,
        terminal_joint_position_weight=0.0,
        terminal_body_position_weight=0.0,
    )


def _execution_state(
    *, batch: int = 1, command: bool = True, history: MotionCommand | None = None
) -> ExecutionState:
    initial_command = None
    if command:
        value = np.ones((1, 1))
        initial_command = PDCommand(JOINTS, value, value, value, value, value)
    return ExecutionState(
        qpos=np.zeros((batch, 1)),
        qvel=np.zeros((batch, 1)),
        qacc_warmstart=np.full((batch, 1), 3.0),
        control_step=4,
        time=0.08,
        motion_command_index=4,
        command=initial_command,
        bfm_state=("one-lane",),
        motion_command_history=history,
    )


def _evaluator(
    *,
    max_worlds: int = 5,
    space: object | None = None,
    adapter: _Adapter | None = None,
    factory: Callable[[int], _Executor] | None = None,
    objective: object | None = None,
    execution_state: ExecutionState | None = None,
    penalty: float = 99.0,
    right_context: MotionCommand | None = None,
    reducer_factory: _ReducerFactory | None = None,
    scenarios: ScenarioBatch | None = None,
    risk_strategy: str = "mean",
    scenario_configurer: Callable[[int, ScenarioBatch], None] | None = None,
) -> RolloutEvaluator:
    return RolloutEvaluator(
        search_space=space or _Space(),  # type: ignore[arg-type]
        objective=objective or _objective(),  # type: ignore[arg-type]
        reference_motion=_motion_command(),
        execution_state=execution_state or _execution_state(),
        bfm_adapter=adapter or _Adapter(),  # type: ignore[arg-type]
        executor_factory=factory or _Factory(),  # type: ignore[arg-type]
        max_worlds=max_worlds,
        failure_penalty=penalty,
        right_context=right_context,
        reducer_factory=reducer_factory,
        scenarios=scenarios,
        risk_strategy=risk_strategy,  # type: ignore[arg-type]
        scenario_configurer=scenario_configurer,
    )


@pytest.mark.parametrize("reduced", (False, True))
def test_valid_cost_must_be_strictly_below_failure_penalty(reduced: bool) -> None:
    evaluator = _evaluator(
        penalty=2.0,
        reducer_factory=_ReducerFactory() if reduced else None,
    )
    candidate = np.ones((1, HORIZON, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="failure_penalty must exceed"):
        evaluator.evaluate(candidate)


def test_host_evaluate_preserves_order_and_dispatches_candidate_groups() -> None:
    adapter = _Adapter()
    factory = _Factory()
    evaluator = _evaluator(max_worlds=5, adapter=adapter, factory=factory)
    values = np.arange(7, dtype=np.float64)
    candidates = np.repeat(values[:, None, None], HORIZON, axis=1)

    result = evaluator.evaluate(candidates)
    costs = result.costs

    np.testing.assert_array_equal(costs, 2.0 * values**2)
    assert costs.dtype == np.float64
    assert factory.capacities == [1]
    assert adapter.expansions == [1]
    assert factory.executor is not None
    assert all(
        np.array_equal(state.qacc_warmstart, np.full_like(state.qvel, 3.0))
        for state in factory.executor.execution_states
    )
    np.testing.assert_array_equal(
        np.concatenate(factory.executor.motion_command_values)[:, 0], values
    )

    record = result.ledger
    assert record.logical_rollouts == 7
    assert record.physical_rollouts == 7
    assert [(wave.candidate_start, wave.candidate_stop) for wave in record.waves] == [
        (index, index + 1) for index in range(7)
    ]
    assert all(wave.physical_rollouts == 1 for wave in record.waves)
    assert record.waves[1].penalized_candidate_ids == ()
    with pytest.raises(FrozenInstanceError):
        record.logical_rollouts = 8  # type: ignore[misc]


def test_reduced_evaluation_reuses_one_capacity_bound_reducer() -> None:
    factory = _Factory()
    reducers = _ReducerFactory()
    evaluator = _evaluator(
        max_worlds=5, factory=factory, reducer_factory=reducers
    )
    values = np.arange(7, dtype=np.float64)
    candidates = np.repeat(values[:, None, None], HORIZON, axis=1)

    result = evaluator.evaluate(candidates)

    np.testing.assert_array_equal(result.costs, 2.0 * values**2)
    assert reducers.capacities == [5]
    assert reducers.reducer is not None
    assert reducers.reducer.prepare_calls == 2
    assert factory.executor is not None
    assert factory.executor.reduced_calls == 2
    assert factory.executor.motion_command_values == []
    assert result.ledger.physical_rollouts == 10
    assert [(wave.candidate_start, wave.candidate_stop) for wave in result.ledger.waves] == [
        (0, 5),
        (5, 7),
    ]


def test_reduced_rollout_failure_records_dispatched_penalties() -> None:
    class FailingReducedExecutor(_Executor):
        def rollout_reduced(self, *args, **kwargs) -> np.ndarray:
            del args, kwargs
            raise RuntimeError("synthetic reduced rollout failure")

    evaluator = _evaluator(
        max_worlds=2,
        factory=FailingReducedExecutor,
        reducer_factory=_ReducerFactory(),
        penalty=99.0,
    )

    result = evaluator.evaluate(np.zeros((2, HORIZON, 1), dtype=np.float64))

    np.testing.assert_array_equal(result.costs, 99.0)
    assert result.ledger.logical_rollouts == 2
    assert result.ledger.physical_rollouts == 2
    wave = result.ledger.waves[0]
    assert wave.penalized_candidate_ids == (0, 1)
    assert "synthetic reduced rollout failure" in (wave.failure or "")


def test_reduced_evaluation_penalizes_only_the_nonfinite_lane() -> None:
    class MixedReducer(_Reducer):
        def finalize(self) -> np.ndarray:
            return np.array([np.nan, 7.0], dtype=np.float64)

    class MixedReducerFactory:
        def __call__(self, capacity: int) -> MixedReducer:
            assert capacity == 2
            return MixedReducer()

    evaluator = _evaluator(
        max_worlds=2,
        reducer_factory=MixedReducerFactory(),  # type: ignore[arg-type]
        penalty=99.0,
    )

    result = evaluator.evaluate(np.zeros((2, HORIZON, 1), dtype=np.float64))

    np.testing.assert_array_equal(result.costs, [99.0, 7.0])
    wave = result.ledger.waves[0]
    assert wave.penalized_candidate_ids == (0,)
    assert wave.failure == "non-finite objective cost"


def test_wave_failure_and_nonfinite_costs_become_finite_penalties() -> None:
    class NonfiniteObjective:
        requires = frozenset(MeasurementPrimitive)

        def evaluate(self, measurements, reference_motion, candidate):
            del measurements, reference_motion
            value = candidate.joint_position[:, 0, 0].copy()
            value[value == 4.0] = np.nan
            return SimpleNamespace(total=value)

    factory = _Factory(fail_at=2.0)
    evaluator = _evaluator(
        max_worlds=2,
        factory=factory,
        objective=NonfiniteObjective(),
        penalty=99.0,
    )
    values = np.arange(5, dtype=np.float64)
    candidates = np.repeat(values[:, None, None], HORIZON, axis=1)

    result = evaluator.evaluate(candidates)
    costs = result.costs

    np.testing.assert_array_equal(costs, [0.0, 1.0, 99.0, 3.0, 99.0])
    assert np.isfinite(costs).all()
    waves = result.ledger.waves
    assert waves[0].failure is None
    assert waves[2].penalized_candidate_ids == (2,)
    assert "synthetic rollout failure" in (waves[2].failure or "")
    assert waves[4].penalized_candidate_ids == (4,)
    assert waves[4].failure == "non-finite objective cost"


@pytest.mark.parametrize("scenario_count", [1, 2])
def test_host_nonfinite_measurement_penalizes_only_its_candidate(
    scenario_count: int,
) -> None:
    class InvalidMeasurementExecutor(_Executor):
        def rollout(self, execution_state, motion_command, **kwargs):
            measurements = super().rollout(execution_state, motion_command, **kwargs)
            arrays = {key: value.copy() for key, value in measurements.arrays.items()}
            if motion_command.joint_position[0, 0, 0] == 1.0:
                arrays[MeasurementPrimitive.JOINT_POSITION][0, 0, 0] = np.nan
            return ControlStepMeasurements(JOINTS, BODIES, arrays)

    scenarios = None if scenario_count == 1 else _robust_scenarios()
    state = ExecutionState(
        qpos=np.zeros((1, 1)),
        qvel=np.zeros((1, 7)),
        control_step=0,
        time=0.0,
        motion_command_index=0,
        command=None,
        bfm_state=("one-lane",),
    )
    candidates = np.repeat(
        np.arange(3, dtype=np.float64)[:, None, None], HORIZON, axis=1
    )
    for capacity in (scenario_count, 3 * scenario_count + 1):
        evaluator = _evaluator(
            max_worlds=capacity,
            factory=InvalidMeasurementExecutor,
            execution_state=state,
            scenarios=scenarios,
        )
        result = evaluator.evaluate(candidates)

        np.testing.assert_array_equal(result.costs, [0.0, 99.0, 8.0])
        assert result.ledger.logical_rollouts == 3 * scenario_count
        assert result.ledger.physical_rollouts == 3 * scenario_count
        assert result.ledger.penalized_candidate_evaluations == 1
        assert result.ledger.waves[1].penalized_candidate_ids == (1,)
        assert "joint_position must be finite" in (
            result.ledger.waves[1].failure or ""
        )
        assert all(
            wave.physical_rollouts == scenario_count for wave in result.ledger.waves
        )


def test_input_and_construction_errors_are_rejected_before_dispatch() -> None:
    evaluator = _evaluator()
    good = np.zeros((1, HORIZON, 1), dtype=np.float64)
    for bad in (
        good.astype(np.float32),
        np.zeros((HORIZON, 1), dtype=np.float64),
        np.zeros((1, HORIZON, 2), dtype=np.float64),
        np.full((1, HORIZON, 1), np.nan),
    ):
        with pytest.raises(ValueError):
            evaluator.evaluate(bad)

    with pytest.raises(ValueError, match="max_worlds"):
        _evaluator(max_worlds=0)
    for penalty in (np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="failure_penalty"):
            _evaluator(penalty=penalty)
    with pytest.raises(ValueError, match="exactly one lane"):
        _evaluator(execution_state=_execution_state(batch=2, command=False))


def test_pre_dispatch_decode_failure_is_not_converted_to_a_rollout_penalty() -> None:
    class BrokenSpace(_Space):
        def decode(self, latents: np.ndarray) -> MotionCommand:
            if np.any(latents):
                raise RuntimeError("synthetic decode failure")
            return super().decode(latents)

    evaluator = _evaluator(space=BrokenSpace())

    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        evaluator.evaluate(np.ones((1, HORIZON, 1), dtype=np.float64))


def test_pre_dispatch_executor_failure_is_not_converted_to_a_rollout_penalty() -> None:
    class FailSingletonFactory:
        def __call__(self, capacity: int) -> _Executor:
            if capacity == 1:
                raise RuntimeError(f"cannot construct capacity {capacity}")
            return _Executor(capacity)

    evaluator = _evaluator(
        max_worlds=3, factory=FailSingletonFactory(), reducer_factory=_ReducerFactory()
    )

    with pytest.raises(RuntimeError, match="cannot construct capacity 1"):
        evaluator.evaluate(np.zeros((1, HORIZON, 1), dtype=np.float64))


def test_decode_motion_command_returns_an_owned_immutable_motion_command() -> None:
    evaluator = _evaluator()
    candidate = np.ones((HORIZON, 1), dtype=np.float64)
    motion_command = evaluator.decode_motion_command(candidate)
    candidate[:] = 7.0
    np.testing.assert_array_equal(motion_command.joint_position, 1.0)
    assert not motion_command.joint_position.flags.writeable


def test_evaluator_uses_exact_candidate_horizon_and_owned_reference_context() -> None:
    context = _constant_motion_command(50.0)
    factory = _Factory()
    evaluator = _evaluator(
        max_worlds=3, factory=factory, right_context=context
    )
    context.joint_position[:] = 500.0

    evaluator.evaluate(np.ones((2, HORIZON, 1), dtype=np.float64))

    assert factory.executor is not None
    assert factory.executor.executed_steps == [HORIZON, HORIZON]
    observed = factory.executor.right_contexts[0]
    assert observed is not None
    np.testing.assert_array_equal(observed.joint_position, 50.0)
    assert observed.joint_position.shape == (1, HORIZON, 1)


def test_expanded_execution_state_owns_and_repeats_motion_command_history() -> None:
    history = _history(7.0)
    execution_state = _execution_state(history=history)
    factory = _Factory()
    evaluator = _evaluator(
        max_worlds=3,
        factory=factory,
        execution_state=execution_state,
    )

    result = evaluator.evaluate(np.zeros((1, HORIZON, 1), dtype=np.float64))

    assert factory.executor is not None
    expanded = factory.executor.execution_states[0].motion_command_history
    assert expanded is not None
    assert expanded.joint_position.shape == (1, HORIZON, 1)
    np.testing.assert_array_equal(expanded.joint_position, 7.0)
    assert not expanded.joint_position.flags.writeable
    assert not np.shares_memory(expanded.joint_position, history.joint_position)
    assert result.ledger.physical_rollouts == 1
    assert factory.capacities == [1]


def test_evaluate_batches_own_costs_and_do_not_share_ledger_state() -> None:
    adapter = _Adapter()
    factory = _Factory()
    evaluator = _evaluator(max_worlds=3, adapter=adapter, factory=factory)
    candidates = np.zeros((2, HORIZON, 1), dtype=np.float64)

    first = evaluator.evaluate(candidates)
    second = evaluator.evaluate(candidates[:1])
    third = evaluator.evaluate(candidates)
    candidates[:] = 8.0

    np.testing.assert_array_equal(first.costs, 0.0)
    assert not first.costs.flags.writeable
    assert first.ledger.logical_rollouts == 2
    assert first.ledger.physical_rollouts == 2
    assert second.ledger.logical_rollouts == 1
    assert second.ledger.physical_rollouts == 1
    assert third.ledger.physical_rollouts == 2
    assert adapter.expansions == [1]
    assert factory.capacities == [1]
    with pytest.raises(ValueError):
        first.costs[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        first.ledger = second.ledger  # type: ignore[misc]


def _robust_scenarios() -> ScenarioBatch:
    return ScenarioBatch(
        contact_friction_scale=np.array([0.8, 1.2]),
        joint_armature_scale=np.array([0.9, 1.1]),
        joint_frictionloss_scale=np.array([0.7, 1.3]),
        actuation_strength_scale=np.array([0.95, 1.05]),
        root_linear_velocity_offset=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        root_angular_velocity_offset=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        joint_velocity_offset=np.array([[13.0], [14.0]]),
    )


class _ScenarioReducer(_Reducer):
    def prepare(self, reference_motion: MotionCommand, candidate: MotionCommand) -> None:
        del reference_motion
        base = candidate.joint_position[:, 0, 0]
        self.costs = base + np.tile(np.array([1.0, 3.0]), base.size // 2)


class _ScenarioReducerFactory:
    def __call__(self, capacity: int) -> _ScenarioReducer:
        assert capacity in (2, 4)
        return _ScenarioReducer()


def test_robust_evaluation_groups_complete_scenarios_and_records_compute() -> None:
    scenarios = _robust_scenarios()
    configured: list[tuple[int, str]] = []
    evaluator = _evaluator(
        max_worlds=5,
        execution_state=_execution_state(command=False).__class__(
            qpos=np.zeros((1, 1)),
            qvel=np.zeros((1, 7)),
            control_step=4,
            time=0.08,
            motion_command_index=4,
            command=None,
            bfm_state=("one-lane",),
        ),
        reducer_factory=_ScenarioReducerFactory(),  # type: ignore[arg-type]
        scenarios=scenarios,
        scenario_configurer=lambda batch, values: configured.append(
            (batch, values.digest)
        ),
    )
    values = np.arange(3, dtype=np.float64)
    candidates = np.repeat(values[:, None, None], HORIZON, axis=1)

    result = evaluator.evaluate(candidates)

    np.testing.assert_array_equal(result.costs, [2.0, 3.0, 4.0])
    ledger = result.ledger
    assert ledger.candidate_evaluations == 3
    assert ledger.scenario_count == 2
    assert ledger.scenario_digest == scenarios.digest
    assert ledger.logical_rollouts == 6
    assert ledger.physical_rollouts == 8
    assert configured == [(4, scenarios.digest)]
    assert [(wave.candidate_start, wave.candidate_stop) for wave in ledger.waves] == [
        (0, 2),
        (2, 3),
    ]
    executor = evaluator._executors[4]  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        executor.execution_states[0].qvel[:, :3],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] * 2,
    )


def test_robust_worst_risk_and_singleton_use_one_complete_group() -> None:
    scenarios = _robust_scenarios()
    configured: list[int] = []
    execution_state = ExecutionState(
        qpos=np.zeros((1, 1)),
        qvel=np.zeros((1, 7)),
        control_step=0,
        time=0.0,
        motion_command_index=0,
        command=None,
        bfm_state=("one-lane",),
    )
    evaluator = _evaluator(
        max_worlds=5,
        execution_state=execution_state,
        reducer_factory=_ScenarioReducerFactory(),  # type: ignore[arg-type]
        scenarios=scenarios,
        risk_strategy="worst",
        scenario_configurer=lambda batch, _values: configured.append(batch),
    )

    result = evaluator.evaluate(np.zeros((1, HORIZON, 1), dtype=np.float64))

    np.testing.assert_array_equal(result.costs, [3.0])
    assert result.ledger.logical_rollouts == 2
    assert result.ledger.physical_rollouts == 2
    assert configured == [4, 2]


def test_one_nonfinite_scenario_penalizes_only_its_candidate() -> None:
    class NonfiniteScenarioReducer(_ScenarioReducer):
        def finalize(self) -> np.ndarray:
            return np.array([1.0, np.nan, 3.0, 5.0], dtype=np.float64)

    execution_state = ExecutionState(
        qpos=np.zeros((1, 1)),
        qvel=np.zeros((1, 7)),
        control_step=0,
        time=0.0,
        motion_command_index=0,
        command=None,
        bfm_state=("one-lane",),
    )
    evaluator = _evaluator(
        max_worlds=4,
        execution_state=execution_state,
        reducer_factory=lambda _capacity: NonfiniteScenarioReducer(),
        scenarios=_robust_scenarios(),
        penalty=99.0,
    )

    result = evaluator.evaluate(np.zeros((2, HORIZON, 1), dtype=np.float64))

    np.testing.assert_array_equal(result.costs, [99.0, 4.0])
    assert result.ledger.penalized_candidate_evaluations == 1
    assert result.ledger.invalid_scenario_rollouts == 1
    assert result.ledger.waves[0].penalized_candidate_ids == (0,)


def test_ledger_rejects_contradictory_candidate_scenario_accounting() -> None:
    wave = WaveDiagnostics(
        wave_index=0,
        candidate_start=0,
        candidate_evaluations=2,
        scenario_count=2,
        logical_rollouts=4,
        physical_rollouts=4,
        penalized_candidate_ids=(),
        invalid_scenario_rollouts=0,
        failure=None,
        minimum_cost=1.0,
        median_cost=2.0,
        maximum_cost=3.0,
    )

    with pytest.raises(ValueError, match=r"candidates \* scenarios"):
        EvaluationLedger(
            candidate_evaluations=2,
            scenario_count=2,
            scenario_digest="sha256:test",
            logical_rollouts=3,
            physical_rollouts=4,
            waves=(wave,),
            cost_quantiles=(1.0, 1.5, 2.0, 2.5, 3.0),
        )
