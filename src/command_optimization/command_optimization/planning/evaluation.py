"""Finite-horizon Candidate Rollout evaluation and dispatch evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bfm_adapters import (
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
    PDCommand,
    RobotStateView,
    BFMAdapter,
)
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.motion_tracking.search_space import QposKnotsSpace
from command_optimization.planning.risk import RiskStrategy, reduce_scenario_costs
from command_optimization.simulation.command_execution import CommandExecutor
from command_optimization.simulation.randomization import ScenarioBatch
from command_optimization.simulation.execution_state import ExecutionState


ExecutorFactory = Callable[[int], CommandExecutor]
ScenarioConfigurer = Callable[[int, ScenarioBatch], None]


class CostReducer(Protocol):
    def prepare(
        self, reference_motion: MotionCommand, candidate: MotionCommand
    ) -> None: ...

    def record(self, step_index: int, state_view: RobotStateView) -> None: ...

    def finalize(self) -> np.ndarray: ...


ReducerFactory = Callable[[int], CostReducer]


@dataclass(frozen=True)
class WaveDiagnostics:
    """Outcome of one physical wave and its stable logical candidate range."""

    wave_index: int
    candidate_start: int
    candidate_evaluations: int
    scenario_count: int
    logical_rollouts: int
    physical_rollouts: int
    penalized_candidate_ids: tuple[int, ...]
    invalid_scenario_rollouts: int
    failure: str | None
    minimum_cost: float
    median_cost: float
    maximum_cost: float

    def __post_init__(self) -> None:
        for field, minimum in (
            ("wave_index", 0),
            ("candidate_start", 0),
            ("candidate_evaluations", 1),
            ("scenario_count", 1),
            ("logical_rollouts", 1),
            ("physical_rollouts", 1),
            ("invalid_scenario_rollouts", 0),
        ):
            value = getattr(self, field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
            object.__setattr__(self, field, int(value))
        if self.logical_rollouts != self.candidate_evaluations * self.scenario_count:
            raise ValueError("logical_rollouts must equal candidates * scenarios")
        if self.physical_rollouts < self.logical_rollouts:
            raise ValueError("physical_rollouts must cover logical_rollouts")
        if self.invalid_scenario_rollouts > self.logical_rollouts:
            raise ValueError("invalid_scenario_rollouts exceeds logical_rollouts")
        ids = tuple(int(value) for value in self.penalized_candidate_ids)
        if len(set(ids)) != len(ids) or any(
            value < self.candidate_start or value >= self.candidate_stop
            for value in ids
        ):
            raise ValueError("penalized_candidate_ids must be unique IDs in this wave")
        object.__setattr__(self, "penalized_candidate_ids", ids)
        if self.failure is not None and not self.failure:
            raise ValueError("failure must be non-empty when present")
        statistics = tuple(
            float(getattr(self, field))
            for field in ("minimum_cost", "median_cost", "maximum_cost")
        )
        if not np.isfinite(statistics).all() or not (
            statistics[0] <= statistics[1] <= statistics[2]
        ):
            raise ValueError("wave cost statistics must be finite and ordered")

    @property
    def candidate_stop(self) -> int:
        return self.candidate_start + self.candidate_evaluations


@dataclass(frozen=True)
class EvaluationLedger:
    """Immutable compute and failure record for one logical evaluator call."""

    candidate_evaluations: int
    scenario_count: int
    scenario_digest: str | None
    logical_rollouts: int
    physical_rollouts: int
    waves: tuple[WaveDiagnostics, ...]
    cost_quantiles: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        for field in (
            "candidate_evaluations",
            "scenario_count",
            "logical_rollouts",
            "physical_rollouts",
        ):
            value = getattr(self, field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        if self.logical_rollouts != self.candidate_evaluations * self.scenario_count:
            raise ValueError("logical_rollouts must equal candidates * scenarios")
        if self.physical_rollouts < self.logical_rollouts:
            raise ValueError("physical_rollouts must cover logical_rollouts")
        if self.scenario_digest is not None and not self.scenario_digest.startswith(
            "sha256:"
        ):
            raise ValueError("scenario_digest must be a sha256 identity when present")
        if not self.waves:
            raise ValueError("an evaluation ledger must contain at least one wave")
        expected_start = 0
        for index, wave in enumerate(self.waves):
            if wave.wave_index != index or wave.candidate_start != expected_start:
                raise ValueError("waves must be indexed and candidate-contiguous")
            if wave.scenario_count != self.scenario_count:
                raise ValueError("all waves must use the ledger scenario_count")
            expected_start = wave.candidate_stop
        if expected_start != self.candidate_evaluations:
            raise ValueError("wave candidate counts disagree with the ledger")
        if sum(wave.logical_rollouts for wave in self.waves) != self.logical_rollouts:
            raise ValueError("wave logical rollout counts disagree with the ledger")
        if sum(wave.physical_rollouts for wave in self.waves) != self.physical_rollouts:
            raise ValueError("wave physical rollout counts disagree with the ledger")
        quantiles = tuple(float(value) for value in self.cost_quantiles)
        if len(quantiles) != 5 or not np.isfinite(quantiles).all() or any(
            left > right for left, right in zip(quantiles, quantiles[1:])
        ):
            raise ValueError("cost_quantiles must contain five finite ordered values")
        object.__setattr__(self, "cost_quantiles", quantiles)

    @property
    def penalized_candidate_evaluations(self) -> int:
        return sum(len(wave.penalized_candidate_ids) for wave in self.waves)

    @property
    def invalid_scenario_rollouts(self) -> int:
        return sum(wave.invalid_scenario_rollouts for wave in self.waves)


@dataclass(frozen=True, eq=False)
class EvaluationBatch:
    """Owned optimizer costs paired with the work that produced them."""

    costs: np.ndarray
    ledger: EvaluationLedger

    def __post_init__(self) -> None:
        costs = np.array(self.costs, dtype=np.float64, order="C", copy=True)
        if costs.ndim != 1 or costs.shape != (
            self.ledger.candidate_evaluations,
        ):
            raise ValueError("costs must match the ledger's candidate count")
        if not np.isfinite(costs).all():
            raise ValueError("costs must be finite")
        costs.flags.writeable = False
        object.__setattr__(self, "costs", costs)


class RolloutEvaluator:
    """Decode and evaluate one logical candidate batch in fixed-size waves.

    ``max_worlds`` is a physical MJWarp lane ceiling. Device-reduced waves
    contain complete scenario groups; the final wave repeats its last group
    as padding and discards those costs. The host reference instead dispatches
    one candidate's scenarios per wave, so finite-measurement validation cannot
    penalize unrelated candidates. A singleton proposal uses exactly ``R``
    lanes. Ledgers count dispatched rollouts, including failed waves.

    Every valid risk-reduced cost must be strictly below ``failure_penalty``.
    An insufficient penalty is a configuration error, not a rollout failure.
    """

    def __init__(
        self,
        *,
        search_space: QposKnotsSpace,
        objective: MotionTrackingObjective,
        reference_motion: MotionCommand,
        execution_state: ExecutionState,
        bfm_adapter: BFMAdapter,
        executor_factory: ExecutorFactory,
        max_worlds: int,
        failure_penalty: float,
        right_context: MotionCommand | None = None,
        reducer_factory: ReducerFactory | None = None,
        scenarios: ScenarioBatch | None = None,
        risk_strategy: RiskStrategy = "mean",
        scenario_configurer: ScenarioConfigurer | None = None,
    ) -> None:
        if int(max_worlds) != max_worlds or max_worlds < 1:
            raise ValueError("max_worlds must be a positive integer")
        penalty = float(failure_penalty)
        if not np.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("failure_penalty must be finite and strictly positive")
        if execution_state.qpos.shape[0] != 1:
            raise ValueError(
                "the planning Execution State must contain exactly one lane"
            )
        if (
            execution_state.command is not None
            and execution_state.command.position_target.shape[0] != 1
        ):
            raise ValueError(
                "the planning Execution State command must contain exactly one lane"
            )
        if reference_motion.joint_position.shape[0] != 1:
            raise ValueError("reference_motion motion_command must contain exactly one lane")
        horizon = len(reference_motion.query_offsets)
        if reference_motion.query_offsets != tuple(range(horizon)):
            raise ValueError("reference_motion motion_command must use a dense zero-based query grid")
        if right_context is not None:
            context_steps = len(right_context.query_offsets)
            if right_context.query_offsets != tuple(range(context_steps)):
                raise ValueError(
                    "right_context must use a dense zero-based query grid"
                )
            if (
                right_context.joint_position.shape[0] != 1
                or right_context.joint_names != reference_motion.joint_names
                or right_context.body_names != reference_motion.body_names
            ):
                raise ValueError(
                    "right_context must be batch-one motion_command with reference_motion names"
                )
        if not callable(executor_factory):
            raise TypeError("executor_factory must be callable")
        if reducer_factory is not None and not callable(reducer_factory):
            raise TypeError("reducer_factory must be callable")

        latent_shape = tuple(search_space.latent_shape)
        if len(latent_shape) != 2 or any(size < 1 for size in latent_shape):
            raise ValueError(
                "search_space.latent_shape must contain two positive sizes"
            )

        self._search_space = search_space
        self._objective = objective
        self._reference_motion = reference_motion
        self._right_context = (
            None if right_context is None else _owned_motion_command(right_context)
        )
        self._execution_state = execution_state
        self._bfm_adapter = bfm_adapter
        if risk_strategy not in ("mean", "worst"):
            raise ValueError(f"unknown risk strategy: {risk_strategy!r}")
        scenario_count = 1 if scenarios is None else scenarios.count
        if int(max_worlds) < scenario_count:
            raise ValueError("max_worlds must fit at least one complete scenario group")
        if scenarios is not None:
            expected_qvel = 6 + scenarios.num_joints
            if execution_state.qvel.shape[1] != expected_qvel:
                raise ValueError(
                    "scenario velocity offsets require Execution State qvel width "
                    f"{expected_qvel}, got {execution_state.qvel.shape[1]}"
                )
        self._max_worlds = int(max_worlds)
        self._scenario_count = scenario_count
        self._candidate_capacity = (
            1 if reducer_factory is None else self._max_worlds // scenario_count
        )
        self._physical_capacity = self._candidate_capacity * scenario_count
        self._scenarios = scenarios
        self._scenario_digest = None if scenarios is None else scenarios.digest
        self._risk_strategy = risk_strategy
        self._scenario_configurer = scenario_configurer
        self._failure_penalty = penalty
        self._latent_shape = latent_shape
        self._horizon = horizon
        self._executor_factory = executor_factory
        if scenario_configurer is not None:
            scenario_configurer(self._physical_capacity, self._scenario_batch())
        self._executors = {
            self._physical_capacity: executor_factory(self._physical_capacity)
        }
        self._reducer_factory = reducer_factory
        self._reducers = (
            {}
            if reducer_factory is None
            else {self._physical_capacity: reducer_factory(self._physical_capacity)}
        )
        self._execution_states: dict[int, ExecutionState] = {}
        self._right_contexts: dict[int, MotionCommand | None] = {}

    def decode_motion_command(self, candidate: np.ndarray) -> MotionCommand:
        """Decode one selected latent without consuming rollout budget."""
        value = np.asarray(candidate)
        if value.dtype != np.float64 or value.shape != self._latent_shape:
            raise ValueError(
                "candidate must be a float64 array with shape "
                f"{self._latent_shape}, got {value.dtype} {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("candidate must be finite")
        return _owned_motion_command(self._search_space.decode(value[None]))

    def evaluate(self, candidates: np.ndarray) -> EvaluationBatch:
        """Return lower-is-better costs and the exact work used to evaluate them."""
        values = self._validate_candidates(candidates)
        count = values.shape[0]
        costs = np.empty(count, dtype=np.float64)
        waves: list[WaveDiagnostics] = []

        for wave_index, start in enumerate(range(0, count, self._candidate_capacity)):
            stop = min(start + self._candidate_capacity, count)
            candidates_in_wave = stop - start
            physical = (
                self._scenario_count if count == 1 else self._physical_capacity
            )
            physical_candidates = physical // self._scenario_count
            padded = self._pad(values[start:stop], physical_candidates)
            expanded = np.repeat(padded, self._scenario_count, axis=0)
            wave_costs, penalized, invalid, failure = self._evaluate_wave(
                expanded, candidates_in_wave, physical
            )
            costs[start:stop] = wave_costs
            waves.append(
                WaveDiagnostics(
                    wave_index=wave_index,
                    candidate_start=start,
                    candidate_evaluations=candidates_in_wave,
                    scenario_count=self._scenario_count,
                    logical_rollouts=candidates_in_wave * self._scenario_count,
                    physical_rollouts=physical,
                    penalized_candidate_ids=tuple(start + index for index in penalized),
                    invalid_scenario_rollouts=invalid,
                    failure=failure,
                    minimum_cost=float(np.min(wave_costs)),
                    median_cost=float(np.median(wave_costs)),
                    maximum_cost=float(np.max(wave_costs)),
                )
            )

        quantile_values = np.quantile(costs, (0.0, 0.25, 0.5, 0.75, 1.0))
        quantiles = (
            float(quantile_values[0]),
            float(quantile_values[1]),
            float(quantile_values[2]),
            float(quantile_values[3]),
            float(quantile_values[4]),
        )
        record = EvaluationLedger(
            candidate_evaluations=count,
            scenario_count=self._scenario_count,
            scenario_digest=self._scenario_digest,
            logical_rollouts=count * self._scenario_count,
            physical_rollouts=sum(wave.physical_rollouts for wave in waves),
            waves=tuple(waves),
            cost_quantiles=quantiles,
        )
        return EvaluationBatch(costs=costs, ledger=record)

    def _validate_candidates(self, candidates: np.ndarray) -> np.ndarray:
        if not isinstance(candidates, np.ndarray) or candidates.dtype != np.float64:
            raise ValueError("candidates must be a float64 numpy array")
        expected_tail = self._latent_shape
        if candidates.ndim != 3 or candidates.shape[0] < 1:
            raise ValueError(
                f"candidates must have shape (N, {expected_tail[0]}, "
                f"{expected_tail[1]}) with N >= 1, got {candidates.shape}"
            )
        if candidates.shape[1:] != expected_tail:
            raise ValueError(
                f"candidate trailing shape must be {expected_tail}, "
                f"got {candidates.shape[1:]}"
            )
        if not np.isfinite(candidates).all():
            raise ValueError("candidates must be finite")
        return candidates

    def _pad(self, candidates: np.ndarray, physical: int) -> np.ndarray:
        missing = physical - candidates.shape[0]
        if missing == 0:
            return candidates
        padding = np.repeat(candidates[-1:], missing, axis=0)
        return np.concatenate((candidates, padding), axis=0)

    def _evaluate_wave(
        self, candidates: np.ndarray, logical_candidates: int, physical: int
    ) -> tuple[np.ndarray, tuple[int, ...], int, str | None]:
        # Construction failures are not rollout failures: no physical work was
        # dispatched, so returning a penalty would fabricate ledger evidence.
        motion_command = self._search_space.decode(candidates)
        executor = self._executor(physical)
        execution_state = self._expanded_execution_state(physical)
        right_context = self._expanded_right_context(physical)
        reducer = None
        requires = None
        if self._reducer_factory is not None:
            reducer = self._reducer(physical)
            reducer.prepare(self._reference_motion, motion_command)
        else:
            requires = self._objective.requires

        try:
            if reducer is None:
                assert requires is not None
                measurements = executor.rollout(
                    execution_state,
                    motion_command,
                    num_steps=self._horizon,
                    right_context=right_context,
                    requires=requires,
                )
                steps = measurements.num_steps
                result = self._objective.evaluate(
                    measurements,
                    _motion_command_prefix(self._reference_motion, steps),
                    _motion_command_prefix(motion_command, steps),
                )
                physical_costs = np.asarray(result.total, dtype=np.float64)
            else:
                physical_costs = np.asarray(
                    executor.rollout_reduced(
                        execution_state,
                        motion_command,
                        reducer,
                        num_steps=self._horizon,
                        right_context=right_context,
                    ),
                    dtype=np.float64,
                )
            if physical_costs.shape != (physical,):
                raise ValueError(
                    "objective returned shape "
                    f"{physical_costs.shape}, expected {(physical,)}"
                )
        except Exception as exc:
            costs = np.full(logical_candidates, self._failure_penalty, dtype=np.float64)
            invalid = logical_candidates * self._scenario_count
            return (
                costs,
                tuple(range(logical_candidates)),
                invalid,
                f"{type(exc).__name__}: {exc}",
            )

        logical_rollouts = logical_candidates * self._scenario_count
        scenario_costs = physical_costs[:logical_rollouts].reshape(
            logical_candidates, self._scenario_count
        )
        finite = np.isfinite(scenario_costs)
        invalid = int(np.count_nonzero(~finite))
        penalized = tuple(
            int(index) for index in np.flatnonzero(~np.all(finite, axis=1))
        )
        safe = np.where(finite, scenario_costs, 0.0)
        logical_costs = reduce_scenario_costs(safe, strategy=self._risk_strategy)
        reduced_nonfinite = ~np.isfinite(logical_costs)
        valid = np.all(finite, axis=1) & ~reduced_nonfinite
        if np.any(logical_costs[valid] >= self._failure_penalty):
            raise ValueError(
                "failure_penalty must exceed every valid risk-reduced cost: "
                f"penalty={self._failure_penalty}, "
                f"maximum_valid_cost={float(np.max(logical_costs[valid]))}"
            )
        if reduced_nonfinite.any():
            penalized_ids = set(penalized)
            penalized_ids.update(
                int(index) for index in np.flatnonzero(reduced_nonfinite)
            )
            penalized = tuple(sorted(penalized_ids))
        if penalized:
            logical_costs[np.asarray(penalized)] = self._failure_penalty
        if invalid:
            failure = "non-finite objective cost"
        elif reduced_nonfinite.any():
            failure = "non-finite risk-reduced cost"
        else:
            failure = None
        return logical_costs, penalized, invalid, failure

    def _executor(self, batch: int) -> CommandExecutor:
        if batch not in self._executors:
            if self._scenario_configurer is not None:
                self._scenario_configurer(batch, self._scenario_batch())
            self._executors[batch] = self._executor_factory(batch)
        return self._executors[batch]

    def _reducer(self, batch: int) -> CostReducer:
        if batch not in self._reducers:
            assert self._reducer_factory is not None
            self._reducers[batch] = self._reducer_factory(batch)
        return self._reducers[batch]

    def _expanded_execution_state(self, batch: int) -> ExecutionState:
        cached = self._execution_states.get(batch)
        if cached is not None:
            return cached
        command = self._execution_state.command
        qvel = np.repeat(self._execution_state.qvel, batch, axis=0)
        scenarios = self._scenarios
        if scenarios is not None:
            if batch % scenarios.count:
                raise ValueError(
                    "Execution State batch must contain complete scenario groups"
                )
            repeats = batch // scenarios.count
            qvel[:, :3] += np.tile(scenarios.root_linear_velocity_offset, (repeats, 1))
            qvel[:, 3:6] += np.tile(
                scenarios.root_angular_velocity_offset,
                (repeats, 1),
            )
            qvel[:, 6:] += np.tile(scenarios.joint_velocity_offset, (repeats, 1))
        expanded = ExecutionState(
            qpos=np.repeat(self._execution_state.qpos, batch, axis=0),
            qvel=qvel,
            qacc_warmstart=(
                None
                if self._execution_state.qacc_warmstart is None
                else np.repeat(self._execution_state.qacc_warmstart, batch, axis=0)
            ),
            control_step=self._execution_state.control_step,
            time=self._execution_state.time,
            motion_command_index=self._execution_state.motion_command_index,
            command=None if command is None else _expand_command(command, batch),
            bfm_state=self._bfm_adapter.expand_state(
                self._execution_state.bfm_state, batch
            ),
            motion_command_history=(
                None
                if self._execution_state.motion_command_history is None
                else _expand_motion_command(
                    self._execution_state.motion_command_history, batch
                )
            ),
        )
        self._execution_states[batch] = expanded
        return expanded

    def _scenario_batch(self) -> ScenarioBatch:
        if self._scenarios is not None:
            return self._scenarios
        joints = max(1, self._execution_state.qvel.shape[1] - 6)
        return ScenarioBatch.nominal(count=1, num_joints=joints)

    def _expanded_right_context(self, batch: int) -> MotionCommand | None:
        if batch not in self._right_contexts:
            self._right_contexts[batch] = (
                None
                if self._right_context is None
                else _expand_motion_command(self._right_context, batch)
            )
        return self._right_contexts[batch]


def _motion_command_prefix(motion_command: MotionCommand, steps: int) -> MotionCommand:
    if steps < 1 or steps > len(motion_command.query_offsets):
        raise ValueError("executed Control Steps exceed the motion_command horizon")
    if steps == len(motion_command.query_offsets):
        return motion_command
    return MotionCommand(
        query_offsets=tuple(range(steps)),
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **{name: getattr(motion_command, name)[:, :steps] for name in _COMMAND_ARRAYS},
    )


def _owned_motion_command(motion_command: MotionCommand) -> MotionCommand:
    arrays: dict[str, np.ndarray] = {}
    for name in _COMMAND_ARRAYS:
        value = np.array(
            getattr(motion_command, name),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        value.flags.writeable = False
        arrays[name] = value
    return MotionCommand(
        query_offsets=motion_command.query_offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **arrays,
    )


def _expand_motion_command(motion_command: MotionCommand, batch: int) -> MotionCommand:
    return MotionCommand(
        query_offsets=motion_command.query_offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **{
            name: np.repeat(getattr(motion_command, name), batch, axis=0)
            for name in _COMMAND_ARRAYS
        },
    )


def _expand_command(command: PDCommand, batch: int) -> PDCommand:
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "position_target",
        "velocity_target",
        "feedforward_torque",
        "proportional_gain",
        "derivative_gain",
    ):
        value = getattr(command, name)
        if value.shape[0] != 1:
            raise ValueError(
                "planning Execution State command must contain exactly one lane"
            )
        arrays[name] = np.repeat(value, batch, axis=0)
    return PDCommand(joint_names=command.joint_names, **arrays)
