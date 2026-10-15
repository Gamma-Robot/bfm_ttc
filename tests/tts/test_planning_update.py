from __future__ import annotations

import numpy as np
import pytest

from sampling_optimizers import ConvergenceCriteria, OptimizerParams, SamplingOptimizer
from command_optimization.planning.update import run_planning_update

from test_planning_evaluation import HORIZON, _Factory, _evaluator


class _CountingOptimizer(SamplingOptimizer):
    def __init__(self, proposal: float) -> None:
        super().__init__(num_candidates=3, num_knots=HORIZON, dim=1)
        self._proposal = np.full((HORIZON, 1), proposal, dtype=np.float64)
        self.init_calls = 0
        self.optimize_calls = 0
        self.proposal_calls = 0
        self.received_mean: np.ndarray | None = None

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> OptimizerParams:
        self.init_calls += 1
        self.received_mean = None if initial_mean is None else initial_mean.copy()
        return super().init_params(seed, initial_mean)

    def optimize(self, params, evaluate, iterations):
        self.optimize_calls += 1
        return super().optimize(params, evaluate, iterations)

    def sample_candidates(
        self, params: OptimizerParams
    ) -> tuple[np.ndarray, OptimizerParams]:
        values = np.asarray([-1.0, 0.5, 1.0], dtype=np.float64)
        candidates = np.repeat(values[:, None, None], HORIZON, axis=1)
        return candidates, params

    def update_params(
        self, params: OptimizerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> OptimizerParams:
        del candidates, costs
        params.mean = self._proposal.copy()
        return params

    def proposal(self, params: OptimizerParams) -> np.ndarray:
        del params
        self.proposal_calls += 1
        return self._proposal.copy()


class _ObservedProposalOptimizer(_CountingOptimizer):
    def sample_candidates(
        self,
        params: OptimizerParams,
    ) -> tuple[np.ndarray, OptimizerParams]:
        candidates, params = super().sample_candidates(params)
        candidates[0] = 0.25
        candidates[1] = 0.0
        return candidates, params


def test_update_initializes_and_optimizes_once_then_evaluates_proposal() -> None:
    optimizer = _CountingOptimizer(proposal=0.0)
    evaluator = _evaluator(max_worlds=2)
    initial_mean = np.full((HORIZON, 1), 0.25, dtype=np.float64)

    result = run_planning_update(
        optimizer,
        evaluator,
        seed=17,
        iterations=1,
        initial_mean=initial_mean,
        evaluate_proposal=True,
    )

    assert optimizer.init_calls == 1
    assert optimizer.optimize_calls == 1
    assert optimizer.proposal_calls == 1
    np.testing.assert_array_equal(optimizer.received_mean, initial_mean)
    np.testing.assert_array_equal(result.selected_latent, 0.0)
    np.testing.assert_array_equal(result.selected_command.joint_position, 0.0)
    assert result.selected_cost == 0.0
    assert result.diagnostics.selected_source == "proposal"
    assert result.diagnostics.best_observed_cost == 0.5
    assert result.diagnostics.proposal_cost == 0.0
    assert result.diagnostics.logical_rollouts == 4
    assert result.diagnostics.physical_rollouts == 4
    assert len(result.diagnostics.evaluations) == 2
    assert result.diagnostics.evaluations[-1].physical_rollouts == 1
    assert not result.selected_latent.flags.writeable
    assert not result.diagnostics.best_observed_candidate.flags.writeable
    assert not result.diagnostics.iterations[0].best_observed_candidate.flags.writeable


def test_update_does_not_charge_or_select_unevaluated_proposal() -> None:
    optimizer = _CountingOptimizer(proposal=0.0)
    evaluator = _evaluator(max_worlds=2)

    result = run_planning_update(
        optimizer,
        evaluator,
        seed=3,
        iterations=1,
        evaluate_proposal=False,
    )

    np.testing.assert_array_equal(result.selected_latent, 0.5)
    assert result.selected_cost == 0.5
    assert result.diagnostics.selected_source == "best_observed"
    assert result.diagnostics.proposal_cost is None
    assert result.diagnostics.logical_rollouts == 3
    assert result.diagnostics.physical_rollouts == 3
    assert len(result.diagnostics.evaluations) == 1


def test_low_failure_penalty_cannot_make_a_failed_candidate_win() -> None:
    evaluator = _evaluator(factory=_Factory(fail_at=-1.0), penalty=0.1)
    with pytest.raises(ValueError, match="failure_penalty must exceed"):
        run_planning_update(
            _CountingOptimizer(proposal=-1.0),
            evaluator,
            seed=3,
            iterations=1,
            evaluate_proposal=False,
        )


def test_valid_candidate_beats_failed_sample_and_proposal() -> None:
    result = run_planning_update(
        _CountingOptimizer(proposal=-1.0),
        _evaluator(factory=_Factory(fail_at=-1.0)),
        seed=3,
        iterations=1,
    )
    np.testing.assert_array_equal(result.selected_latent, 0.5)
    assert result.selected_cost == 0.5
    assert result.diagnostics.penalized_candidate_evaluations == 1


def test_update_reuses_an_identical_evaluated_proposal() -> None:
    result = run_planning_update(
        _CountingOptimizer(proposal=0.5),
        _evaluator(max_worlds=2),
        seed=3,
        iterations=1,
        evaluate_proposal=True,
    )

    np.testing.assert_array_equal(result.selected_latent, 0.5)
    assert result.selected_cost == result.diagnostics.proposal_cost == 0.5
    assert result.diagnostics.selected_source == "best_observed"
    assert result.diagnostics.logical_rollouts == 3
    assert result.diagnostics.physical_rollouts == 3
    assert len(result.diagnostics.evaluations) == 1


def test_evaluated_proposal_must_beat_best_observed_to_be_selected() -> None:
    optimizer = _CountingOptimizer(proposal=2.0)
    result = run_planning_update(
        optimizer,
        _evaluator(max_worlds=2),
        seed=5,
        iterations=1,
        evaluate_proposal=True,
    )

    np.testing.assert_array_equal(result.selected_latent, 0.5)
    assert result.selected_cost == 0.5
    assert result.diagnostics.proposal_cost == 8.0
    assert result.diagnostics.selected_source == "best_observed"


def test_update_reuses_a_proposal_observed_but_not_selected_as_best() -> None:
    result = run_planning_update(
        _ObservedProposalOptimizer(proposal=0.25),
        _evaluator(max_worlds=2),
        seed=5,
        iterations=1,
        evaluate_proposal=True,
    )

    assert result.diagnostics.proposal_cost == 0.125
    assert result.diagnostics.best_observed_cost == 0.0
    assert result.diagnostics.selected_source == "best_observed"
    assert result.diagnostics.logical_rollouts == 3
    assert result.diagnostics.physical_rollouts == 3
    assert len(result.diagnostics.evaluations) == 1


def test_convergence_records_actual_iterations_and_rollouts() -> None:
    optimizer = _CountingOptimizer(proposal=0.0)
    result = run_planning_update(
        optimizer, _evaluator(max_worlds=2), seed=17,
        iterations=ConvergenceCriteria(patience=3, min_cost_improvement=0.0),
        evaluate_proposal=False,
    )
    assert optimizer.optimize_calls == 1
    assert len(result.diagnostics.iterations) == 4
    assert len(result.diagnostics.evaluations) == 4
    assert result.diagnostics.candidate_evaluations == 12
    assert result.diagnostics.logical_rollouts == 12
    assert result.diagnostics.physical_rollouts == 12
    assert result.diagnostics.best_observed_cost == 0.5
    assert result.diagnostics.proposal_cost is None
