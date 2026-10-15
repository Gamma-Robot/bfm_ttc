"""Convergence stopping preserves sampled candidates and actual compute accounting."""

import numpy as np
import pytest

from sampling_optimizers import ConvergenceCriteria
from sampling_optimizers.algs.cem import CEM


def _cem():
    return CEM(4, 2, 1, num_elites=2, sigma_start=1.6, sigma_min=0.075)


@pytest.mark.parametrize(
    "costs,patience,threshold,expected",
    [
        ([10, 10, 10, 10, 0], 3, 0.0, 4),
        ([10, 10, 9, 9, 9, 9, 0], 3, 0.0, 6),
        ([10, 9, 10, 9.5, 10, 0], 3, 0.0, 5),
        ([10, 9, 8.75, 8.5, 0], 2, 0.5, 4),
        ([10, 9.5, 9, 0], 2, 0.5, 3),
    ],
)
def test_convergence_uses_all_time_best_and_consecutive_stalls(
    costs, patience, threshold, expected
):
    optimizer = _cem()
    seen = []

    def evaluate(candidates):
        seen.append(candidates.copy())
        return np.full(len(candidates), costs[len(seen) - 1])

    params, report = optimizer.optimize(
        optimizer.init_params(47), evaluate, ConvergenceCriteria(patience, threshold)
    )
    assert params.iteration == len(report.iterations) == len(seen) == expected
    assert [r.best_observed_cost for r in report.iterations] == costs[:expected]
    assert [r.iteration for r in report.iterations] == list(range(expected))
    assert report.best_observed_cost == min(costs[:expected])


def test_fixed_budget_matches_convergence_prefix_and_rng_state():
    optimizer = _cem()
    candidate_runs = []
    states = []
    for limit in (4, np.int64(4), ConvergenceCriteria(3, 0.0)):
        candidates_seen = []

        def evaluate(candidates):
            candidates_seen.append(candidates.copy())
            return np.ones(len(candidates))

        state, report = optimizer.optimize(optimizer.init_params(47), evaluate, limit)
        assert len(report.iterations) == 4
        candidate_runs.append(candidates_seen)
        states.append(state)
    for actual, state in zip(candidate_runs[1:], states[1:]):
        np.testing.assert_array_equal(actual, candidate_runs[0])
        np.testing.assert_array_equal(state.mean, states[0].mean)
        np.testing.assert_array_equal(state.cov, states[0].cov)
        assert state.rng.bit_generator.state == states[0].rng.bit_generator.state


def test_stopping_history_is_reset_for_each_optimize_call():
    optimizer = _cem()
    for seed, cost in ((47, 1.0), (48, 100.0)):
        _, report = optimizer.optimize(
            optimizer.init_params(seed),
            lambda candidates: np.full(len(candidates), cost),
            ConvergenceCriteria(3, 0.0),
        )
        assert len(report.iterations) == 4
        assert report.best_observed_cost == cost


@pytest.mark.parametrize("patience", [True, np.bool_(True), 0, -1, 1.5, "3", None])
def test_criteria_rejects_invalid_patience(patience):
    with pytest.raises(ValueError, match="patience"):
        ConvergenceCriteria(patience, 0.0)


@pytest.mark.parametrize("threshold", [True, np.bool_(False), -1.0, np.inf, -np.inf, np.nan, "0", None])
def test_criteria_rejects_invalid_improvement(threshold):
    with pytest.raises(ValueError, match="min_cost_improvement"):
        ConvergenceCriteria(3, threshold)


def test_criteria_normalizes_numpy_numeric_scalars():
    criteria = ConvergenceCriteria(np.int64(3), np.float32(0.0))
    assert type(criteria.patience) is int
    assert type(criteria.min_cost_improvement) is float


@pytest.mark.parametrize("limit", [True, np.bool_(True), 1.5, "inf", None])
def test_optimizer_rejects_unparsed_iteration_limits(limit):
    optimizer = _cem()
    with pytest.raises(TypeError, match="iterations"):
        optimizer.optimize(optimizer.init_params(0), lambda _: pytest.fail("evaluated"), limit)
