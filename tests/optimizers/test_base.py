import numpy as np
import pytest

from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import available, get_optimizer, register


class _ArgminOptimizer(SamplingOptimizer):
    """Minimal concrete optimizer: Gaussian noise, keep the best candidate."""

    def sample_candidates(self, params):
        noise = params.rng.standard_normal((self.num_candidates, self.num_knots, self.dim))
        return params.mean + noise, params

    def update_params(self, params, candidates, costs):
        params.mean = candidates[int(np.argmin(costs))].copy()
        return params


class _AverageOptimizer(_ArgminOptimizer):
    """Update to the candidate average: proposal is never an evaluated sample."""

    def update_params(self, params, candidates, costs):
        params.mean = candidates.mean(axis=0)
        return params


def _quadratic(target):
    def evaluate(candidates):
        return ((candidates - target) ** 2).sum(axis=(1, 2))

    return evaluate


class _CountingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.batch_sizes = []

    def __call__(self, candidates):
        self.calls += 1
        self.batch_sizes.append(len(candidates))
        return self.inner(candidates)


def _make(cls=_ArgminOptimizer, **kw):
    defaults = dict(num_candidates=8, num_knots=3, dim=2)
    defaults.update(kw)
    return cls(**defaults)


def test_init_params_shape_and_determinism() -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    assert params.mean.shape == (3, 2)
    assert params.mean.dtype == np.float64
    a = optimizer.init_params(seed=1).rng.standard_normal(4)
    b = optimizer.init_params(seed=1).rng.standard_normal(4)
    np.testing.assert_array_equal(a, b)


def test_constructor_rejects_nonpositive_sizes() -> None:
    for kw in (dict(num_candidates=0), dict(num_knots=0), dict(dim=0)):
        with pytest.raises(ValueError):
            _make(**kw)


def test_init_params_rejects_bad_mean() -> None:
    optimizer = _make()
    with pytest.raises(ValueError):
        optimizer.init_params(seed=0, initial_mean=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        optimizer.init_params(seed=0, initial_mean=np.full((3, 2), np.nan))


def test_optimize_runs_iterations_and_reports() -> None:
    target = np.ones((3, 2))
    optimizer = _make(num_candidates=64)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, _quadratic(target), iterations=5)
    assert len(report.iterations) == 5
    assert params.iteration == 5
    assert (
        report.iterations[-1].best_observed_cost
        <= report.iterations[0].best_observed_cost
    )
    assert report.best_observed_cost == min(
        r.best_observed_cost for r in report.iterations
    )
    assert report.iterations[0].best_observed_candidate.shape == (3, 2)


def test_optimize_budget_accounting() -> None:
    optimizer = _make(num_candidates=16)
    params = optimizer.init_params(seed=0)
    evaluator = _CountingEvaluator(_quadratic(np.zeros((3, 2))))
    optimizer.optimize(params, evaluator, iterations=7)
    assert evaluator.calls == 7
    assert evaluator.batch_sizes == [16] * 7


def test_proposal_distinct_from_best_observed() -> None:
    optimizer = _make(_AverageOptimizer, num_candidates=16)
    params = optimizer.init_params(seed=0)
    evaluator = _CountingEvaluator(_quadratic(np.ones((3, 2))))
    params, report = optimizer.optimize(params, evaluator, iterations=3)
    proposal = optimizer.proposal(params)
    np.testing.assert_array_equal(proposal, params.mean)
    assert not np.shares_memory(proposal, params.mean)
    # The averaged mean is not any evaluated candidate, and proposal() must
    # not have spent extra evaluations.
    assert not np.array_equal(proposal, report.best_observed_candidate)
    assert evaluator.calls == 3


def test_report_does_not_alias_mutable_state() -> None:
    optimizer = _make(num_candidates=16)
    params = optimizer.init_params(seed=0)
    inner = _quadratic(np.ones((3, 2)))
    captured = []

    def evaluate(candidates):
        captured.append(candidates)
        return inner(candidates)

    params, report = optimizer.optimize(params, evaluate, iterations=2)
    record = report.iterations[-1]
    assert not np.shares_memory(record.best_observed_candidate, params.mean)
    # The record must not be a view into the candidates buffer the evaluator saw.
    assert not np.shares_memory(record.best_observed_candidate, captured[-1])
    candidate_before_update = record.best_observed_candidate.copy()
    captured[-1][:] = 777.0
    params.mean[:] = -999.0
    np.testing.assert_array_equal(
        record.best_observed_candidate, candidate_before_update
    )


def test_report_captures_observation_before_update() -> None:
    class _MutatesInputs(_ArgminOptimizer):
        def update_params(self, params, candidates, costs):
            params.mean = candidates[int(np.argmin(costs))].copy()
            candidates[:] = 777.0
            costs[:] = 999.0
            return params

    optimizer = _make(_MutatesInputs, num_candidates=16)
    params = optimizer.init_params(seed=0)
    observed = {}

    def evaluate(candidates):
        costs = _quadratic(np.ones((3, 2)))(candidates)
        observed["candidates"] = candidates.copy()
        observed["costs"] = costs.copy()
        return costs

    _, report = optimizer.optimize(params, evaluate, iterations=1)
    best = int(np.argmin(observed["costs"]))
    assert report.best_observed_cost == observed["costs"][best]
    np.testing.assert_array_equal(
        report.best_observed_candidate, observed["candidates"][best]
    )


def test_optimize_rejects_float32_updated_mean() -> None:
    class _Bad(_ArgminOptimizer):
        def update_params(self, params, candidates, costs):
            params.mean = candidates[0].astype(np.float32)
            return params

    optimizer = _make(_Bad)
    with pytest.raises(ValueError, match="float64"):
        optimizer.optimize(
            optimizer.init_params(seed=0), _quadratic(np.zeros((3, 2))), 1
        )


def test_optimize_rejects_zero_iterations() -> None:
    optimizer = _make()
    with pytest.raises(ValueError):
        optimizer.optimize(optimizer.init_params(seed=0), _quadratic(np.zeros((3, 2))), 0)


def test_report_best_observed_candidate_matches_cost() -> None:
    evaluate = _quadratic(np.ones((3, 2)))
    optimizer = _make(num_candidates=32)
    params = optimizer.init_params(seed=3)
    params, report = optimizer.optimize(params, evaluate, iterations=4)
    recomputed = evaluate(report.best_observed_candidate[None])[0]
    assert recomputed == report.best_observed_cost
    best_record = min(report.iterations, key=lambda r: r.best_observed_cost)
    assert not np.shares_memory(
        report.best_observed_candidate, best_record.best_observed_candidate
    )


def test_optimize_rejects_float32_candidates() -> None:
    class _Bad(_ArgminOptimizer):
        def sample_candidates(self, params):
            out = np.zeros(
                (self.num_candidates, self.num_knots, self.dim), dtype=np.float32
            )
            return out, params

    optimizer = _make(_Bad)
    with pytest.raises(ValueError, match="float64"):
        optimizer.optimize(optimizer.init_params(seed=0), _quadratic(np.zeros((3, 2))), 1)


def test_optimize_rejects_bad_candidate_shape() -> None:
    class _Bad(_ArgminOptimizer):
        def sample_candidates(self, params):
            return np.zeros((self.num_candidates, self.num_knots + 1, self.dim)), params

    optimizer = _make(_Bad)
    with pytest.raises(ValueError, match="shape"):
        optimizer.optimize(optimizer.init_params(seed=0), _quadratic(np.zeros((3, 2))), 1)


def test_optimize_rejects_nonfinite_candidates() -> None:
    class _Bad(_ArgminOptimizer):
        def sample_candidates(self, params):
            out = np.zeros((self.num_candidates, self.num_knots, self.dim))
            out[0, 0, 0] = np.inf
            return out, params

    optimizer = _make(_Bad)
    with pytest.raises(ValueError, match="finite"):
        optimizer.optimize(optimizer.init_params(seed=0), _quadratic(np.zeros((3, 2))), 1)


def test_optimize_rejects_bad_cost_shape() -> None:
    optimizer = _make()
    evaluate = lambda c: np.zeros((len(c), 1))  # noqa: E731
    with pytest.raises(ValueError, match="shape"):
        optimizer.optimize(optimizer.init_params(seed=0), evaluate, 1)


def test_optimize_rejects_nonfinite_costs() -> None:
    optimizer = _make()

    def evaluate(candidates):
        costs = np.zeros(len(candidates))
        costs[0] = np.nan
        return costs

    with pytest.raises(ValueError, match="finite"):
        optimizer.optimize(optimizer.init_params(seed=0), evaluate, 1)


def test_optimize_seed_determinism() -> None:
    finals = []
    for _ in range(2):
        optimizer = _make(num_candidates=32)
        params = optimizer.init_params(seed=9)
        params, _ = optimizer.optimize(params, _quadratic(np.ones((3, 2))), iterations=4)
        finals.append(params.mean.copy())
    np.testing.assert_array_equal(finals[0], finals[1])


def test_registry_round_trip() -> None:
    @register("argmin-test")
    class _P(_ArgminOptimizer):
        pass

    assert get_optimizer("argmin-test") is _P
    assert "argmin-test" in available()
    with pytest.raises(KeyError, match="unknown optimizer"):
        get_optimizer("nope")
    with pytest.raises(ValueError, match="already registered"):
        register("argmin-test")(_P)


def test_registry_rejects_non_optimizer_class() -> None:
    class _NotAOptimizer:
        pass

    with pytest.raises(TypeError, match="SamplingOptimizer"):
        register("not-a-optimizer")(_NotAOptimizer)


def test_softmax_weights_normalized_and_cost_ordered() -> None:
    from sampling_optimizers._weights import softmax_weights

    costs = np.array([3.0, 1.0, 2.0])
    w = softmax_weights(costs, temperature=0.5)
    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[2] > w[0]


def test_softmax_weights_stable_at_extremes_and_validates() -> None:
    from sampling_optimizers._weights import softmax_weights

    w = softmax_weights(np.array([1e12, 0.0, 5e11]), temperature=1e-6)
    assert np.isfinite(w).all()
    assert np.isclose(w.sum(), 1.0)
    for bad_temp in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            softmax_weights(np.array([1.0, 2.0]), temperature=bad_temp)
    with pytest.raises(ValueError):
        softmax_weights(np.array([1.0, np.nan]), temperature=0.5)


def test_softmax_weights_rejects_bad_shape() -> None:
    from sampling_optimizers._weights import softmax_weights

    with pytest.raises(ValueError):
        softmax_weights(np.zeros((2, 3)), temperature=0.5)
    with pytest.raises(ValueError):
        softmax_weights(np.array([]), temperature=0.5)
