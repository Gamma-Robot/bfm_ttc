import numpy as np

from sampling_optimizers.algs.predictive_sampling import PredictiveSampling
from sampling_optimizers.registry import get_optimizer


def _make():
    return PredictiveSampling(num_candidates=64, num_knots=3, dim=2, noise_level=0.4)


def test_registered() -> None:
    assert get_optimizer("predictive_sampling") is PredictiveSampling


def test_candidate_zero_is_mean() -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0, initial_mean=np.full((3, 2), 0.5))
    candidates, _ = optimizer.sample_candidates(params)
    assert candidates.shape == (64, 3, 2)
    np.testing.assert_array_equal(candidates[0], params.mean)


def test_update_keeps_best(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.ones((3, 2)))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    np.testing.assert_array_equal(params.mean, candidates[np.argmin(costs)])


def test_converges_and_never_regresses(quadratic) -> None:
    target = np.full((3, 2), 1.0)
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=20)
    bests = [r.best_observed_cost for r in report.iterations]
    # Mean is always re-sampled as candidate 0, so best cost is monotone.
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(bests, bests[1:]))
    assert bests[-1] < bests[0]


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
