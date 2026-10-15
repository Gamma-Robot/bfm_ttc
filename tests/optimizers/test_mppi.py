import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.algs.mppi import MPPI
from sampling_optimizers.registry import get_optimizer


def _make(temperature: float = 0.1):
    return MPPI(
        num_candidates=64, num_knots=3, dim=2, noise_level=0.4, temperature=temperature
    )


def test_registered() -> None:
    assert get_optimizer("mppi") is MPPI


def test_update_is_softmax_weighted_average(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.ones((3, 2)))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    w = softmax_weights(costs, temperature=0.1)
    expected = (w[:, None, None] * candidates).sum(axis=0)
    np.testing.assert_allclose(params.mean, expected)


def test_low_temperature_approaches_argmin(quadratic) -> None:
    optimizer = _make(temperature=1e-6)
    params = optimizer.init_params(seed=0)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.ones((3, 2)))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    np.testing.assert_allclose(
        params.mean, candidates[np.argmin(costs)], atol=1e-6
    )


def test_converges(quadratic) -> None:
    target = np.full((3, 2), 1.0)
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert report.iterations[-1].best_observed_cost < 0.1 * report.iterations[0].best_observed_cost


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
