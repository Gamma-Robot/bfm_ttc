import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.algs.mppi_cma import MppiCma, MppiCmaParams
from sampling_optimizers.registry import get_optimizer


def _make(
    temperature: float = 0.1,
    initial_noise_level: float = 0.5,
    minimum_noise_level: float | None = None,
    covariance_adaptation_rate: float = 0.1,
):
    return MppiCma(
        num_candidates=64,
        num_knots=3,
        dim=2,
        initial_noise_level=initial_noise_level,
        temperature=temperature,
        minimum_noise_level=minimum_noise_level,
        covariance_adaptation_rate=covariance_adaptation_rate,
    )


def test_registered_and_init_has_covariance() -> None:
    assert get_optimizer("mppi_cma") is MppiCma
    params = _make().init_params(seed=0)
    assert isinstance(params, MppiCmaParams)
    assert params.covariance.shape == (3, 2, 2)
    expected = np.broadcast_to(np.eye(2) * 0.25, (3, 2, 2))
    np.testing.assert_allclose(params.covariance, expected)


def test_sample_uses_covariance() -> None:
    optimizer = _make(initial_noise_level=1.0)
    params = optimizer.init_params(seed=0)
    # Stretch the second knot's covariance along dim 0.
    params.covariance[1] = np.array([[4.0, 0.0], [0.0, 0.25]])
    candidates, _ = optimizer.sample_candidates(params)
    assert candidates.shape == (64, 3, 2)
    # Knot 1 should spread more along dim 0 than dim 1.
    assert candidates[:, 1, 0].std() > 1.5 * candidates[:, 1, 1].std()


def test_update_matches_weighted_mean_and_covariance() -> None:
    optimizer = MppiCma(
        num_candidates=3,
        num_knots=1,
        dim=2,
        initial_noise_level=1.0,
        temperature=1.0,
        minimum_noise_level=0.0,
        covariance_adaptation_rate=0.25,
    )
    params = optimizer.init_params(seed=0)
    params.mean[:] = np.array([[0.5, -0.5]])
    params.covariance[0] = np.array([[2.0, 0.25], [0.25, 1.0]])
    candidates = np.array([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
    costs = np.array([0.0, 1.0, 2.0])
    old_mean = params.mean.copy()
    old_cov = params.covariance.copy()
    weights = softmax_weights(costs, temperature=1.0)
    delta = candidates - old_mean
    sample_cov = np.einsum(
        "i,ijkl->jkl",
        weights,
        np.einsum("ijk,ijl->ijkl", delta, delta),
    )
    expected_cov = 0.75 * old_cov + 0.25 * sample_cov
    expected_mean = (weights[:, None, None] * candidates).sum(axis=0)

    params = optimizer.update_params(params, candidates, costs)

    np.testing.assert_allclose(params.mean, expected_mean)
    np.testing.assert_allclose(params.covariance, expected_cov)
    np.testing.assert_allclose(
        params.covariance, params.covariance.swapaxes(-1, -2)
    )
    assert np.linalg.eigvalsh(params.covariance).min() >= -1e-12


def test_minimum_noise_floor() -> None:
    optimizer = _make(
        initial_noise_level=1.0,
        minimum_noise_level=0.5,
        covariance_adaptation_rate=1.0,
    )
    params = optimizer.init_params(seed=0)
    # Force a near-collapse update: all candidates identical.
    candidates = np.broadcast_to(params.mean, (64, 3, 2)).copy()
    costs = np.arange(64, dtype=np.float64)
    params = optimizer.update_params(params, candidates, costs)
    eigvals = np.linalg.eigvalsh(params.covariance)
    assert np.min(eigvals) >= 0.5**2 - 1e-12


def test_converges(quadratic) -> None:
    target = np.full((3, 2), 1.0)
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert (
        report.iterations[-1].best_observed_cost
        < 0.1 * report.iterations[0].best_observed_cost
    )


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
