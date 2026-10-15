import numpy as np
import pytest

from sampling_optimizers.algs.cem import CEM, CEMParams
from sampling_optimizers.registry import get_optimizer


def _make(**kw):
    defaults = dict(
        num_candidates=64,
        num_knots=3,
        dim=2,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
    )
    defaults.update(kw)
    return CEM(**defaults)


def test_registered_and_params_have_cov() -> None:
    assert get_optimizer("cem") is _make().__class__
    params = _make().init_params(seed=0)
    assert isinstance(params, CEMParams)
    np.testing.assert_array_equal(params.cov, np.full((3, 2), 0.5))


def test_rejects_bad_explore_fraction() -> None:
    with pytest.raises(ValueError):
        _make(explore_fraction=1.5)


def test_update_fits_elites(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.ones((3, 2)))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    elites = candidates[np.argsort(costs)[:8]]
    np.testing.assert_allclose(params.mean, elites.mean(axis=0))
    np.testing.assert_allclose(
        params.cov, np.maximum(elites.std(axis=0), 0.05)
    )


def test_cov_never_below_sigma_min(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    params, _ = optimizer.optimize(params, quadratic(np.zeros((3, 2))), iterations=30)
    assert (params.cov >= 0.05 - 1e-12).all()


def test_converges(quadratic) -> None:
    target = np.full((3, 2), 1.0)
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert report.iterations[-1].best_observed_cost < 0.1 * report.iterations[0].best_observed_cost


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
