import numpy as np
import pytest

from sampling_optimizers.algs.spider import Spider
from sampling_optimizers.registry import get_optimizer


def _make(**kw):
    defaults = dict(
        num_candidates=10,
        num_knots=4,
        dim=2,
        noise_scale=np.array([0.2, 0.4]),
        first_knot_scale=0.5,
        last_knot_scale=1.0,
        final_noise_scale=0.1,
        num_iterations=4,
        elite_fraction=0.3,
        temperature=0.2,
        exploit_fraction=0.2,
        exploit_noise_scale=0.05,
    )
    defaults.update(kw)
    return Spider(**defaults)


def _reference_sample(
    mean,
    rng,
    *,
    num_candidates,
    noise_scale,
    first_knot_scale,
    last_knot_scale,
    final_noise_scale,
    num_iterations,
    iteration,
    exploit_fraction,
    exploit_noise_scale,
):
    num_knots, dim = mean.shape
    dimension_scale = np.broadcast_to(
        np.asarray(noise_scale, dtype=np.float64), (dim,)
    )
    knot_scale = np.geomspace(
        first_knot_scale, last_knot_scale, num=num_knots, dtype=np.float64
    )
    beta = (
        final_noise_scale ** (1.0 / num_iterations)
        if num_iterations > 1
        else 1.0
    )
    sigma = knot_scale[:, None] * dimension_scale[None, :] * beta**iteration
    noise = rng.standard_normal((num_candidates, num_knots, dim)) * sigma[None]
    noise[0] = 0.0
    num_exploit = int(num_candidates * exploit_fraction)
    if num_exploit > 0:
        noise[-num_exploit:] *= exploit_noise_scale
    return mean[None] + noise


def _reference_update(candidates, costs, elite_fraction, temperature):
    elite_count = max(1, int(len(candidates) * elite_fraction))
    elite_indices = np.argsort(costs, kind="stable")[:elite_count]
    elite_costs = costs[elite_indices].astype(np.float64)
    spread = float(elite_costs.std(ddof=1)) if elite_count > 1 else 0.0
    logits = -(elite_costs - elite_costs.mean()) / (
        temperature * (spread + 1e-2)
    )
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return (candidates[elite_indices] * weights[:, None, None]).sum(axis=0)


def test_registered() -> None:
    assert get_optimizer("spider") is Spider


def test_sample_matches_independent_reference() -> None:
    optimizer = _make()
    initial = np.full((4, 2), 0.1)
    params = optimizer.init_params(seed=42, initial_mean=initial)
    params.iteration = 2
    candidates, _ = optimizer.sample_candidates(params)

    expected = _reference_sample(
        initial,
        np.random.default_rng(42),
        num_candidates=10,
        noise_scale=[0.2, 0.4],
        first_knot_scale=0.5,
        last_knot_scale=1.0,
        final_noise_scale=0.1,
        num_iterations=4,
        iteration=2,
        exploit_fraction=0.2,
        exploit_noise_scale=0.05,
    )
    np.testing.assert_array_equal(candidates, expected)
    np.testing.assert_array_equal(candidates[0], initial)


def test_sigma_shrinks_over_iterations_and_grows_along_horizon() -> None:
    optimizer = _make()
    sigma_0 = optimizer._sigma(0)
    sigma_3 = optimizer._sigma(3)
    assert sigma_0.shape == (4, 2)
    assert (sigma_3 < sigma_0).all()
    assert (np.diff(sigma_0, axis=0) > 0.0).all()


def test_zero_exploit_fraction_does_not_scale_every_candidate() -> None:
    optimizer = _make(
        num_candidates=4,
        exploit_fraction=0.0,
        exploit_noise_scale=0.0,
    )
    params = optimizer.init_params(seed=0)
    candidates, _ = optimizer.sample_candidates(params)
    assert np.any(candidates[1:] != params.mean)


def test_update_matches_independent_reference(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=7)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.full((4, 2), 0.5))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    expected = _reference_update(
        candidates, costs, elite_fraction=0.3, temperature=0.2
    )
    np.testing.assert_array_equal(params.mean, expected)


def test_single_elite_is_finite_and_selects_best_candidate() -> None:
    optimizer = _make(num_candidates=4, elite_fraction=0.1)
    params = optimizer.init_params(seed=0)
    candidates = np.arange(32, dtype=np.float64).reshape(4, 4, 2)
    costs = np.array([3.0, 2.0, 0.0, 1.0], dtype=np.float64)
    params = optimizer.update_params(params, candidates, costs)
    np.testing.assert_array_equal(params.mean, candidates[2])
    assert np.isfinite(params.mean).all()


def test_optimize_rejects_mismatched_iteration_budget(quadratic) -> None:
    optimizer = _make(num_iterations=4)
    params = optimizer.init_params(seed=0)
    with pytest.raises(ValueError, match="must match"):
        optimizer.optimize(params, quadratic(np.zeros((4, 2))), iterations=3)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"num_candidates": 1}, "num_candidates"),
        ({"num_iterations": 0}, "num_iterations"),
        ({"first_knot_scale": 0.0}, "first_knot_scale"),
        ({"last_knot_scale": np.inf}, "last_knot_scale"),
        ({"last_knot_scale": 0.4}, "last_knot_scale"),
        ({"final_noise_scale": 0.0}, "final_noise_scale"),
        ({"elite_fraction": 0.0}, "elite_fraction"),
        ({"temperature": 0.0}, "temperature"),
        ({"exploit_fraction": 1.0}, "exploit_fraction"),
        ({"exploit_noise_scale": -1.0}, "exploit_noise_scale"),
        ({"exploit_noise_scale": 1.1}, "exploit_noise_scale"),
        ({"noise_scale": [0.1, 0.2, 0.3]}, "noise_scale"),
    ],
)
def test_rejects_invalid_configuration(override, match) -> None:
    with pytest.raises(ValueError, match=match):
        _make(**override)


def test_converges(quadratic) -> None:
    optimizer = _make(
        num_candidates=128,
        num_knots=4,
        dim=2,
        noise_scale=0.5,
        final_noise_scale=0.1,
        num_iterations=30,
        elite_fraction=0.25,
        temperature=0.5,
        exploit_fraction=0.1,
        exploit_noise_scale=0.1,
    )
    target = np.full((4, 2), 1.0)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert (
        report.iterations[-1].best_observed_cost
        < 0.2 * report.iterations[0].best_observed_cost
    )


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
