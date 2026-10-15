import math

import numpy as np
import pytest

from sampling_optimizers.algs.weighted_elites import WeightedElites
from sampling_optimizers.registry import get_optimizer


def _make(**kw):
    defaults = dict(
        num_candidates=17,  # 1 mean + 16 antithetic
        num_knots=4,
        dim=3,
        noise_scale=np.array([0.1, 0.2, 0.3]),
        elite_fraction=0.25,
        temperature=0.5,
        noise_anneal=0.8,
        antithetic=True,
        smoothing_passes=1,
    )
    defaults.update(kw)
    return WeightedElites(**defaults)


def _reference_sample(mean, n_random, sigma, rng, antithetic, smoothing_passes,
                      freeze_first, freeze_last):
    """Transcribed from sil_wbc qpos_knots.py:374-411 (project = identity)."""
    n_draw = n_random // 2 if antithetic else n_random
    K, D = mean.shape
    z = rng.standard_normal((n_draw, K, D))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    z = z * np.asarray(sigma, dtype=np.float64).reshape(1, 1, -1)
    for _ in range(smoothing_passes):
        s = z.copy()
        s[:, 1:-1] = 0.25 * z[:, :-2] + 0.5 * z[:, 1:-1] + 0.25 * z[:, 2:]
        z = s
    if freeze_first:
        z[:, 0] = 0.0
    if freeze_last:
        z[:, -1] = 0.0
    return np.concatenate([mean[None], mean[None] + z], axis=0)


def _reference_update(candidates, costs, elite_fraction, temperature):
    """Transcribed from sil_wbc receding_sampling.py:114-164 + 3269-3312."""
    order = np.argsort(costs, kind="stable")
    elite_count = max(1, int(math.ceil(len(candidates) * elite_fraction)))
    elites = order[:elite_count]
    elite_costs = costs[elites].astype(np.float64)
    scale = max(float(elite_costs.std()), 1e-9)
    logits = -(elite_costs - float(elite_costs.min())) / (
        float(temperature) * scale
    )
    logits -= float(logits.max())
    w = np.exp(logits)
    w /= w.sum()
    return (candidates[elites] * w[:, None, None]).sum(axis=0)


def test_registered() -> None:
    assert get_optimizer("weighted_elites") is WeightedElites


def test_antithetic_requires_even_random_count() -> None:
    with pytest.raises(ValueError, match="antithetic"):
        _make(num_candidates=16)  # 15 random draws: odd


def test_sample_matches_transcribed_reference_bytes() -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=42, initial_mean=np.full((4, 3), 0.1))
    candidates, _ = optimizer.sample_candidates(params)
    ref_rng = np.random.default_rng(42)
    expected = _reference_sample(
        params.mean, n_random=16, sigma=[0.1, 0.2, 0.3], rng=ref_rng,
        antithetic=True, smoothing_passes=1,
        freeze_first=False, freeze_last=False,
    )
    np.testing.assert_array_equal(candidates, expected)
    np.testing.assert_array_equal(candidates[0], params.mean)


def test_sample_applies_anneal_and_freeze() -> None:
    optimizer = _make(freeze_first_knot=True, freeze_last_knot=True)
    params = optimizer.init_params(seed=0, initial_mean=np.full((4, 3), 0.1))
    params.iteration = 2  # sigma scaled by 0.8**2
    candidates, _ = optimizer.sample_candidates(params)
    np.testing.assert_array_equal(candidates[1:, 0], np.broadcast_to(params.mean[0], (16, 3)))
    np.testing.assert_array_equal(candidates[1:, -1], np.broadcast_to(params.mean[-1], (16, 3)))
    ref_rng = np.random.default_rng(0)
    expected = _reference_sample(
        params.mean, 16, np.array([0.1, 0.2, 0.3]) * 0.8**2, ref_rng,
        antithetic=True, smoothing_passes=1, freeze_first=True, freeze_last=True,
    )
    np.testing.assert_array_equal(candidates, expected)


def test_update_matches_transcribed_reference(quadratic) -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=7)
    candidates, params = optimizer.sample_candidates(params)
    costs = quadratic(np.full((4, 3), 0.5))(candidates)
    params = optimizer.update_params(params, candidates, costs)
    expected = _reference_update(candidates, costs, elite_fraction=0.25, temperature=0.5)
    np.testing.assert_array_equal(params.mean, expected)


def test_converges(quadratic) -> None:
    target = np.full((4, 3), 0.5)
    optimizer = _make(noise_scale=0.3, noise_anneal=0.95)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert report.iterations[-1].best_observed_cost < 0.2 * report.iterations[0].best_observed_cost


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(lambda: _make(num_knots=3, dim=2, noise_scale=0.3))
