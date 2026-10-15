import numpy as np

from sampling_optimizers.algs.dial import DIAL
from sampling_optimizers.registry import get_optimizer


def _make(num_iterations=4):
    return DIAL(
        num_candidates=64,
        num_knots=4,
        dim=2,
        noise_level=0.5,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        num_iterations=num_iterations,
    )


def test_registered() -> None:
    assert get_optimizer("dial") is DIAL


def test_sigma_schedule_shrinks_over_iterations_and_grows_along_horizon() -> None:
    optimizer = _make()
    s0 = optimizer._sigma(iteration=0)
    s3 = optimizer._sigma(iteration=3)
    assert s0.shape == (4,)
    assert (s3 < s0).all()  # trajectory-level annealing
    assert (np.diff(s0) > 0).all()  # later knots noisier
    # Wraps after num_iterations:
    np.testing.assert_allclose(optimizer._sigma(iteration=4), s0)


def test_sample_uses_scheduled_sigma() -> None:
    optimizer = _make()
    params = optimizer.init_params(seed=0)
    candidates, _ = optimizer.sample_candidates(params)
    assert candidates.shape == (64, 4, 2)
    spread = (candidates - params.mean).std(axis=(0, 2))
    assert (np.diff(spread) > 0).all()  # empirical spread grows along knots


def test_converges(quadratic) -> None:
    target = np.full((4, 2), 1.0)
    optimizer = _make(num_iterations=32)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=32)
    assert report.iterations[-1].best_observed_cost < 0.2 * report.iterations[0].best_observed_cost


def test_optimize_rejects_mismatched_iteration_budget(quadratic) -> None:
    optimizer = _make(num_iterations=4)
    params = optimizer.init_params(seed=0)
    with np.testing.assert_raises_regex(ValueError, "must match"):
        optimizer.optimize(params, quadratic(np.zeros((4, 2))), iterations=3)


def test_deterministic(assert_deterministic) -> None:
    assert_deterministic(_make)
