import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

import sampling_optimizers  # noqa: F401  registers built-ins via eager algs import
from sampling_optimizers.registry import available, get_optimizer

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "optimizers"

# CBO values deviate from the plan's verbatim block for the same reason as
# tests/optimizers/test_cbo.py: the plan's initial_noise_level=0.5 /
# noise_weight=0.3 collapse the swarm before the convergence bound is met.
FACTORIES = {
    "predictive_sampling": dict(noise_level=0.4),
    "mppi": dict(noise_level=0.4, temperature=0.1),
    "mppi_cma": dict(
        initial_noise_level=0.5,
        temperature=0.1,
        minimum_noise_level=0.05,
        covariance_adaptation_rate=0.1,
    ),
    "cem": dict(num_elites=8, sigma_start=0.5, sigma_min=0.05),
    "dial": dict(
        noise_level=0.5,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        num_iterations=30,
    ),
    "cbo": dict(
        initial_noise_level=1.0,
        temperature=0.1,
        consensus_weight=1.0,
        noise_weight=0.7,
        step_size=0.5,
    ),
    "weighted_elites": dict(
        noise_scale=0.4, elite_fraction=0.25, temperature=0.5, noise_anneal=0.95
    ),
    "spider": dict(
        noise_scale=0.5,
        first_knot_scale=0.5,
        last_knot_scale=1.0,
        final_noise_scale=0.1,
        num_iterations=30,
        elite_fraction=0.25,
        temperature=0.5,
        exploit_fraction=0.1,
        exploit_noise_scale=0.1,
    ),
}

OPTIONAL_CONFIGS = {"evosax"}


def _build(name: str):
    return get_optimizer(name)(num_candidates=65, num_knots=3, dim=2, **FACTORIES[name])


def test_builtin_qualification_set_is_closed() -> None:
    # `import sampling_optimizers` alone must suffice (registry loading contract).
    registered = {name for name in FACTORIES if name in available()}
    configured = {path.stem for path in CONFIGS.glob("*.yaml")}
    assert registered == set(FACTORIES)
    assert configured == set(FACTORIES) | OPTIONAL_CONFIGS


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_improves_on_quadratic(name: str, quadratic) -> None:
    target = np.full((3, 2), 1.0)
    optimizer = _build(name)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=30)
    assert (
        report.iterations[-1].best_observed_cost
        < 0.5 * report.iterations[0].best_observed_cost
    )


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_seed_determinism(name: str, quadratic) -> None:
    target = np.full((3, 2), 0.7)
    finals = []
    for _ in range(2):
        optimizer = _build(name)
        params = optimizer.init_params(seed=11)
        params, _ = optimizer.optimize(params, quadratic(target), iterations=30)
        finals.append(params.mean.copy())
    np.testing.assert_array_equal(finals[0], finals[1])


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_default_yaml_constructs_optimizer(name: str) -> None:
    data = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert data.pop("algorithm") == name
    iterations = data["num_iterations"]
    optimizer_type = get_optimizer(name)
    accepted = inspect.signature(optimizer_type).parameters
    optimizer = optimizer_type(
        num_knots=3,
        dim=2,
        **{key: value for key, value in data.items() if key in accepted},
    )
    assert optimizer.num_candidates >= 1
    params = optimizer.init_params(seed=0)
    optimizer.optimize(params, lambda x: np.square(x).sum(axis=(1, 2)), iterations)
