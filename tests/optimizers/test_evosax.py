import os
from pathlib import Path

import numpy as np
import pytest
import yaml

if (
    os.environ.get("JAX_PLATFORMS") != "cpu"
    or os.environ.get("JAX_ENABLE_X64") != "1"
):
    pytest.skip(
        "EvoSax qualification requires JAX_PLATFORMS=cpu and JAX_ENABLE_X64=1",
        allow_module_level=True,
    )

pytest.importorskip("evosax")

from sampling_optimizers.algs.evosax import Evosax, EvosaxParams
from sampling_optimizers.registry import available, get_optimizer


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "optimizers" / "evosax.yaml"


def _make(**overrides) -> Evosax:
    values = dict(
        num_candidates=64,
        num_knots=3,
        dim=2,
        strategy="sep_cma_es",
        sigma_start=0.5,
        sigma_min=0.02,
        sigma_max=2.0,
    )
    values.update(overrides)
    return Evosax(**values)


def test_registered_and_default_yaml_constructs() -> None:
    assert "evosax" in available()
    assert get_optimizer("evosax") is Evosax
    data = yaml.safe_load(CONFIG.read_text())
    assert data.pop("algorithm") == "evosax"
    assert data.pop("num_iterations") > 0
    optimizer = Evosax(num_knots=3, dim=2, **data)
    assert optimizer.strategy_name == "sep_cma_es"


def test_init_and_sample_preserve_numpy_float64_interface() -> None:
    optimizer = _make()
    initial = np.full((3, 2), 0.25, dtype=np.float64)
    params = optimizer.init_params(seed=7, initial_mean=initial)
    assert isinstance(params, EvosaxParams)
    np.testing.assert_array_equal(params.mean, initial)

    candidates, params = optimizer.sample_candidates(params)
    assert candidates.shape == (64, 3, 2)
    assert candidates.dtype == np.float64
    assert np.isfinite(candidates).all()


def test_seed_determinism_is_byte_exact(quadratic) -> None:
    finals = []
    first_populations = []
    for _ in range(2):
        optimizer = _make()
        params = optimizer.init_params(seed=123)
        population, params = optimizer.sample_candidates(params)
        first_populations.append(population)
        costs = quadratic(np.full((3, 2), 0.7))(population)
        params = optimizer.update_params(params, population, costs)
        params, _ = optimizer.optimize(
            params,
            quadratic(np.full((3, 2), 0.7)),
            iterations=4,
        )
        finals.append(params.mean)

    np.testing.assert_array_equal(first_populations[0], first_populations[1])
    np.testing.assert_array_equal(finals[0], finals[1])


def test_improves_on_quadratic(quadratic) -> None:
    optimizer = _make()
    target = np.ones((3, 2), dtype=np.float64)
    params = optimizer.init_params(seed=0)
    params, report = optimizer.optimize(params, quadratic(target), iterations=20)
    assert report.iterations[-1].best_observed_cost < (
        0.1 * report.iterations[0].best_observed_cost
    )
    assert np.linalg.norm(params.mean - target) < 0.2


def test_rejects_unqualified_strategy_and_bad_scales() -> None:
    with pytest.raises(ValueError, match="unsupported EvoSax strategy"):
        _make(strategy="cma_es")
    with pytest.raises(ValueError, match="sigma scales"):
        _make(sigma_min=0.6)
    with pytest.raises(ValueError, match="num_candidates"):
        _make(num_candidates=1)
