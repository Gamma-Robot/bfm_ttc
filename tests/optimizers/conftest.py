import numpy as np
import pytest


@pytest.fixture
def quadratic():
    def factory(target: np.ndarray):
        def evaluate(candidates: np.ndarray) -> np.ndarray:
            return ((candidates - target) ** 2).sum(axis=(1, 2))

        return evaluate

    return factory


@pytest.fixture
def assert_deterministic(quadratic):
    def check(make_optimizer, iterations: int = 4) -> None:
        finals = []
        for _ in range(2):
            optimizer = make_optimizer()
            target = np.full((optimizer.num_knots, optimizer.dim), 0.7)
            params = optimizer.init_params(seed=123)
            params, _ = optimizer.optimize(params, quadratic(target), iterations)
            finals.append(params.mean.copy())
        np.testing.assert_array_equal(finals[0], finals[1])

    return check
