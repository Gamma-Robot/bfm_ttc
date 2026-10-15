"""Optional adapter for qualified distribution-based EvoSax strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based import Sep_CMA_ES

from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.types import OptimizerParams


@dataclass
class EvosaxParams(OptimizerParams):
    """Optimizer-owned EvoSax distribution state."""

    strategy_state: Any = None


class Evosax(SamplingOptimizer):
    """CPU/x64 adapter for qualified EvoSax distribution strategies.

    The public optimizer interface remains NumPy float64. JAX arrays, PRNG keys,
    and the EvoSax state stay inside this adapter. Random keys are derived from
    ``OptimizerParams.rng``, preserving the repository-wide seed contract.

    Only ``sep_cma_es`` is qualified initially. Population-based EvoSax
    algorithms do not have the distribution-mean proposal required by the
    current ``SamplingOptimizer`` interface.
    """

    _QUALIFIED_STRATEGIES = ("sep_cma_es",)

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        strategy: str,
        sigma_start: float,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        if strategy not in self._QUALIFIED_STRATEGIES:
            raise ValueError(
                f"unsupported EvoSax strategy {strategy!r}; qualified: "
                f"{list(self._QUALIFIED_STRATEGIES)}"
            )
        if num_candidates < 2:
            raise ValueError("EvoSax requires num_candidates >= 2")
        if not all(
            np.isfinite(value) for value in (sigma_start, sigma_min, sigma_max)
        ):
            raise ValueError("sigma_start, sigma_min, and sigma_max must be finite")
        if not 0.0 < sigma_min <= sigma_start <= sigma_max:
            raise ValueError(
                "sigma scales must satisfy 0 < sigma_min <= sigma_start <= "
                f"sigma_max; got ({sigma_min}, {sigma_start}, {sigma_max})"
            )
        if not jax.config.x64_enabled:
            raise RuntimeError(
                "EvoSax requires JAX x64; set JAX_ENABLE_X64=1 before Python starts"
            )
        if jax.default_backend() != "cpu":
            raise RuntimeError(
                "EvoSax is qualified on CPU only; set JAX_PLATFORMS=cpu before "
                "Python starts"
            )

        super().__init__(num_candidates, num_knots, dim)
        self.strategy_name = strategy
        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._flat_dim = num_knots * dim
        self._device = jax.devices("cpu")[0]

        with jax.default_device(self._device):
            solution = jnp.zeros(self._flat_dim, dtype=jnp.float64)
            self._strategy = Sep_CMA_ES(
                population_size=num_candidates,
                solution=solution,
            )
            self._strategy_params = self._strategy.default_params.replace(
                std_init=self.sigma_start,
                std_min=self.sigma_min,
                std_max=self.sigma_max,
            )

    def _next_key(self, params: OptimizerParams) -> jax.Array:
        seed = int(
            params.rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        return jax.random.PRNGKey(seed)

    def _to_device(self, value: np.ndarray) -> jax.Array:
        with jax.default_device(self._device):
            return jax.device_put(
                jnp.asarray(value, dtype=jnp.float64),
                self._device,
            )

    @staticmethod
    def _to_numpy(value: jax.Array, *, label: str) -> np.ndarray:
        out = np.asarray(jax.device_get(value))
        if out.dtype != np.float64:
            raise RuntimeError(
                f"EvoSax produced {label} with dtype {out.dtype}; expected float64"
            )
        return out.copy()

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> EvosaxParams:
        base = super().init_params(seed, initial_mean)
        mean = self._to_device(base.mean.reshape(self._flat_dim))
        with jax.default_device(self._device):
            state = self._strategy.init(
                key=self._next_key(base),
                mean=mean,
                params=self._strategy_params,
            )
        return EvosaxParams(
            mean=base.mean,
            rng=base.rng,
            iteration=base.iteration,
            strategy_state=state,
        )

    def sample_candidates(
        self, params: EvosaxParams
    ) -> tuple[np.ndarray, EvosaxParams]:
        with jax.default_device(self._device):
            population, state = self._strategy.ask(
                self._next_key(params),
                params.strategy_state,
                self._strategy_params,
            )
        candidates = self._to_numpy(population, label="population").reshape(
            self.num_candidates,
            self.num_knots,
            self.dim,
        )
        params.strategy_state = state
        return candidates, params

    def update_params(
        self,
        params: EvosaxParams,
        candidates: np.ndarray,
        costs: np.ndarray,
    ) -> EvosaxParams:
        population = self._to_device(
            np.asarray(candidates, dtype=np.float64).reshape(
                self.num_candidates,
                self._flat_dim,
            )
        )
        fitness = self._to_device(np.asarray(costs, dtype=np.float64))
        with jax.default_device(self._device):
            state, _ = self._strategy.tell(
                key=self._next_key(params),
                population=population,
                fitness=fitness,
                state=params.strategy_state,
                params=self._strategy_params,
            )
            mean = self._strategy.get_mean(state)
        params.strategy_state = state
        params.mean = self._to_numpy(mean, label="mean").reshape(
            self.num_knots,
            self.dim,
        )
        return params
