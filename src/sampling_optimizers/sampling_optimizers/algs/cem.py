# Copyright (c) 2024 Vince Kurtz
# Adapted from hydrax for the BFM-TTC NumPy optimizer interface.
# The upstream MIT notice is retained in THIRD_PARTY_NOTICES.md.
"""Cross-entropy method with diagonal covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizerParams


@dataclass
class CEMParams(OptimizerParams):
    """Policy parameters for the cross-entropy method.

    Attributes:
        cov: Per-coordinate standard deviations of the sampling distribution,
            shape (num_knots, dim).
    """

    cov: np.ndarray = None  # set by init_params


@register("cem")
class CEM(SamplingOptimizer):
    """Cross-entropy method with diagonal covariance.

    Fits a diagonal Gaussian to the ``num_elites`` lowest-cost samples,
    flooring standard deviations at ``sigma_min``. An ``explore_fraction`` of
    candidates may be redrawn at ``sigma_start`` each iteration.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        explore_fraction: float = 0.0,
    ) -> None:
        if not 0.0 <= explore_fraction <= 1.0:
            raise ValueError(
                f"explore_fraction must be in [0, 1], got {explore_fraction}"
            )
        super().__init__(num_candidates, num_knots, dim)
        self.num_elites = int(num_elites)
        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.num_explore = int(num_candidates * explore_fraction)

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> CEMParams:
        base = super().init_params(seed, initial_mean)
        cov = np.full_like(base.mean, self.sigma_start)
        return CEMParams(mean=base.mean, rng=base.rng, cov=cov)

    def sample_candidates(
        self, params: CEMParams
    ) -> tuple[np.ndarray, CEMParams]:
        num_main = self.num_candidates - self.num_explore
        shape = (self.num_knots, self.dim)
        main = params.mean + params.cov * params.rng.standard_normal(
            (num_main, *shape)
        )
        explore = params.mean + self.sigma_start * params.rng.standard_normal(
            (self.num_explore, *shape)
        )
        return np.concatenate([main, explore], axis=0), params

    def update_params(
        self, params: CEMParams, candidates: np.ndarray, costs: np.ndarray
    ) -> CEMParams:
        elites = candidates[np.argsort(costs, kind="stable")[: self.num_elites]]
        params.mean = elites.mean(axis=0)
        params.cov = np.maximum(elites.std(axis=0), self.sigma_min)
        return params
