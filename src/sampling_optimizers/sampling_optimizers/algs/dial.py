# Copyright (c) 2024 Vince Kurtz
# Adapted from hydrax for the BFM-TTC NumPy optimizer interface.
# The upstream MIT notice is retained in THIRD_PARTY_NOTICES.md.
"""Diffusion-Inspired Annealing for Legged MPC (DIAL)."""

from __future__ import annotations

import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.base import BatchEvaluator, SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizeReport, OptimizerParams


@register("dial")
class DIAL(SamplingOptimizer):
    """Diffusion-Inspired Annealing for Legged MPC (DIAL) based on
    https://arxiv.org/abs/2409.15610.

    DIAL-MPC is MPPI with a dual-loop, annealed sampling covariance that:
        - Decreases across optimisation iterations (trajectory-level annealing).
        - Increases along the planning horizon (action-level annealing).

    The noise level is given by:

        sigma[i, k] = sigma_0 * exp(-i/(beta_1*N) - (K-1-k)/(beta_2*K))

    where:
        - sigma_0 is the tunable `noise_level`,
        - beta_1 is the tunable `beta_opt_iter`,
        - beta_2 is the tunable `beta_horizon`,
        - i in {0,...,N-1} is the optimisation iteration (wrapped modulo N),
        - k in {0,...,K-1} indexes the knot along the horizon, and
        - N is `num_iterations` and K is `num_knots`.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        noise_level: float,
        beta_opt_iter: float,
        beta_horizon: float,
        temperature: float,
        num_iterations: int,
    ) -> None:
        if beta_opt_iter <= 0 or beta_horizon <= 0:
            raise ValueError("beta_opt_iter and beta_horizon must be positive")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        super().__init__(num_candidates, num_knots, dim)
        self.noise_level = float(noise_level)
        self.beta_opt_iter = float(beta_opt_iter)
        self.beta_horizon = float(beta_horizon)
        self.temperature = float(temperature)
        self.num_iterations = int(num_iterations)

    def optimize(
        self, params: OptimizerParams, evaluate: BatchEvaluator, iterations: int
    ) -> tuple[OptimizerParams, OptimizeReport]:
        if iterations != self.num_iterations:
            raise ValueError(
                "DIAL optimize iterations must match num_iterations; "
                f"got {iterations} and {self.num_iterations}"
            )
        return super().optimize(params, evaluate, iterations)

    def _sigma(self, iteration: int) -> np.ndarray:
        i = iteration % self.num_iterations
        k = np.arange(self.num_knots)
        return self.noise_level * np.exp(
            -i / (self.beta_opt_iter * self.num_iterations)
            - (self.num_knots - 1 - k) / (self.beta_horizon * self.num_knots)
        )

    def sample_candidates(
        self, params: OptimizerParams
    ) -> tuple[np.ndarray, OptimizerParams]:
        noise = params.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        sigma = self._sigma(params.iteration)
        return params.mean + sigma[None, :, None] * noise, params

    def update_params(
        self, params: OptimizerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> OptimizerParams:
        weights = softmax_weights(costs, self.temperature)
        params.mean = (weights[:, None, None] * candidates).sum(axis=0)
        return params
