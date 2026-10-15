# Copyright (c) 2024 Vince Kurtz
# Adapted from hydrax for the BFM-TTC NumPy optimizer interface.
# The upstream MIT notice is retained in THIRD_PARTY_NOTICES.md.
"""Model-predictive path integral control."""

from __future__ import annotations

import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizerParams


@register("mppi")
class MPPI(SamplingOptimizer):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        noise_level: float,
        temperature: float,
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)

    def sample_candidates(
        self, params: OptimizerParams
    ) -> tuple[np.ndarray, OptimizerParams]:
        noise = params.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        return params.mean + self.noise_level * noise, params

    def update_params(
        self, params: OptimizerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> OptimizerParams:
        weights = softmax_weights(costs, self.temperature)
        params.mean = (weights[:, None, None] * candidates).sum(axis=0)
        return params
