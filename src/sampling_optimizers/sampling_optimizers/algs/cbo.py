# Copyright (c) 2024 Vince Kurtz
# Adapted from hydrax for the BFM-TTC NumPy optimizer interface.
# The upstream MIT notice is retained in THIRD_PARTY_NOTICES.md.
"""Consensus-based optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizerParams


@dataclass
class CBOParams(OptimizerParams):
    """Policy parameters for consensus-based optimization.

    Attributes:
        samples: Particle positions, shape (num_candidates, num_knots, dim).
    """

    samples: np.ndarray = None  # set by init_params


@register("cbo")
class CBO(SamplingOptimizer):
    """Consensus-based optimization.

    CBO (https://en.wikipedia.org/wiki/Consensus_based_optimization) simulates
    a swarm of particles Uᵢ according to a stochastic differential equation

        dUᵢ = - λ (Uᵢ - μ) dt + σ |Uᵢ - μ| dBᵢ,

    where μ = ∑ᵢ wᵢ Uᵢ is the consensus, wᵢ = exp(-J(Uᵢ)/T) / ∑ⱼ exp(-J(Uⱼ)/T)
    is the normalized weight of each particle, J(Uᵢ) is the cost of the control
    sequence Uᵢ, and Bᵢ is Brownian noise.

    This algorithm has global convergence properties for nonconvex problems in
    the mean-field (infinite particle) limit (https://arxiv.org/abs/2103.15130).
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        initial_noise_level: float,
        temperature: float,
        consensus_weight: float,
        noise_weight: float,
        step_size: float = 0.1,
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        self.initial_noise_level = float(initial_noise_level)
        self.temperature = float(temperature)
        self.consensus_weight = float(consensus_weight)
        self.noise_weight = float(noise_weight)
        self.step_size = float(step_size)

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> CBOParams:
        base = super().init_params(seed, initial_mean)
        noise = base.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        samples = base.mean + self.initial_noise_level * noise
        return CBOParams(mean=base.mean, rng=base.rng, samples=samples)

    def sample_candidates(
        self, params: CBOParams
    ) -> tuple[np.ndarray, CBOParams]:
        return params.samples, params

    def update_params(
        self, params: CBOParams, candidates: np.ndarray, costs: np.ndarray
    ) -> CBOParams:
        weights = softmax_weights(costs, self.temperature)
        mean = (weights[:, None, None] * candidates).sum(axis=0)
        deviation = candidates - mean
        noise = params.rng.standard_normal(candidates.shape)
        params.samples = (
            candidates
            - self.consensus_weight * deviation * self.step_size
            + self.noise_weight
            * np.abs(deviation)
            * np.sqrt(self.step_size)
            * noise
        )
        params.mean = mean
        return params
