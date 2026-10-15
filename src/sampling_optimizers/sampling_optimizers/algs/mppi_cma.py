# Copyright (c) 2024 Vince Kurtz
# Adapted from hydrax for the BFM-TTC NumPy optimizer interface.
# The upstream MIT notice is retained in THIRD_PARTY_NOTICES.md.
"""Model-predictive path integral control with covariance matrix adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sampling_optimizers._weights import softmax_weights
from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizerParams


@dataclass
class MppiCmaParams(OptimizerParams):
    """Policy parameters for MPPI with covariance matrix adaptation.

    Attributes:
        covariance: Block-diagonal sampling covariance, shape
            (num_knots, dim, dim).
    """

    covariance: np.ndarray = None  # set by init_params


@register("mppi_cma")
class MppiCma(SamplingOptimizer):
    """Model-predictive path integral control with covariance matrix adaptation.

    Implements the block-diagonal variant of MPPI-CMA, as described in
    https://arxiv.org/abs/2506.22087, with the addition of a noise level floor.

    This minimum noise level is important for avoiding mode collapse in a
    predictive control setting, where the cost landscape is constantly shifting.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        initial_noise_level: float,
        temperature: float,
        minimum_noise_level: float | None = None,
        covariance_adaptation_rate: float = 0.1,
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        self.initial_noise_level = float(initial_noise_level)
        self.minimum_noise_level = float(
            initial_noise_level
            if minimum_noise_level is None
            else minimum_noise_level
        )
        self.temperature = float(temperature)
        self.alpha = float(covariance_adaptation_rate)

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> MppiCmaParams:
        base = super().init_params(seed, initial_mean)
        cov = np.eye(self.dim, dtype=np.float64) * self.initial_noise_level**2
        covariance = np.broadcast_to(cov, (self.num_knots, self.dim, self.dim)).copy()
        return MppiCmaParams(mean=base.mean, rng=base.rng, covariance=covariance)

    def sample_candidates(
        self, params: MppiCmaParams
    ) -> tuple[np.ndarray, MppiCmaParams]:
        noise = np.empty(
            (self.num_candidates, self.num_knots, self.dim), dtype=np.float64
        )
        zero = np.zeros(self.dim, dtype=np.float64)
        for k in range(self.num_knots):
            noise[:, k, :] = params.rng.multivariate_normal(
                mean=zero, cov=params.covariance[k], size=self.num_candidates
            )
        return params.mean + noise, params

    @staticmethod
    def _clamp_eigenvalues(cov: np.ndarray, min_eig: float) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(cov)
        clamped = np.maximum(eigvals, min_eig)
        return (eigvecs * clamped) @ eigvecs.T

    def update_params(
        self, params: MppiCmaParams, candidates: np.ndarray, costs: np.ndarray
    ) -> MppiCmaParams:
        weights = softmax_weights(costs, self.temperature)
        delta = candidates - params.mean
        outer = np.einsum("ijk,ijl->ijkl", delta, delta)
        new_cov = np.einsum("i,ijkl->jkl", weights, outer)
        new_cov = (1.0 - self.alpha) * params.covariance + self.alpha * new_cov
        min_eig = self.minimum_noise_level**2
        for k in range(self.num_knots):
            new_cov[k] = self._clamp_eigenvalues(new_cov[k], min_eig)
        params.mean = (weights[:, None, None] * candidates).sum(axis=0)
        params.covariance = new_cov
        return params
