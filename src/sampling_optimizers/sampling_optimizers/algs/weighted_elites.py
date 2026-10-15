"""Weighted-elites sampling over normalized latent knots."""

from __future__ import annotations

import math

import numpy as np

from sampling_optimizers.base import SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizerParams


@register("weighted_elites")
class WeightedElites(SamplingOptimizer):
    """Weighted-elites sampling over normalized latent knots.

    Samples: candidate 0 is the mean; others are (optionally antithetic)
    Gaussian noise scaled by ``noise_scale * noise_anneal**iteration``,
    1-2-1 smoothed along knots, with optional first/last-knot freeze.

    Update: softmax-weighted mean over the top ``ceil(N * elite_fraction)``
    elites (negated standardized costs, temperature ``T``).
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        noise_scale: float | np.ndarray,
        elite_fraction: float,
        temperature: float,
        noise_anneal: float = 1.0,
        antithetic: bool = True,
        smoothing_passes: int = 1,
        freeze_first_knot: bool = False,
        freeze_last_knot: bool = False,
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        num_random = num_candidates - 1
        if antithetic and num_random % 2 != 0:
            raise ValueError(
                "antithetic sampling needs an even random-candidate count; "
                f"got num_candidates={num_candidates} (candidate 0 is the mean)"
            )
        self.noise_scale = np.broadcast_to(
            np.asarray(noise_scale, dtype=np.float64), (dim,)
        ).copy()
        self.elite_fraction = float(elite_fraction)
        self.temperature = float(temperature)
        self.noise_anneal = float(noise_anneal)
        self.antithetic = bool(antithetic)
        self.smoothing_passes = int(smoothing_passes)
        self.freeze_first_knot = bool(freeze_first_knot)
        self.freeze_last_knot = bool(freeze_last_knot)

    def sample_candidates(
        self, params: OptimizerParams
    ) -> tuple[np.ndarray, OptimizerParams]:
        num_random = self.num_candidates - 1
        num_draw = num_random // 2 if self.antithetic else num_random
        noise = params.rng.standard_normal(
            (num_draw, self.num_knots, self.dim)
        )
        if self.antithetic:
            noise = np.concatenate([noise, -noise], axis=0)
        sigma = self.noise_scale * self.noise_anneal**params.iteration
        noise = noise * sigma.reshape(1, 1, -1)
        for _ in range(max(0, self.smoothing_passes)):
            smoothed = noise.copy()
            smoothed[:, 1:-1] = (
                0.25 * noise[:, :-2]
                + 0.5 * noise[:, 1:-1]
                + 0.25 * noise[:, 2:]
            )
            noise = smoothed
        if self.freeze_first_knot:
            noise[:, 0] = 0.0
        if self.freeze_last_knot:
            noise[:, -1] = 0.0
        return (
            np.concatenate([params.mean[None], params.mean[None] + noise], axis=0),
            params,
        )

    def update_params(
        self, params: OptimizerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> OptimizerParams:
        order = np.argsort(costs, kind="stable")
        elite_count = max(
            1, int(math.ceil(len(candidates) * self.elite_fraction))
        )
        elites = order[:elite_count]
        elite_costs = costs[elites].astype(np.float64)
        scale = max(float(elite_costs.std()), 1e-9)
        logits = -(elite_costs - float(elite_costs.min())) / (
            self.temperature * scale
        )
        logits -= float(logits.max())
        weights = np.exp(logits)
        weights /= weights.sum()
        params.mean = (candidates[elites] * weights[:, None, None]).sum(axis=0)
        return params
