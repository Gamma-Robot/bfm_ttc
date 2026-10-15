# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted for BFM-TTC: NumPy sampling and lower-is-better cost updates.
# See THIRD_PARTY_NOTICES.md.
"""SPIDER-style structured weighted-elites sampling."""

from __future__ import annotations

import numpy as np

from sampling_optimizers.base import BatchEvaluator, SamplingOptimizer
from sampling_optimizers.registry import register
from sampling_optimizers.types import OptimizeReport, OptimizerParams


_COST_STD_EPSILON = 1e-2


@register("spider")
class Spider(SamplingOptimizer):
    """Structured sampling used by SPIDER, adapted to lower-is-better costs.

    The sampler combines three ideas:

    - independent Gaussian perturbations whose scale grows along the knot
      horizon;
    - a global noise scale annealed across optimization iterations; and
    - a small exploit subpopulation sampled closer to the current mean.

    Candidate 0 is always the unperturbed mean. The update selects the
    lowest-cost elite fraction, standardizes its costs, and computes a
    temperature-softmax weighted mean. This is a NumPy implementation of the
    optimizer described by https://github.com/facebookresearch/spider; rollout,
    interpolation, early stopping, and embodiment-specific scaling are not
    part of this optimizer.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        noise_scale: float | np.ndarray,
        first_knot_scale: float,
        last_knot_scale: float,
        final_noise_scale: float,
        num_iterations: int,
        elite_fraction: float,
        temperature: float,
        exploit_fraction: float = 0.01,
        exploit_noise_scale: float = 0.01,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("SPIDER requires num_candidates >= 2")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError(
                f"elite_fraction must be in (0, 1], got {elite_fraction}"
            )
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive and finite, got {temperature}"
            )
        if not 0.0 <= exploit_fraction < 1.0:
            raise ValueError(
                f"exploit_fraction must be in [0, 1), got {exploit_fraction}"
            )
        if (
            not np.isfinite(exploit_noise_scale)
            or not 0.0 <= exploit_noise_scale <= 1.0
        ):
            raise ValueError(
                "exploit_noise_scale must be in [0, 1], got "
                f"{exploit_noise_scale}"
            )

        knot_endpoints = np.asarray(
            [first_knot_scale, last_knot_scale], dtype=np.float64
        )
        if not np.isfinite(knot_endpoints).all() or (knot_endpoints <= 0.0).any():
            raise ValueError(
                "first_knot_scale and last_knot_scale must be positive and finite"
            )
        if last_knot_scale < first_knot_scale:
            raise ValueError(
                "last_knot_scale must be >= first_knot_scale; got "
                f"{last_knot_scale} < {first_knot_scale}"
            )
        if not np.isfinite(final_noise_scale) or not 0.0 < final_noise_scale <= 1.0:
            raise ValueError(
                "final_noise_scale must be in (0, 1], got "
                f"{final_noise_scale}"
            )

        super().__init__(num_candidates, num_knots, dim)
        try:
            dimension_scale = np.broadcast_to(
                np.asarray(noise_scale, dtype=np.float64), (dim,)
            ).copy()
        except ValueError as exc:
            raise ValueError(
                f"noise_scale must be scalar or broadcast to shape {(dim,)}"
            ) from exc
        if not np.isfinite(dimension_scale).all() or (dimension_scale < 0.0).any():
            raise ValueError("noise_scale must be non-negative and finite")

        self.noise_scale = dimension_scale
        self.first_knot_scale = float(first_knot_scale)
        self.last_knot_scale = float(last_knot_scale)
        self.final_noise_scale = float(final_noise_scale)
        self.num_iterations = int(num_iterations)
        self.elite_fraction = float(elite_fraction)
        self.temperature = float(temperature)
        self.exploit_fraction = float(exploit_fraction)
        self.exploit_noise_scale = float(exploit_noise_scale)
        self.knot_scale = np.geomspace(
            self.first_knot_scale,
            self.last_knot_scale,
            num=self.num_knots,
            dtype=np.float64,
        )
        self.beta_trajectory = (
            self.final_noise_scale ** (1.0 / self.num_iterations)
            if self.num_iterations > 1
            else 1.0
        )
        self.num_exploit = int(self.num_candidates * self.exploit_fraction)

    def optimize(
        self, params: OptimizerParams, evaluate: BatchEvaluator, iterations: int
    ) -> tuple[OptimizerParams, OptimizeReport]:
        if iterations != self.num_iterations:
            raise ValueError(
                "SPIDER optimize iterations must match num_iterations; "
                f"got {iterations} and {self.num_iterations}"
            )
        return super().optimize(params, evaluate, iterations)

    def _sigma(self, iteration: int) -> np.ndarray:
        return (
            self.knot_scale[:, None]
            * self.noise_scale[None, :]
            * self.beta_trajectory**iteration
        )

    def sample_candidates(
        self, params: OptimizerParams
    ) -> tuple[np.ndarray, OptimizerParams]:
        noise = params.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        noise *= self._sigma(params.iteration)[None]
        noise[0] = 0.0
        if self.num_exploit > 0:
            noise[-self.num_exploit :] *= self.exploit_noise_scale
        return params.mean[None] + noise, params

    def update_params(
        self, params: OptimizerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> OptimizerParams:
        elite_count = max(1, int(len(candidates) * self.elite_fraction))
        elite_indices = np.argsort(costs, kind="stable")[:elite_count]
        elite_costs = costs[elite_indices].astype(np.float64)
        spread = (
            float(elite_costs.std(ddof=1)) if elite_count > 1 else 0.0
        )
        logits = -(elite_costs - float(elite_costs.mean())) / (
            self.temperature * (spread + _COST_STD_EPSILON)
        )
        logits -= float(logits.max())
        weights = np.exp(logits)
        weights /= weights.sum()
        params.mean = (
            candidates[elite_indices] * weights[:, None, None]
        ).sum(axis=0)
        return params
