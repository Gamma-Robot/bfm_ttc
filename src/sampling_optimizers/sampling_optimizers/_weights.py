"""Numerically stable cost-softmax weights (lower cost -> higher weight)."""

from __future__ import annotations

import numpy as np


def softmax_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or costs.size < 1:
        raise ValueError(
            f"costs must be a non-empty 1-D array, got shape {costs.shape}"
        )
    if not np.isfinite(costs).all():
        raise ValueError("costs must be finite")
    logits = -(costs - costs.min()) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()
