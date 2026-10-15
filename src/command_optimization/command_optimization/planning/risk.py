"""Trajectory-level risk reduction across realized physics scenarios."""

from __future__ import annotations

from typing import Literal

import numpy as np


RiskStrategy = Literal["mean", "worst"]


def reduce_scenario_costs(
    costs: np.ndarray, *, strategy: RiskStrategy
) -> np.ndarray:
    """Reduce finite full-trajectory costs from ``(N, R)`` to ``(N,)``."""
    values = np.asarray(costs)
    if values.dtype != np.float64 or values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("costs must be a non-empty float64 array with shape (N, R)")
    if not np.isfinite(values).all():
        raise ValueError("costs must be finite")
    if strategy == "mean":
        # Scale before summing so finite, non-negative objective terms cannot
        # overflow merely because several scenarios are aggregated.
        return np.sum(values / values.shape[1], axis=1, dtype=np.float64)
    if strategy == "worst":
        return np.max(values, axis=1)
    raise ValueError(f"unknown risk strategy {strategy!r}")
