"""Shared dataclasses for sampling optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np


@dataclass
class OptimizerParams:
    """Mutable state of the sampling distribution.

        Optimizer-owned opaque state: callers (command_optimization) never mutate fields
    directly; warm starts go through ``init_params(initial_mean=...)``.

    Attributes:
        mean: Distribution center, shape (num_knots, dim), float64.
        rng: Stateful numpy generator; consuming it advances the stream.
        iteration: Number of completed optimize iterations within the
            current planning update (never spans planning updates).
    """

    mean: np.ndarray
    rng: np.random.Generator
    iteration: int = 0


@dataclass(frozen=True)
class ConvergenceCriteria:
    """Stop after consecutive iterations without sufficient best-cost improvement.

    The first finite observation establishes the best cost. The threshold is
    absolute and strict; stopping history is local to each optimize call.
    """

    patience: int
    min_cost_improvement: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.patience, (bool, np.bool_))
            or not isinstance(self.patience, Integral)
            or self.patience < 1
        ):
            raise ValueError("patience must be an integer >= 1")
        if (
            isinstance(self.min_cost_improvement, (bool, np.bool_))
            or not isinstance(self.min_cost_improvement, Real)
            or not np.isfinite(self.min_cost_improvement)
            or self.min_cost_improvement < 0.0
        ):
            raise ValueError("min_cost_improvement must be finite and non-negative")
        object.__setattr__(self, "patience", int(self.patience))
        object.__setattr__(self, "min_cost_improvement", float(self.min_cost_improvement))


IterationLimit = int | ConvergenceCriteria


# eq=False: generated __eq__/__hash__ are broken with ndarray fields
# (== raises ValueError, hash raises TypeError).
@dataclass(frozen=True, eq=False)
class IterationRecord:
    """Per-iteration record. "Observed" = from an actually evaluated candidate."""

    iteration: int
    best_observed_cost: float
    best_observed_candidate: np.ndarray
    mean_cost: float


@dataclass(frozen=True, eq=False)
class OptimizeReport:
    iterations: tuple[IterationRecord, ...] = field(default_factory=tuple)

    @property
    def best_observed_cost(self) -> float:
        if not self.iterations:
            raise ValueError("report has no iterations")
        return min(record.best_observed_cost for record in self.iterations)

    @property
    def best_observed_candidate(self) -> np.ndarray:
        if not self.iterations:
            raise ValueError("report has no iterations")
        best = min(self.iterations, key=lambda r: r.best_observed_cost)
        return best.best_observed_candidate.copy()
