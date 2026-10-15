"""sampling_optimizers: sampling-based black-box optimizers with a NumPy core."""

from sampling_optimizers.base import BatchEvaluator, SamplingOptimizer
from sampling_optimizers.registry import available, get_optimizer, register
from sampling_optimizers.types import (
    ConvergenceCriteria,
    IterationLimit,
    IterationRecord,
    OptimizeReport,
    OptimizerParams,
)

__all__ = [
    "BatchEvaluator",
    "SamplingOptimizer",
    "available",
    "get_optimizer",
    "register",
    "ConvergenceCriteria",
    "IterationLimit",
    "IterationRecord",
    "OptimizeReport",
    "OptimizerParams",
]

from sampling_optimizers import algs  # noqa: E402,F401  (registers built-ins)
