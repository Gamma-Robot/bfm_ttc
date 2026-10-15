"""Algorithm implementations. Importing this package registers all built-ins."""

from sampling_optimizers.registry import _register_optional

from sampling_optimizers.algs import (  # noqa: F401
    cbo,
    cem,
    dial,
    mppi,
    mppi_cma,
    predictive_sampling,
    spider,
    weighted_elites,
)

_register_optional(
    "evosax",
    module="sampling_optimizers.algs.evosax",
    attribute="Evosax",
    dependency="evosax",
    extra="evosax",
)
