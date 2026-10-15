"""Name-keyed optimizer registry for config-driven selection.

Loading semantics: built-in algorithms register as a side effect of
importing `sampling_optimizers.algs`; `sampling_optimizers/__init__` imports it eagerly,
so `import sampling_optimizers` alone guarantees built-ins are available. Optional
optimizers are registered as lazy import paths and load only when selected.
Custom optimizers register when their defining module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

from sampling_optimizers.base import SamplingOptimizer

_REGISTRY: dict[str, type[SamplingOptimizer]] = {}


@dataclass(frozen=True)
class _OptionalOptimizer:
    module: str
    attribute: str
    dependency: str
    extra: str


_OPTIONAL_REGISTRY: dict[str, _OptionalOptimizer] = {}


def _validate_optimizer(name: str, cls: object) -> type[SamplingOptimizer]:
    if not (isinstance(cls, type) and issubclass(cls, SamplingOptimizer)):
        raise TypeError(
            f"optimizer {name!r} must be a SamplingOptimizer subclass, got {cls!r}"
        )
    return cls


def register(name: str):
    def decorator(cls: type[SamplingOptimizer]) -> type[SamplingOptimizer]:
        cls = _validate_optimizer(name, cls)
        if name in _REGISTRY or name in _OPTIONAL_REGISTRY:
            raise ValueError(f"optimizer {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def _register_optional(
    name: str,
    *,
    module: str,
    attribute: str,
    dependency: str,
    extra: str,
) -> None:
    """Register an optional optimizer without importing its dependency stack."""
    if name in _REGISTRY or name in _OPTIONAL_REGISTRY:
        raise ValueError(f"optimizer {name!r} already registered")
    _OPTIONAL_REGISTRY[name] = _OptionalOptimizer(
        module=module,
        attribute=attribute,
        dependency=dependency,
        extra=extra,
    )


def get_optimizer(name: str) -> type[SamplingOptimizer]:
    try:
        return _REGISTRY[name]
    except KeyError:
        pass

    optional = _OPTIONAL_REGISTRY.get(name)
    if optional is None:
        raise KeyError(
            f"unknown optimizer {name!r}; available: {list(available())}"
        ) from None
    if find_spec(optional.dependency) is None:
        raise ModuleNotFoundError(
            f"optimizer {name!r} requires optional dependency {optional.dependency!r}; "
            f"install sampling-optimizers[{optional.extra}]"
        ) from None

    module = import_module(optional.module)
    cls = _validate_optimizer(name, getattr(module, optional.attribute))
    _REGISTRY[name] = cls
    return cls


def available() -> tuple[str, ...]:
    optional = {
        name
        for name, entry in _OPTIONAL_REGISTRY.items()
        if find_spec(entry.dependency) is not None
    }
    return tuple(sorted(_REGISTRY.keys() | optional))
