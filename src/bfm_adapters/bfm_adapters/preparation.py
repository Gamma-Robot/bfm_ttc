"""Offline ONNX transforms registered by stable recipe name and version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


ModelTransform = Callable[[Any], None]
_TRANSFORMS: dict[str, tuple[int, ModelTransform]] = {}


def register_transform(name: str, version: int):
    """Register one in-place, model-only offline transform."""
    if not name or isinstance(version, bool) or version < 1:
        raise ValueError("transform name and positive version are required")

    def decorator(transform: ModelTransform) -> ModelTransform:
        if name in _TRANSFORMS:
            raise ValueError(f"transform {name!r} is already registered")
        _TRANSFORMS[name] = (version, transform)
        return transform

    return decorator


def apply_transform(name: str, version: int, model: Any) -> None:
    """Apply exactly the registered recipe version to an ONNX model value."""
    try:
        registered_version, transform = _TRANSFORMS[name]
    except KeyError:
        raise KeyError(
            f"unknown bfm transform {name!r}; available: {sorted(_TRANSFORMS)}"
        ) from None
    if version != registered_version:
        raise ValueError(
            f"transform {name!r} version {version} != registered {registered_version}"
        )
    transform(model)


@register_transform("relax_leading", 1)
def relax_leading_batch(model: Any) -> None:
    """Replace fixed leading-one policy I/O dimensions with symbolic batch."""
    io_names = {value.name for value in (*model.graph.input, *model.graph.output)}
    keep = [value for value in model.graph.value_info if value.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        dims = value.type.tensor_type.shape.dim
        if len(dims) >= 2 and dims[0].HasField("dim_value") and dims[0].dim_value == 1:
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "batch"
