"""Plain-YAML configuration: ref resolution and strict dataclass binding."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

REF_KEY = "ref"
RESOLVED_EXPERIMENT_FILE = "resolved_experiment.yaml"

T = TypeVar("T")


class ConfigError(Exception):
    pass


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc


def _deep_merge(base: Any, override: Any) -> Any:
    # An explicit ref selects a component; ordinary mappings override its parameters.
    if isinstance(base, dict) and isinstance(override, dict) and REF_KEY not in override:
        out = copy.deepcopy(base)
        for key, value in override.items():
            out[key] = _deep_merge(out[key], value) if key in out else copy.deepcopy(value)
        return out
    return copy.deepcopy(override)


def _resolve(node: Any, configs_root: Path, stack: tuple[str, ...]) -> Any:
    if isinstance(node, dict):
        if REF_KEY in node:
            ref = str(node[REF_KEY])
            if ref in stack:
                raise ConfigError(f"ref cycle detected: {' -> '.join((*stack, ref))}")
            base = _resolve(_load_yaml(configs_root / ref), configs_root, (*stack, ref))
            overrides = {k: v for k, v in node.items() if k != REF_KEY}
            return _resolve(_deep_merge(base, overrides), configs_root, stack)
        return {k: _resolve(v, configs_root, stack) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(item, configs_root, stack) for item in node]
    return node


def resolve_config(path: Path, configs_root: Path) -> dict:
    raw = _load_yaml(Path(path))
    if not isinstance(raw, dict):
        raise ConfigError(f"top-level config must be a mapping: {path}")
    return _resolve(raw, Path(configs_root), stack=(str(path),))


def dataclass_from_dict(cls: type[T], data: dict) -> T:
    return _dataclass_from_dict(cls, data, field_path=cls.__name__)


def _dataclass_from_dict(cls: type[T], data: dict, *, field_path: str) -> T:
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ConfigError(f"{field_path} must be a mapping")
    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data).difference(field_names))
    if unknown:
        raise ConfigError(f"unknown fields at {field_path}: {unknown}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if dataclasses.is_dataclass(hint):
            value = _dataclass_from_dict(
                hint, value, field_path=f"{field_path}.{f.name}"
            )
        kwargs[f.name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"cannot build {field_path}: {exc}") from exc


def write_resolved_experiment(resolved: dict, out_dir: Path) -> Path:
    """Persist the fully resolved configuration executed by one experiment."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESOLVED_EXPERIMENT_FILE
    path.write_text(yaml.safe_dump(resolved, sort_keys=True))
    return path
