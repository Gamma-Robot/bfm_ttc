"""BFM instance configuration and explicit family registry."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from bfm_adapters.adapter import BFMAdapter
from bfm_adapters.contracts import TensorSpec


def _make_immutable(value: object) -> object:
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("family_config keys must be non-empty strings")
            normalized[key] = _make_immutable(item)
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(_make_immutable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported family_config value {type(value).__name__}")


@dataclass(frozen=True)
class BFMInstanceConfig:
    """Shared, family-neutral portion of one pretrained bfm instance."""

    family: str
    instance: str
    artifacts: Mapping[str, str]
    control_period: float
    motion_command_query_offsets: tuple[int, ...]
    policy_inputs: tuple[TensorSpec, ...]
    policy_outputs: tuple[TensorSpec, ...]
    policy_joint_names: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    motion_command_body_names: tuple[str, ...]
    family_config: Mapping[str, object]
    motion_command_velocity_stencil: str = "centered"
    allow_cpu_ep_fallback: bool = False

    def __post_init__(self) -> None:
        if self.motion_command_velocity_stencil not in {"centered", "backward"}:
            raise ValueError(
                "motion_command_velocity_stencil must be 'centered' or 'backward'"
            )
        if not isinstance(self.allow_cpu_ep_fallback, bool):
            raise ValueError("allow_cpu_ep_fallback must be boolean")
        if (
            not isinstance(self.family, str)
            or not self.family
            or not isinstance(self.instance, str)
            or not self.instance
        ):
            raise ValueError("family and instance must be non-empty")
        artifacts = dict(self.artifacts)
        if not artifacts or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in artifacts.items()
        ):
            raise ValueError("artifacts must contain non-empty identities")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        if not math.isfinite(self.control_period) or self.control_period <= 0:
            raise ValueError("control_period must be finite and positive")
        offsets = tuple(self.motion_command_query_offsets)
        if (
            not offsets
            or any(not isinstance(offset, int) for offset in offsets)
            or len(offsets) != len(set(offsets))
        ):
            raise ValueError(
                "motion_command_query_offsets must contain unique integer offsets"
            )
        object.__setattr__(self, "motion_command_query_offsets", offsets)
        family_config = _make_immutable(self.family_config)
        if not isinstance(family_config, Mapping):
            raise TypeError("family_config must be a mapping")
        object.__setattr__(self, "family_config", family_config)
        for field in ("policy_inputs", "policy_outputs"):
            specs = getattr(self, field)
            names = [spec.name for spec in specs]
            if not specs or len(names) != len(set(names)):
                raise ValueError(f"{field} must have unique tensor names")
        for field in (
            "policy_joint_names",
            "command_joint_names",
            "motion_command_body_names",
        ):
            names = getattr(self, field)
            if not names or any(not name for name in names) or len(names) != len(set(names)):
                raise ValueError(f"{field} must contain unique, non-empty names")
        if set(self.policy_joint_names) != set(self.command_joint_names):
            raise ValueError(
                "policy_joint_names and command_joint_names must describe the same joints"
            )


def instance_config_from_mapping(data: Mapping[str, object]) -> BFMInstanceConfig:
    """Bind a resolved config mapping to the family-neutral instance schema."""

    if not isinstance(data, Mapping):
        raise ValueError("bfm config must be a mapping")
    fields = {
        "family",
        "instance",
        "artifacts",
        "control_period",
        "motion_command_query_offsets",
        "policy_inputs",
        "policy_outputs",
        "policy_joint_names",
        "command_joint_names",
        "motion_command_body_names",
        "family_config",
        "motion_command_velocity_stencil",
        "allow_cpu_ep_fallback",
    }
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown bfm config fields: {sorted(unknown)}")

    def sequence(field: str) -> tuple[object, ...]:
        values = data.get(field)
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        return tuple(values)

    def names(field: str) -> tuple[str, ...]:
        values = sequence(field)
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"{field} must contain strings")
        return values  # type: ignore[return-value]

    def specs(field: str) -> tuple[TensorSpec, ...]:
        values = data.get(field)
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        try:
            return tuple(
                TensorSpec(
                    name=value["name"],
                    dtype=value["dtype"],
                    non_batch_shape=tuple(value["non_batch_shape"]),
                )
                for value in values
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid {field}") from exc

    try:
        return BFMInstanceConfig(
            family=data["family"],
            instance=data["instance"],
            artifacts=data["artifacts"],
            control_period=float(data["control_period"]),
            motion_command_query_offsets=sequence("motion_command_query_offsets"),
            policy_inputs=specs("policy_inputs"),
            policy_outputs=specs("policy_outputs"),
            policy_joint_names=names("policy_joint_names"),
            command_joint_names=names("command_joint_names"),
            motion_command_body_names=names("motion_command_body_names"),
            family_config=data["family_config"],
            motion_command_velocity_stencil=data.get(
                "motion_command_velocity_stencil", "centered"
            ),
            allow_cpu_ep_fallback=data.get("allow_cpu_ep_fallback", False),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid bfm config") from exc


def load_instance_config(path: str | Path) -> BFMInstanceConfig:
    """Load the shared fields from one committed bfm YAML."""

    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"bfm config not found: {config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"bfm config must be a mapping: {config_path}")
    return instance_config_from_mapping(data)


BFMFactory = Callable[[BFMInstanceConfig], BFMAdapter]
_REGISTRY: dict[str, BFMFactory] = {}


def register(family: str):
    """Register one explicit bfm-family factory."""

    if not family:
        raise ValueError("bfm family must be non-empty")

    def decorator(factory: BFMFactory) -> BFMFactory:
        if not callable(factory):
            raise TypeError(f"register({family!r}) requires a callable factory")
        if family in _REGISTRY:
            raise ValueError(f"bfm family {family!r} already registered")
        _REGISTRY[family] = factory
        return factory

    return decorator


def get_family(family: str) -> BFMFactory:
    try:
        return _REGISTRY[family]
    except KeyError:
        raise KeyError(
            f"unknown bfm family {family!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
