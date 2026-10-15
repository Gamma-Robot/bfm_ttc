"""Deterministic physics scenarios for robust Candidate Rollout evaluation."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, fields
from typing import Literal

import numpy as np


_DIGEST_DOMAIN = b"bfm-ttc.scenario-batch.v1\0"
_SCALE_FIELDS = (
    "contact_friction_scale",
    "joint_armature_scale",
    "joint_frictionloss_scale",
    "actuation_strength_scale",
)
_ARRAY_FIELDS = (
    *_SCALE_FIELDS,
    "root_linear_velocity_offset",
    "root_angular_velocity_offset",
    "joint_velocity_offset",
)


def _range(value: tuple[float, float], field: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must contain exactly two values")
    lower, upper = (float(item) for item in value)
    if not np.isfinite((lower, upper)).all() or lower <= 0.0 or lower > upper:
        raise ValueError(
            f"{field} must be finite, strictly positive, and ordered"
        )
    return lower, upper


@dataclass(frozen=True)
class RandomizationRanges:
    """Qualification distribution for one planning or execution domain."""

    contact_friction_scale: tuple[float, float] = (1.0, 1.0)
    joint_armature_scale: tuple[float, float] = (1.0, 1.0)
    joint_frictionloss_scale: tuple[float, float] = (1.0, 1.0)
    actuation_strength_scale: tuple[float, float] = (1.0, 1.0)
    root_linear_velocity_std: float = 0.0
    root_angular_velocity_std: float = 0.0
    joint_velocity_std: float = 0.0

    def __post_init__(self) -> None:
        for field in _SCALE_FIELDS:
            object.__setattr__(self, field, _range(getattr(self, field), field))
        for field in (
            "root_linear_velocity_std",
            "root_angular_velocity_std",
            "joint_velocity_std",
        ):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, value)

    @property
    def is_nominal(self) -> bool:
        return all(
            getattr(self, field) == (1.0, 1.0) for field in _SCALE_FIELDS
        ) and all(
            getattr(self, field) == 0.0
            for field in (
                "root_linear_velocity_std",
                "root_angular_velocity_std",
                "joint_velocity_std",
            )
        )


@dataclass(frozen=True)
class RobustnessConfig:
    """Finite scenario set fixed for each Planning Update."""

    enabled: bool = False
    num_scenarios: int = 1
    scenario_seed: int = 0
    risk: Literal["mean", "worst"] = "mean"
    ranges: RandomizationRanges = field(default_factory=RandomizationRanges)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("robustness.enabled must be bool")
        for name, value, minimum in (
            ("num_scenarios", self.num_scenarios, 1),
            ("scenario_seed", self.scenario_seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"robustness.{name} must be an integer >= {minimum}")
        if self.risk not in ("mean", "worst"):
            raise ValueError("robustness.risk must be mean or worst")
        if not isinstance(self.ranges, RandomizationRanges):
            raise TypeError("robustness.ranges must be RandomizationRanges")
        if not self.enabled and (
            self.num_scenarios != 1
            or self.scenario_seed != 0
            or self.risk != "mean"
            or not self.ranges.is_nominal
        ):
            raise ValueError("disabled robustness must keep its nominal defaults")
        if self.enabled and self.num_scenarios == 1 and not self.ranges.is_nominal:
            raise ValueError(
                "a single enabled scenario is reserved for nominal parity"
            )


def _owned(value: np.ndarray, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim or array.shape[0] < 1:
        raise ValueError(
            f"{field} must have {ndim} dimensions and a non-empty scenario axis"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class ScenarioBatch:
    """Realized scenario values with one immutable leading scenario axis."""

    contact_friction_scale: np.ndarray
    joint_armature_scale: np.ndarray
    joint_frictionloss_scale: np.ndarray
    actuation_strength_scale: np.ndarray
    root_linear_velocity_offset: np.ndarray
    root_angular_velocity_offset: np.ndarray
    joint_velocity_offset: np.ndarray

    def __post_init__(self) -> None:
        for field in _SCALE_FIELDS:
            value = _owned(getattr(self, field), field, 1)
            if (value <= 0.0).any():
                raise ValueError(f"{field} must be strictly positive")
            object.__setattr__(self, field, value)
        for field in (
            "root_linear_velocity_offset",
            "root_angular_velocity_offset",
            "joint_velocity_offset",
        ):
            object.__setattr__(
                self, field, _owned(getattr(self, field), field, 2)
            )
        count = self.contact_friction_scale.shape[0]
        if any(getattr(self, field).shape[0] != count for field in _ARRAY_FIELDS):
            raise ValueError("all scenario arrays must share one leading axis")
        for field in ("root_linear_velocity_offset", "root_angular_velocity_offset"):
            if getattr(self, field).shape[1] != 3:
                raise ValueError(f"{field} must have shape (R, 3)")
        if self.joint_velocity_offset.shape[1] < 1:
            raise ValueError("joint_velocity_offset must contain at least one joint")

    @property
    def count(self) -> int:
        return int(self.contact_friction_scale.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.joint_velocity_offset.shape[1])

    @property
    def digest(self) -> str:
        digest = hashlib.sha256(_DIGEST_DOMAIN)
        for field in fields(self):
            name = field.name.encode("ascii")
            value = np.ascontiguousarray(getattr(self, field.name), dtype="<f8")
            digest.update(struct.pack("<Q", len(name)))
            digest.update(name)
            digest.update(struct.pack("<Q", value.ndim))
            digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
            digest.update(value.tobytes(order="C"))
        return f"sha256:{digest.hexdigest()}"

    @classmethod
    def nominal(cls, *, count: int, num_joints: int) -> ScenarioBatch:
        return sample_scenarios(
            RandomizationRanges(),
            count=count,
            seed=0,
            num_joints=num_joints,
        )


def sample_scenarios(
    ranges: RandomizationRanges,
    *,
    count: int,
    seed: int,
    num_joints: int,
) -> ScenarioBatch:
    """Sample one reproducible batch using NumPy PCG64."""
    if not isinstance(ranges, RandomizationRanges):
        raise TypeError("ranges must be RandomizationRanges")
    for value, field in ((count, "count"), (num_joints, "num_joints")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.Generator(np.random.PCG64(seed))

    def uniform(field: str) -> np.ndarray:
        lower, upper = getattr(ranges, field)
        if lower == upper:
            return np.full(count, lower, dtype=np.float64)
        return rng.uniform(lower, upper, size=count)

    def normal(std: float, shape: tuple[int, ...]) -> np.ndarray:
        if std == 0.0:
            return np.zeros(shape, dtype=np.float64)
        return rng.normal(0.0, std, size=shape)

    return ScenarioBatch(
        contact_friction_scale=uniform("contact_friction_scale"),
        joint_armature_scale=uniform("joint_armature_scale"),
        joint_frictionloss_scale=uniform("joint_frictionloss_scale"),
        actuation_strength_scale=uniform("actuation_strength_scale"),
        root_linear_velocity_offset=normal(
            ranges.root_linear_velocity_std, (count, 3)
        ),
        root_angular_velocity_offset=normal(
            ranges.root_angular_velocity_std, (count, 3)
        ),
        joint_velocity_offset=normal(
            ranges.joint_velocity_std, (count, num_joints)
        ),
    )
