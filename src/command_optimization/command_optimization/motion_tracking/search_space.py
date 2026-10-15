"""Motion-tracking search space: normalized qpos knots to ``MotionCommand``.

Optimizer candidates are normalized float64 latents of shape ``(N, K, D)`` with
``D = 6 + J``, laid out per knot as ``[dx, dy, dz, rotvec_x, rotvec_y,
rotvec_z, joint deltas...]``. Decoding clips latents to ``[-1, 1]``, scales
them to physical knot deltas, and interpolates over the dense Control Step grid;
interpolation is linear (a degree-1 spline) in the current interface.

The search space owns latent layout and interpolation only. An injected
``KinematicProjector`` maps each dense root/joint trajectory to complete
``MotionCommand`` using the same G1 model as rollout physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bfm_adapters.contracts import MotionCommand

_SMALL_ANGLE = 1e-8


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of ``(w, x, y, z)`` quaternion arrays."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector to ``(w, x, y, z)`` quaternion, series branch near zero."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    safe = np.where(angle < _SMALL_ANGLE, 1.0, angle)
    scale = np.where(
        angle < _SMALL_ANGLE,
        0.5 - angle * angle / 48.0,
        np.sin(0.5 * angle) / safe,
    )
    return np.concatenate((np.cos(0.5 * angle), rotvec * scale), axis=-1)


def _canonical_unit_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize and flip sign so that ``w >= 0``."""
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


@dataclass(frozen=True)
class KnotScales:
    """Physical magnitude of one normalized latent unit, per dimension group."""

    root_xy_m: float
    root_z_m: float
    root_rotation_rad: float
    joint_position_rad: float

    def __post_init__(self) -> None:
        for name in ("root_xy_m", "root_z_m", "root_rotation_rad", "joint_position_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    def scale_vector(self, num_joints: int) -> np.ndarray:
        """Scale per latent dimension: [dx, dy, dz, rotvec (3), joint deltas (J)]."""
        return np.concatenate(
            (
                (self.root_xy_m, self.root_xy_m, self.root_z_m),
                np.full(3, self.root_rotation_rad),
                np.full(num_joints, self.joint_position_rad),
            )
        )


class KinematicProjector(Protocol):
    """Application-owned root/joint trajectory → complete motion_command seam."""

    def project(
        self,
        *,
        query_offsets: tuple[int, ...],
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        root_position_world: np.ndarray,
        root_orientation_world: np.ndarray,
        joint_position: np.ndarray,
        control_period: float,
    ) -> MotionCommand: ...


class QposKnotsSpace:
    """Decode normalized ``(N, K, D)`` knot latents into batched motion_command.

    The reference is a dense per-Control-Step target segment (batch 1,
    ``query_offsets == (0, ..., H-1)``); K knots sit at uniformly spaced
    float step positions spanning ``0..H-1``. ``decode`` is a pure function
    of the latents and constructor state.
    """

    def __init__(
        self,
        reference: MotionCommand,
        num_knots: int,
        control_period: float,
        scales: KnotScales,
        projector: KinematicProjector,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
        pelvis_body: str = "pelvis",
    ) -> None:
        horizon = len(reference.query_offsets)
        if reference.joint_position.shape[0] != 1:
            raise ValueError("reference must have batch size 1")
        if horizon < 2 or reference.query_offsets != tuple(range(horizon)):
            raise ValueError("reference query_offsets must be (0, 1, ..., H-1) with H >= 2")
        num_knots = int(num_knots)
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        control_period = float(control_period)
        if not np.isfinite(control_period) or control_period <= 0.0:
            raise ValueError("control_period must be finite and > 0")
        if pelvis_body not in reference.body_names:
            raise ValueError(f"pelvis_body {pelvis_body!r} not in reference body_names")
        if not callable(getattr(projector, "project", None)):
            raise TypeError("projector must implement project(...)")
        if (joint_lower is None) != (joint_upper is None):
            raise ValueError("joint_lower and joint_upper must be provided together")

        joints = len(reference.joint_names)
        self._joint_lower = self._joint_upper = None
        if joint_lower is not None:
            lower = np.array(joint_lower, dtype=np.float64)
            upper = np.array(joint_upper, dtype=np.float64)
            if lower.shape != (joints,) or upper.shape != (joints,):
                raise ValueError(f"joint limits must have shape ({joints},)")
            if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
                raise ValueError("joint limits must be finite")
            if not (lower < upper).all():
                raise ValueError("joint_lower must be < joint_upper elementwise")
            if ((reference.joint_position < lower) | (reference.joint_position > upper)).any():
                raise ValueError("reference joint targets violate the joint limits")
            self._joint_lower, self._joint_upper = lower, upper

        self._reference = MotionCommand(
            query_offsets=reference.query_offsets,
            joint_names=reference.joint_names,
            body_names=reference.body_names,
            joint_position=np.array(reference.joint_position, dtype=np.float64),
            joint_velocity=np.array(reference.joint_velocity, dtype=np.float64),
            body_position_world=np.array(reference.body_position_world, dtype=np.float64),
            body_orientation_world=np.array(
                reference.body_orientation_world, dtype=np.float64
            ),
            body_linear_velocity_world=np.array(
                reference.body_linear_velocity_world, dtype=np.float64
            ),
            body_angular_velocity_world=np.array(
                reference.body_angular_velocity_world, dtype=np.float64
            ),
        )
        self._pelvis_index = reference.body_names.index(pelvis_body)
        self._control_period = control_period
        self._projector = projector
        self._scale = scales.scale_vector(joints)
        self._weights = self._interpolation_weights(horizon, num_knots)

    @staticmethod
    def _interpolation_weights(horizon: int, num_knots: int) -> np.ndarray:
        """Linear-interpolation matrix ``(H, K)``; one-hot rows at knot steps."""
        positions = np.linspace(0.0, horizon - 1.0, num_knots)
        steps = np.arange(horizon, dtype=np.float64)
        segment = np.clip(
            np.searchsorted(positions, steps, side="right") - 1, 0, num_knots - 2
        )
        fraction = (steps - positions[segment]) / (
            positions[segment + 1] - positions[segment]
        )
        weights = np.zeros((horizon, num_knots))
        rows = np.arange(horizon)
        weights[rows, segment] = 1.0 - fraction
        weights[rows, segment + 1] = fraction
        return weights

    @property
    def latent_shape(self) -> tuple[int, int]:
        return (self._weights.shape[1], self._scale.shape[0])

    def zero(self) -> np.ndarray:
        """Null latent: ``decode`` maps it to the unmodified reference."""
        return np.zeros(self.latent_shape)

    def project(self, latent: np.ndarray) -> np.ndarray:
        """Return the canonical representative of one normalized latent."""
        value = np.asarray(latent, dtype=np.float64)
        if value.shape != self.latent_shape or not np.isfinite(value).all():
            raise ValueError(
                f"latent must be finite with shape {self.latent_shape}, got {value.shape}"
            )
        return np.clip(value, -1.0, 1.0)

    def shift(self, latent: np.ndarray, steps: int) -> np.ndarray:
        """Warm-start the next update by advancing a knot deformation in time."""
        value = self.project(latent)
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        horizon, knots = self._weights.shape
        if steps >= horizon:
            return np.zeros_like(value)
        positions = np.linspace(0.0, horizon - 1.0, knots)
        shifted = np.stack(
            [
                np.interp(
                    positions + int(steps),
                    positions,
                    value[:, dimension],
                    right=0.0,
                )
                for dimension in range(value.shape[1])
            ],
            axis=-1,
        )
        return np.clip(shifted, -1.0, 1.0)

    def decode(self, latents: np.ndarray) -> MotionCommand:
        """Decode ``(N, K, D)`` normalized latents into batch-``N`` motion_command."""
        latents = np.asarray(latents, dtype=np.float64)
        if latents.ndim != 3 or latents.shape[1:] != self.latent_shape:
            raise ValueError(
                f"latents must have shape (N, *{self.latent_shape}), got {latents.shape}"
            )
        knots = np.clip(latents, -1.0, 1.0) * self._scale
        dense = np.einsum("hk,nkd->nhd", self._weights, knots)
        root_delta = dense[:, :, :3]
        rotvec = dense[:, :, 3:6]
        joint_delta = dense[:, :, 6:]
        reference = self._reference

        joint_position = reference.joint_position + joint_delta
        if self._joint_lower is not None:
            joint_position = np.clip(joint_position, self._joint_lower, self._joint_upper)

        batch = latents.shape[0]
        pelvis = self._pelvis_index
        root_position = (
            np.repeat(
                reference.body_position_world[:, :, pelvis], batch, axis=0
            )
            + root_delta
        )
        root_orientation = _canonical_unit_quat(
            _quat_multiply(
                _rotvec_to_quat(rotvec),
                reference.body_orientation_world[:, :, pelvis],
            )
        )
        motion_command = self._projector.project(
            query_offsets=reference.query_offsets,
            joint_names=reference.joint_names,
            body_names=reference.body_names,
            root_position_world=root_position,
            root_orientation_world=root_orientation,
            joint_position=joint_position,
            control_period=self._control_period,
        )
        if (
            motion_command.query_offsets != reference.query_offsets
            or motion_command.joint_names != reference.joint_names
            or motion_command.body_names != reference.body_names
            or motion_command.joint_position.shape[0] != batch
        ):
            raise ValueError("projector returned motion_command with incompatible metadata")
        for field in (
            "joint_position",
            "joint_velocity",
            "body_position_world",
            "body_orientation_world",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
        ):
            if not np.isfinite(getattr(motion_command, field)).all():
                raise ValueError(f"projector returned non-finite {field}")
        return motion_command
