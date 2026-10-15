"""Load the supported Any4HDMI reference_motion-target format."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from bfm_adapters.contracts import MotionCommand
from command_optimization.motion_tracking.search_space import KinematicProjector

_FORMAT_VERSION = 2
ROOT_QPOS_NAMES = (
    "root_tx",
    "root_ty",
    "root_tz",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
)
_ARRAY_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)
_MOTION_ID_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._()-]*")


def validate_any4hdmi_motion_id(motion_id: str) -> str:
    """Validate a relative motion ID without normalizing its source spelling."""

    if not isinstance(motion_id, str) or any(
        _MOTION_ID_COMPONENT.fullmatch(part) is None
        for part in motion_id.split("/")
    ):
        raise ValueError(f"invalid motion_id {motion_id!r}")
    return motion_id


def resolve_any4hdmi_motion(
    dataset_root: str | Path, motion_id: str
) -> Path:
    """Resolve one flat or nested Any4HDMI motion without escaping its tree."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Any4HDMI dataset root does not exist: {root}")
    validate_any4hdmi_motion_id(motion_id)

    candidates = (
        root / "motions" / f"{motion_id}.npz",
        root / "motions" / motion_id / "motion.npz",
    )
    matches: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Any4HDMI motion path escapes dataset root: {candidate}"
            )
        if candidate.exists():
            if not resolved.is_file():
                raise ValueError(
                    f"Any4HDMI motion path is not a regular file: {candidate}"
                )
            matches.append(resolved)
    if not matches:
        raise FileNotFoundError(
            f"Any4HDMI motion {motion_id!r} does not exist under {root / 'motions'}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Any4HDMI motion {motion_id!r} is ambiguous: both flat and nested layouts exist"
        )
    return matches[0]


def load_reference_motion(
    dataset_root: str | Path,
    motion_id: str,
    *,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    control_period: float,
    projector: KinematicProjector,
    clip_duration_seconds: float | None = None,
) -> MotionCommand:
    """Return one immutable, batch-1 target on the bfm Control Step grid.

    ``dataset_root`` may be a local user dataset or a verified artifact tree.
    This reader deliberately supports only the Any4HDMI v2
    ``manifest.json`` plus qpos-only ``motion.npz`` representation.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Any4HDMI dataset root does not exist: {root}")
    motion_path = resolve_any4hdmi_motion(root, motion_id)
    joints = _validate_names(joint_names, "joint_names")
    bodies = _validate_names(body_names, "body_names")
    period = float(control_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("control_period must be finite and positive")
    if not callable(getattr(projector, "project", None)):
        raise TypeError("projector must implement project(...)")

    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Any4HDMI manifest does not exist: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Any4HDMI manifest must be a JSON object")
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError("Any4HDMI format_version must be 2")
    if manifest.get("motions_subdir") != "motions":
        raise ValueError("Any4HDMI motions_subdir must be 'motions'")
    for field in ("dataset_name", "mjcf"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Any4HDMI {field} must be a non-empty string")

    source_period = _positive_number(manifest.get("timestep"), "timestep")
    raw_names = manifest.get("qpos_names")
    if not isinstance(raw_names, list) or any(
        not isinstance(name, str) or not name for name in raw_names
    ):
        raise ValueError("Any4HDMI qpos_names must be a list of non-empty strings")
    qpos_names = tuple(raw_names)
    if len(set(qpos_names)) != len(qpos_names):
        raise ValueError("Any4HDMI qpos_names must be unique")
    if qpos_names[:7] != ROOT_QPOS_NAMES:
        raise ValueError("Any4HDMI qpos_names must begin with canonical free-root names")
    source_joints = qpos_names[7:]
    if set(source_joints) != set(joints) or len(source_joints) != len(joints):
        raise ValueError("Any4HDMI joints do not match requested canonical joint_names")
    qpos_dim = manifest.get("qpos_dim")
    if isinstance(qpos_dim, bool) or not isinstance(qpos_dim, int):
        raise ValueError("Any4HDMI qpos_dim must be an integer")
    if qpos_dim != len(qpos_names):
        raise ValueError("Any4HDMI qpos_dim does not match qpos_names")

    try:
        with np.load(motion_path, allow_pickle=False) as payload:
            if payload.files != ["qpos"]:
                raise ValueError("Any4HDMI motion file must contain only 'qpos'")
            qpos = np.asarray(payload["qpos"], dtype=np.float64)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Any4HDMI motion does not exist: {motion_path}") from exc
    expected_dim = 7 + len(joints)
    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != expected_dim:
        raise ValueError(
            f"Any4HDMI qpos must have shape (T, {expected_dim}) with T >= 2, "
            f"got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("Any4HDMI qpos must contain only finite values")
    quaternion = qpos[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Any4HDMI root quaternions must be normalized")
    quaternion = quaternion / norms[:, None]

    source_times = np.arange(qpos.shape[0], dtype=np.float64) * source_period
    duration = source_times[-1]
    if clip_duration_seconds is not None:
        requested = float(clip_duration_seconds)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("clip_duration_seconds must be finite and positive")
        if requested > duration + 1e-9:
            raise ValueError(
                f"clip_duration_seconds exceeds the source duration {duration}"
            )
        requested_steps = round(requested / period)
        if requested_steps < 1 or not math.isclose(
            requested, requested_steps * period, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                "clip_duration_seconds must align with the control_period grid"
            )
        target_count = requested_steps + 1
    else:
        target_count = max(1, int(math.floor(duration / period + 1e-9)) + 1)
    target_times = np.arange(target_count, dtype=np.float64) * period
    root_position = _linear_resample(source_times, target_times, qpos[:, :3])
    root_orientation = _slerp(source_times, target_times, quaternion)
    source_indices = np.asarray([source_joints.index(name) for name in joints]) + 7
    joint_position = _linear_resample(
        source_times, target_times, qpos[:, source_indices]
    )
    offsets = tuple(range(target_count))
    motion_command = projector.project(
        query_offsets=offsets,
        joint_names=joints,
        body_names=bodies,
        root_position_world=root_position[None],
        root_orientation_world=root_orientation[None],
        joint_position=joint_position[None],
        control_period=period,
    )
    if not isinstance(motion_command, MotionCommand):
        raise TypeError("projector must return MotionCommand")
    if (
        motion_command.query_offsets != offsets
        or motion_command.joint_names != joints
        or motion_command.body_names != bodies
        or motion_command.joint_position.shape[0] != 1
    ):
        raise ValueError("projector returned motion_command with incompatible metadata")
    return _immutable_motion_command(motion_command)


def _validate_names(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must be unique")
    return names


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Any4HDMI {field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Any4HDMI {field} must be finite and positive")
    return result


def _linear_resample(
    source_times: np.ndarray, target_times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            np.interp(target_times, source_times, values[:, index])
            for index in range(values.shape[1])
        ],
        axis=-1,
    )


def _slerp(
    source_times: np.ndarray, target_times: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    alpha = (
        (target_times - source_times[left])
        / (source_times[right] - source_times[left])
    )[:, None]
    q0, q1 = quaternion[left], quaternion[right]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    denominator = np.sin(angle)
    safe = denominator > 1e-8
    safe_denominator = np.where(safe, denominator, 1.0)
    scale0 = np.sin((1.0 - alpha) * angle) / safe_denominator
    scale1 = np.sin(alpha * angle) / safe_denominator
    scale0 = np.where(safe, scale0, 1.0 - alpha)
    scale1 = np.where(safe, scale1, alpha)
    result = scale0 * q0 + scale1 * q1
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _immutable_motion_command(motion_command: MotionCommand) -> MotionCommand:
    arrays: dict[str, np.ndarray] = {}
    for field in _ARRAY_FIELDS:
        value = np.array(
            getattr(motion_command, field), dtype=np.float64, order="C", copy=True
        )
        if not np.isfinite(value).all():
            raise ValueError(f"projector returned non-finite {field}")
        value.setflags(write=False)
        arrays[field] = value
    return MotionCommand(
        query_offsets=motion_command.query_offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **arrays,
    )
