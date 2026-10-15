"""Pure NumPy observation semantics for the released TWIST2 policy."""

from __future__ import annotations

import numpy as np

ACTION_DIM = 29
MIMIC_DIM = 35
PROPRIOCEPTION_DIM = 92
FRAME_DIM = MIMIC_DIM + PROPRIOCEPTION_DIM
HISTORY_LENGTH = 10
RESET_FILL_STEPS = 2
INPUT_DIM = FRAME_DIM * (HISTORY_LENGTH + 1) + MIMIC_DIM
# Checkpoint training selected the first privileged target (t+1) for the
# current-motion slot and the declared t+0 target for the future slot.
CURRENT_OFFSET = 1
FUTURE_OFFSET = 0
QUERY_OFFSETS = (0, 1)


def _unit_quat(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("TWIST2 received a zero-norm quaternion")
    return (values / norm).astype(np.float32, copy=False)


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _unit_quat(quaternion)
    vector = np.asarray(vector, dtype=np.float32)
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * np.cross(xyz, vector)
        + 2.0 * xyz * np.sum(xyz * vector, axis=-1, keepdims=True)
    ).astype(np.float32, copy=False)


def roll_pitch(quaternion: np.ndarray) -> np.ndarray:
    """Return roll and pitch for normalized ``(w, x, y, z)`` quaternions."""

    w, x, y, z = np.moveaxis(_unit_quat(quaternion), -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.stack((roll, pitch), axis=-1).astype(np.float32, copy=False)


def pack_mimic(
    root_position_world: np.ndarray,
    root_orientation_world: np.ndarray,
    root_linear_velocity_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
    joint_position: np.ndarray,
) -> np.ndarray:
    linear_local = quat_rotate_inverse(
        root_orientation_world, root_linear_velocity_world
    )
    angular_local = quat_rotate_inverse(
        root_orientation_world, root_angular_velocity_world
    )
    result = np.concatenate(
        (
            linear_local[..., :2],
            np.asarray(root_position_world, dtype=np.float32)[..., 2:3],
            roll_pitch(root_orientation_world),
            angular_local[..., 2:3],
            np.asarray(joint_position, dtype=np.float32),
        ),
        axis=-1,
    )
    if result.shape[-1] != MIMIC_DIM:
        raise ValueError(f"TWIST2 mimic observation must contain {MIMIC_DIM} values")
    return result.astype(np.float32, copy=False)


def pack_proprioception(
    root_orientation_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    *,
    default_joint_position: np.ndarray,
    ankle_indices: np.ndarray,
    angular_velocity_scale: float,
    joint_position_scale: float,
    joint_velocity_scale: float,
) -> np.ndarray:
    velocity = np.asarray(joint_velocity, dtype=np.float32).copy()
    velocity[:, ankle_indices] = 0.0
    result = np.concatenate(
        (
            quat_rotate_inverse(
                root_orientation_world, root_angular_velocity_world
            )
            * np.float32(angular_velocity_scale),
            roll_pitch(root_orientation_world),
            (np.asarray(joint_position, dtype=np.float32)
             - default_joint_position[None])
            * np.float32(joint_position_scale),
            velocity * np.float32(joint_velocity_scale),
            np.asarray(previous_action, dtype=np.float32),
        ),
        axis=-1,
    )
    if result.shape[-1] != PROPRIOCEPTION_DIM:
        raise ValueError(
            f"TWIST2 proprioception must contain {PROPRIOCEPTION_DIM} values"
        )
    return result.astype(np.float32, copy=False)


def pack_input(
    current_mimic: np.ndarray,
    proprioception: np.ndarray,
    history: np.ndarray,
    history_fill_steps_remaining: np.ndarray,
    future_mimic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.concatenate((current_mimic, proprioception), axis=-1).astype(
        np.float32, copy=False
    )
    if history.shape != (len(current), HISTORY_LENGTH, FRAME_DIM):
        raise ValueError("TWIST2 history shape differs from the policy contract")
    if history_fill_steps_remaining.shape != (len(current),):
        raise ValueError("TWIST2 history reset counter shape differs from the batch")
    policy_input = np.concatenate(
        (current, history.reshape(len(current), -1), future_mimic), axis=-1
    ).astype(np.float32, copy=False)
    if policy_input.shape != (len(current), INPUT_DIM):
        raise ValueError("TWIST2 input shape differs from the policy contract")
    next_history = history.copy()
    next_history[:, :-1] = history[:, 1:]
    next_history[:, -1] = current
    next_history = np.where(
        (history_fill_steps_remaining <= 0)[:, None, None],
        next_history,
        current[:, None, :],
    ).astype(np.float32, copy=False)
    return policy_input, next_history
