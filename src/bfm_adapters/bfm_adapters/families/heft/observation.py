"""Pure NumPy HEFT-PMG observation semantics."""

from __future__ import annotations

import numpy as np

ACTION_DIM = 29
QUERY_OFFSETS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
HISTORY_LAGS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
HISTORY_LENGTH = max(HISTORY_LAGS) + 1
PREVIOUS_ACTIONS = 8
CONTEXT_DIM = 1
MOTION_COMMAND_DIM = 114
TARGET_MOTION_DIM = 806
PROPRIOCEPTION_DIM = 808

_DOWN = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("HEFT received a zero-norm quaternion")
    return (values / norm).astype(np.float32, copy=False)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float32)
    return np.concatenate((values[..., :1], -values[..., 1:]), axis=-1)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"quaternion shape mismatch: {left.shape} != {right.shape}")
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quat)
    vector = np.asarray(vector, dtype=np.float32)
    q_w = quat[..., :1]
    q_vec = quat[..., 1:]
    return (
        vector * (2.0 * q_w * q_w - 1.0)
        - 2.0 * q_w * np.cross(q_vec, vector)
        + 2.0 * q_vec * np.sum(q_vec * vector, axis=-1, keepdims=True)
    ).astype(np.float32, copy=False)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3).astype(np.float32, copy=False)


def append_newest(history: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Return current-first history after appending one sample at lag zero."""

    result = history.copy()
    result[:, 1:] = history[:, :-1]
    result[:, 0] = value
    return result


def pack_motion_command(
    reference_root_position: np.ndarray,
    reference_root_orientation: np.ndarray,
    robot_root_orientation: np.ndarray,
) -> np.ndarray:
    batch, queries = reference_root_position.shape[:2]
    if queries != len(QUERY_OFFSETS):
        raise ValueError(f"HEFT requires {len(QUERY_OFFSETS)} motion_command queries")
    delta_world = reference_root_position[:, 1:] - reference_root_position[:, :1]
    anchor = np.broadcast_to(
        reference_root_orientation[:, :1], (batch, queries - 1, 4)
    )
    delta_anchor = quat_rotate_inverse(anchor, delta_world)
    robot = np.broadcast_to(robot_root_orientation[:, None], (batch, queries, 4))
    relative = quat_mul(quat_conjugate(_unit_quat(robot)), reference_root_orientation)
    # Upstream stores the first two rotation-matrix columns as two xyz rows.
    rot6d = matrix_from_quat(relative)[..., :, :2].swapaxes(-1, -2)
    return np.concatenate(
        (delta_anchor.reshape(batch, -1), rot6d.reshape(batch, -1)), axis=-1
    ).astype(np.float32, copy=False)


def pack_target_motion(
    reference_joint_position: np.ndarray,
    current_joint_position: np.ndarray,
    reference_root_position: np.ndarray,
    reference_root_orientation: np.ndarray,
) -> np.ndarray:
    batch = reference_joint_position.shape[0]
    difference = reference_joint_position - current_joint_position[:, None]
    down = np.broadcast_to(_DOWN, (*reference_root_orientation.shape[:-1], 3))
    gravity = quat_rotate_inverse(reference_root_orientation, down)
    return np.concatenate(
        (
            reference_joint_position.reshape(batch, -1),
            difference.reshape(batch, -1),
            reference_root_position[..., 2],
            gravity.reshape(batch, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def pack_proprioception(
    root_angular_velocity: np.ndarray,
    projected_gravity: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_actions: np.ndarray,
) -> np.ndarray:
    lags = np.asarray(HISTORY_LAGS, dtype=np.int64)
    batch = root_angular_velocity.shape[0]
    return np.concatenate(
        (
            root_angular_velocity[:, lags].reshape(batch, -1),
            projected_gravity[:, lags].reshape(batch, -1),
            joint_position[:, lags].reshape(batch, -1),
            joint_velocity[:, lags].reshape(batch, -1),
            previous_actions.reshape(batch, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def projected_gravity(root_orientation: np.ndarray) -> np.ndarray:
    down = np.broadcast_to(_DOWN, (*root_orientation.shape[:-1], 3))
    gravity = quat_rotate_inverse(root_orientation, down)
    norm = np.linalg.norm(gravity, axis=-1, keepdims=True)
    return (gravity / np.maximum(norm, 1.0e-8)).astype(np.float32, copy=False)


def root_angular_velocity_body(
    root_orientation: np.ndarray, angular_velocity_world: np.ndarray
) -> np.ndarray:
    return quat_rotate_inverse(root_orientation, angular_velocity_world)
