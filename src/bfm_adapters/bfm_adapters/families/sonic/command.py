"""Released SONIC action to canonical G1 PD command."""

from __future__ import annotations

import numpy as np

from bfm_adapters.contracts import PDCommand
from bfm_adapters.families.sonic.observation import ACTION_DIM


def action_to_pd(
    action: np.ndarray,
    *,
    policy_joint_names: tuple[str, ...],
    command_joint_names: tuple[str, ...],
    default_joint_position: np.ndarray,
    action_scale: np.ndarray,
    proportional_gain: np.ndarray,
    derivative_gain: np.ndarray,
) -> PDCommand:
    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(
            f"action must have shape (B, {ACTION_DIM}), got {values.shape}"
        )
    order = [policy_joint_names.index(name) for name in command_joint_names]
    position_policy = default_joint_position[None] + values * action_scale[None]
    position = position_policy[:, order].astype(np.float32, copy=False)
    batch = len(values)
    zeros = np.zeros_like(position)
    return PDCommand(
        joint_names=command_joint_names,
        position_target=position,
        velocity_target=zeros,
        feedforward_torque=zeros,
        proportional_gain=np.broadcast_to(
            proportional_gain[order], (batch, ACTION_DIM)
        ).astype(np.float32, copy=True),
        derivative_gain=np.broadcast_to(
            derivative_gain[order], (batch, ACTION_DIM)
        ).astype(np.float32, copy=True),
    )
