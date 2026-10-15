"""Conformance contribution for the released native-G1 SONIC family."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bfm_adapters import (
    MotionCommand,
    NamedTensors,
    RobotState,
    BFMInstanceConfig,
    load_instance_config,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "bfms"
    / "sonic_release_g1.yaml"
)
_POLICY_ARTIFACT = "sonic-release-g1-policy.batched"


def instance_config() -> BFMInstanceConfig:
    return load_instance_config(_CONFIG_PATH)


def make_robot_state(
    config: BFMInstanceConfig, batch_size: int, seed: int
) -> RobotState:
    rng = np.random.default_rng(seed)
    joint_names = tuple(reversed(config.policy_joint_names))
    body_names = tuple(reversed(config.motion_command_body_names))

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, *shape)).astype(np.float32)

    quaternions = draw(len(body_names), 4)
    quaternions[..., 0] = np.abs(quaternions[..., 0]) + 1.0
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return RobotState(
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=quaternions.astype(np.float32),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def make_motion_command(
    config: BFMInstanceConfig, batch_size: int, seed: int
) -> MotionCommand:
    rng = np.random.default_rng(seed)
    offsets = config.motion_command_query_offsets
    joint_names = config.command_joint_names
    body_names = tuple(reversed(config.motion_command_body_names))

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, len(offsets), *shape)).astype(
            np.float32
        )

    quaternions = draw(len(body_names), 4)
    quaternions[..., 0] = np.abs(quaternions[..., 0]) + 1.0
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return MotionCommand(
        query_offsets=offsets,
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=quaternions.astype(np.float32),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def fake_policy(config: BFMInstanceConfig, inputs: NamedTensors) -> NamedTensors:
    signal = sum(
        np.asarray(value, dtype=np.float64).reshape(len(value), -1).mean(axis=1)
        for value in inputs.values()
    )
    grid = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
    return {"action": np.tanh(signal[:, None] + grid).astype(np.float32)}


def conformance_case() -> dict[str, object]:
    config = instance_config()
    return {
        "config": config,
        "make_robot_state": make_robot_state,
        "make_motion_command": make_motion_command,
        "fake_policy": fake_policy,
        "artifact_io": {
            _POLICY_ARTIFACT: {
                "inputs": config.policy_inputs,
                "outputs": config.policy_outputs,
            }
        },
    }
