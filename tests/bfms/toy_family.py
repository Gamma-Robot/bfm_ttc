"""Test-only "toy" bfm family proving the locality invariant closes.

Registered purely from test code; only this module may know the internal
structure of the toy bfm state.
"""

from __future__ import annotations

import numpy as np

from bfm_adapters import (
    MotionCommand,
    NamedTensors,
    PDCommand,
    RobotState,
    TensorSpec,
    BFMInstanceConfig,
    available,
    register,
)

FAMILY = "toy"
_HISTORY = 4
_JOINTS = ("hip", "knee", "ankle")
_COMMAND_JOINTS = ("ankle", "hip", "knee")
_BODIES = ("pelvis", "torso")
_OBS_DIM = 3 * len(_JOINTS) + 3 * len(_BODIES)
# Fake artifact identity and the I/O layout that identity claims to expose.
_POLICY_ARTIFACT = "sha256:toy"
_POLICY_INPUT_SPECS = (
    TensorSpec("observation", "float32", (_OBS_DIM,)),
    TensorSpec("history", "float32", (_HISTORY * _OBS_DIM,)),
)
_POLICY_OUTPUT_SPECS = (TensorSpec("action", "float32", (len(_JOINTS),)),)
_ARTIFACT_IO = {
    _POLICY_ARTIFACT: {
        "inputs": _POLICY_INPUT_SPECS,
        "outputs": _POLICY_OUTPUT_SPECS,
    }
}


def _column_order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    return [names.index(name) for name in target]


class ToyAdapter:
    """Fake-policy bfm with a rolling observation-history state."""

    def __init__(self, config: BFMInstanceConfig) -> None:
        self._config = config
        self._obs_dim = 3 * len(config.policy_joint_names) + 3 * len(
            config.motion_command_body_names
        )
        shapes = {spec.name: spec.non_batch_shape for spec in config.policy_inputs}
        expected = {
            "observation": (self._obs_dim,),
            "history": (_HISTORY * self._obs_dim,),
        }
        if shapes != expected:
            raise ValueError(f"toy policy_inputs must be {expected}, got {shapes}")
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        return {
            "history": np.zeros(
                (batch_size, _HISTORY, self._obs_dim), dtype=np.float32
            ),
            "step": np.zeros(batch_size, dtype=np.int64),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError("toy expand_state requires one lane and positive batch_size")
        return {
            name: np.repeat(value, batch_size, axis=0)
            for name, value in state.items()
        }

    def build_policy_inputs(
        self,
        state: RobotState,
        motion_command: MotionCommand,
        bfm_state: dict[str, np.ndarray],
    ) -> tuple[NamedTensors, dict[str, np.ndarray]]:
        joints = _column_order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _column_order(
            motion_command.joint_names, self._config.policy_joint_names
        )
        bodies = _column_order(motion_command.body_names, self._config.motion_command_body_names)
        batch = state.joint_position.shape[0]
        observation = np.concatenate(
            [
                state.joint_position[:, joints],
                state.joint_velocity[:, joints],
                motion_command.joint_position[:, 0, :][:, guide_joints],
                motion_command.body_position_world[:, 0, bodies].reshape(batch, -1),
            ],
            axis=1,
            dtype=np.float32,
        )
        history = np.concatenate(
            [bfm_state["history"][:, 1:], observation[:, None]], axis=1
        )
        inputs = {
            "observation": observation.astype(self._input_dtypes["observation"]),
            "history": history.reshape(batch, -1).astype(
                self._input_dtypes["history"]
            ),
        }
        return inputs, {"history": history, "step": bfm_state["step"].copy()}

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        bfm_state: dict[str, np.ndarray],
    ) -> tuple[PDCommand, dict[str, np.ndarray]]:
        order = _column_order(
            self._config.policy_joint_names, self._config.command_joint_names
        )
        action = np.asarray(outputs["action"], dtype=np.float32)[:, order]
        kp = np.tile(
            50.0 + 10.0 * np.arange(action.shape[1], dtype=np.float32),
            (action.shape[0], 1),
        )
        command = PDCommand(
            joint_names=self._config.command_joint_names,
            position_target=action,
            velocity_target=np.zeros_like(action),
            feedforward_torque=np.zeros_like(action),
            proportional_gain=kp,
            derivative_gain=0.1 * kp,
        )
        next_state = {
            "history": bfm_state["history"].copy(),
            "step": bfm_state["step"] + 1,
        }
        return command, next_state


if FAMILY not in available():
    register(FAMILY)(ToyAdapter)


def instance_config() -> BFMInstanceConfig:
    return BFMInstanceConfig(
        family=FAMILY,
        instance="toy_default",
        artifacts={"policy": _POLICY_ARTIFACT},
        control_period=0.02,
        motion_command_query_offsets=(0, 5),
        policy_inputs=_POLICY_INPUT_SPECS,
        policy_outputs=_POLICY_OUTPUT_SPECS,
        policy_joint_names=_JOINTS,
        command_joint_names=_COMMAND_JOINTS,
        motion_command_body_names=_BODIES,
        family_config={},
    )


def make_robot_state(
    config: BFMInstanceConfig, batch_size: int, seed: int
) -> RobotState:
    """Synthetic state whose joint order differs from the policy order."""

    rng = np.random.default_rng(seed)
    joint_names = tuple(reversed(config.policy_joint_names))
    body_names = config.motion_command_body_names

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, *shape)).astype(np.float32)

    return RobotState(
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=draw(len(body_names), 4),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def make_motion_command(
    config: BFMInstanceConfig, batch_size: int, seed: int
) -> MotionCommand:
    """Synthetic motion_command whose name orders differ from the config orders."""

    rng = np.random.default_rng(seed)
    offsets = config.motion_command_query_offsets
    joint_names = config.command_joint_names
    body_names = tuple(reversed(config.motion_command_body_names))

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, len(offsets), *shape)).astype(
            np.float32
        )

    return MotionCommand(
        query_offsets=offsets,
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=draw(len(body_names), 4),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def fake_policy(config: BFMInstanceConfig, inputs: NamedTensors) -> NamedTensors:
    """Deterministic pure-numpy stand-in for policy inference."""

    signal = sum(
        np.asarray(value, dtype=np.float64).reshape(len(value), -1).mean(axis=1)
        for value in inputs.values()
    )
    outputs = {}
    for spec in config.policy_outputs:
        size = int(np.prod(spec.non_batch_shape))
        grid = np.linspace(-1.0, 1.0, size).reshape(spec.non_batch_shape)
        lanes = signal.reshape((-1,) + (1,) * grid.ndim)
        outputs[spec.name] = np.tanh(lanes + grid).astype(spec.dtype)
    return outputs


def conformance_case() -> dict[str, object]:
    """Case contribution consumed by test_conformance.cases()."""

    return {
        "config": instance_config(),
        "make_robot_state": make_robot_state,
        "make_motion_command": make_motion_command,
        "fake_policy": fake_policy,
        "artifact_io": _ARTIFACT_IO,
    }
