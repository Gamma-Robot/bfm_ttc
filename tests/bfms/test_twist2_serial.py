"""TWIST2 serial observation, history, and command semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bfm_adapters import MotionCommand, RobotState, available, get_family, load_instance_config
from bfm_adapters.families.twist2.observation import FRAME_DIM, HISTORY_LENGTH

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "bfms"
    / "twist2_1017_20k.yaml"
)


def _scene(config) -> tuple[RobotState, MotionCommand]:
    policy_joints = config.policy_joint_names
    state_names = tuple(reversed(policy_joints))
    position_by_policy = np.arange(1, 30, dtype=np.float32)
    velocity_by_policy = np.arange(101, 130, dtype=np.float32)
    state = RobotState(
        joint_names=state_names,
        body_names=("pelvis",),
        joint_position=position_by_policy[::-1][None],
        joint_velocity=velocity_by_policy[::-1][None],
        body_position_world=np.zeros((1, 1, 3), np.float32),
        body_orientation_world=np.asarray([[[1.0, 0.0, 0.0, 0.0]]], np.float32),
        body_linear_velocity_world=np.zeros((1, 1, 3), np.float32),
        body_angular_velocity_world=np.asarray([[[4.0, 8.0, 12.0]]], np.float32),
    )
    current_joint = np.arange(29, dtype=np.float32)
    future_joint = current_joint + np.float32(50.0)
    reference_joint = np.stack((future_joint, current_joint))[None]
    motion_command = MotionCommand(
        query_offsets=(0, 1),
        joint_names=policy_joints,
        body_names=("pelvis",),
        joint_position=reference_joint,
        joint_velocity=np.zeros_like(reference_joint),
        body_position_world=np.asarray(
            [[[[0.0, 0.0, 1.8]], [[0.0, 0.0, 0.8]]]], np.float32
        ),
        body_orientation_world=np.asarray(
            [[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]], np.float32
        ),
        body_linear_velocity_world=np.asarray(
            [[[[7.0, 8.0, 9.0]], [[1.0, 2.0, 3.0]]]], np.float32
        ),
        body_angular_velocity_world=np.asarray(
            [[[[10.0, 11.0, 12.0]], [[4.0, 5.0, 6.0]]]], np.float32
        ),
    )
    return state, motion_command


def test_twist2_is_registered_and_declares_official_flat_onnx_abi() -> None:
    assert "twist2" in available()
    config = load_instance_config(CONFIG_PATH)
    assert config.artifacts == {"policy": "twist2-1017-20k-policy"}
    assert config.control_period == 0.02
    assert config.allow_cpu_ep_fallback
    assert config.motion_command_query_offsets == (0, 1)
    assert [(spec.name, spec.non_batch_shape) for spec in config.policy_inputs] == [
        ("input", (1432,))
    ]
    assert [(spec.name, spec.non_batch_shape) for spec in config.policy_outputs] == [
        ("output", (29,))
    ]


def test_first_observation_matches_checkpoint_training_layout() -> None:
    config = load_instance_config(CONFIG_PATH)
    state, motion_command = _scene(config)
    adapter = get_family("twist2")(config)
    inputs, bfm_state = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )
    value = inputs["input"][0]
    current_mimic = np.concatenate(
        ([1.0, 2.0, 0.8, 0.0, 0.0, 6.0], np.arange(29, dtype=np.float32))
    ).astype(np.float32)
    default = np.asarray(config.family_config["default_joint_position"], np.float32)
    joint_velocity = np.arange(101, 130, dtype=np.float32)
    joint_velocity[[4, 5, 10, 11]] = 0.0
    proprioception = np.concatenate(
        (
            [1.0, 2.0, 3.0, 0.0, 0.0],
            np.arange(1, 30, dtype=np.float32) - default,
            joint_velocity * np.float32(0.05),
            np.zeros(29, np.float32),
        )
    ).astype(np.float32)
    current = np.concatenate((current_mimic, proprioception))

    np.testing.assert_allclose(value[:FRAME_DIM], current, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(
        value[FRAME_DIM : FRAME_DIM * (HISTORY_LENGTH + 1)],
        np.zeros(HISTORY_LENGTH * FRAME_DIM, np.float32),
    )
    future_mimic = np.concatenate(
        ([7.0, 8.0, 1.8, 0.0, 0.0, 12.0], np.arange(29, dtype=np.float32) + 50.0)
    ).astype(np.float32)
    np.testing.assert_allclose(value[-35:], future_mimic, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(
        bfm_state["history"][0],
        np.broadcast_to(current, (HISTORY_LENGTH, FRAME_DIM)),
    )


def test_history_uses_pre_step_frames_and_previous_action_uses_raw_output() -> None:
    config = load_instance_config(CONFIG_PATH)
    state, motion_command = _scene(config)
    adapter = get_family("twist2")(config)
    first, observed = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )
    action = np.linspace(-12.0, 12.0, 29, dtype=np.float32)[None]
    _, advanced = adapter.map_policy_outputs({"output": action}, observed)
    second, second_state = adapter.build_policy_inputs(state, motion_command, advanced)

    first_frame = first["input"][0, :FRAME_DIM]
    history = second["input"][0, FRAME_DIM : -35].reshape(HISTORY_LENGTH, FRAME_DIM)
    np.testing.assert_array_equal(
        history, np.broadcast_to(first_frame, (HISTORY_LENGTH, FRAME_DIM))
    )
    np.testing.assert_array_equal(second["input"][0, 98:127], action[0])
    second_frame = second["input"][0, :FRAME_DIM]
    np.testing.assert_array_equal(
        second_state["history"][0],
        np.broadcast_to(second_frame, (HISTORY_LENGTH, FRAME_DIM)),
    )
    assert second_state["history_fill_steps_remaining"].item() == 0


def test_action_is_clipped_scaled_and_mapped_to_pd() -> None:
    config = load_instance_config(CONFIG_PATH)
    adapter = get_family("twist2")(config)
    state = adapter.initial_state(1)
    action = np.linspace(-12.0, 12.0, 29, dtype=np.float32)[None]
    command, next_state = adapter.map_policy_outputs({"output": action}, state)
    expected = np.asarray(config.family_config["default_joint_position"], np.float32)
    expected += np.clip(action[0], -10.0, 10.0) * np.float32(0.5)
    np.testing.assert_array_equal(command.position_target[0], expected)
    np.testing.assert_array_equal(next_state["last_action"], action)
    np.testing.assert_array_equal(state["last_action"], np.zeros((1, 29), np.float32))
