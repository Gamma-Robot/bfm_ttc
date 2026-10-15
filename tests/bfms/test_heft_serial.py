"""HEFT-PMG serial observation, history, and command parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bfm_adapters import (
    MotionCommand,
    RobotState,
    available,
    get_family,
    load_instance_config,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "bfms" / "heft_pmg.yaml"
)
SQRT_HALF = np.float32(np.sqrt(0.5))


def _scene(config) -> tuple[RobotState, MotionCommand]:
    policy_joints = config.policy_joint_names
    state_joint_names = tuple(reversed(policy_joints))
    state_joint_position = np.asarray(
        [[policy_joints.index(name) + 1 for name in state_joint_names]], np.float32
    )
    state_joint_velocity = state_joint_position + np.float32(9.0)
    state_body_names = tuple(reversed(config.motion_command_body_names))
    bodies = len(state_body_names)
    state_orientation = np.zeros((1, bodies, 4), np.float32)
    state_orientation[..., 0] = 1.0
    root = state_body_names.index("pelvis")
    state_angular_velocity = np.zeros((1, bodies, 3), np.float32)
    state_angular_velocity[:, root] = (1.0, 2.0, 3.0)
    state = RobotState(
        joint_names=state_joint_names,
        body_names=state_body_names,
        joint_position=state_joint_position,
        joint_velocity=state_joint_velocity,
        body_position_world=np.zeros((1, bodies, 3), np.float32),
        body_orientation_world=state_orientation,
        body_linear_velocity_world=np.zeros((1, bodies, 3), np.float32),
        body_angular_velocity_world=state_angular_velocity,
    )

    offsets = config.motion_command_query_offsets
    queries = len(offsets)
    guide_joints = config.command_joint_names
    guide_bodies = tuple(reversed(config.motion_command_body_names))
    joint_position = np.empty((1, queries, len(guide_joints)), np.float32)
    for column, name in enumerate(guide_joints):
        joint_position[0, :, column] = (
            np.arange(queries, dtype=np.float32) * 100.0 + policy_joints.index(name)
        )
    body_position = np.zeros((1, queries, len(guide_bodies), 3), np.float32)
    body_orientation = np.zeros((1, queries, len(guide_bodies), 4), np.float32)
    body_orientation[..., 0] = 1.0
    root = guide_bodies.index("pelvis")
    step = np.arange(queries, dtype=np.float32)
    body_position[0, :, root] = np.stack((step, 2.0 * step, 0.5 + step), axis=-1)
    body_orientation[0, :, root] = (SQRT_HALF, 0.0, 0.0, SQRT_HALF)
    motion_command = MotionCommand(
        query_offsets=offsets,
        joint_names=guide_joints,
        body_names=guide_bodies,
        joint_position=joint_position,
        joint_velocity=np.zeros_like(joint_position),
        body_position_world=body_position,
        body_orientation_world=body_orientation,
        body_linear_velocity_world=np.zeros_like(body_position),
        body_angular_velocity_world=np.zeros_like(body_position),
    )
    return state, motion_command


def test_heft_is_registered_and_config_is_exact() -> None:
    assert "heft" in available()
    config = load_instance_config(CONFIG_PATH)
    assert config.motion_command_query_offsets == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        -1,
        -2,
        -4,
        -8,
        -12,
        -16,
    )
    assert tuple(spec.non_batch_shape for spec in config.policy_inputs) == (
        (1,),
        (114,),
        (806,),
        (808,),
    )
    other = load_instance_config(CONFIG_PATH.with_name("sonic_release_g1.yaml"))
    assert config.motion_command_body_names == other.motion_command_body_names


def test_first_observation_matches_independent_fixed_formula() -> None:
    config = load_instance_config(CONFIG_PATH)
    state, motion_command = _scene(config)
    adapter = get_family("heft")(config)
    inputs, bfm_state = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )
    np.testing.assert_array_equal(inputs["context"], [[np.float32(24.0 / 25.0)]])

    step = np.arange(1, 13, dtype=np.float32)
    expected_delta = np.stack((2.0 * step, -step, step), axis=-1).reshape(-1)
    expected_rot6d = np.tile(
        np.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], np.float32), 13
    )
    np.testing.assert_allclose(
        inputs["motion_command"][0],
        np.concatenate((expected_delta, expected_rot6d)),
        rtol=0.0,
        atol=1.0e-5,
    )

    reference = np.stack(
        [np.arange(29, dtype=np.float32) + 100.0 * query for query in range(13)]
    )
    current = np.arange(1, 30, dtype=np.float32)
    root_height = np.float32(0.5) + np.arange(13, dtype=np.float32)
    gravity = np.tile(np.asarray([0.0, 0.0, -1.0], np.float32), 13)
    expected_target = np.concatenate(
        (reference.reshape(-1), (reference - current).reshape(-1), root_height, gravity)
    )
    np.testing.assert_allclose(inputs["target_motion"][0], expected_target, atol=1.0e-6)

    expected_proprio = np.zeros(808, np.float32)
    expected_proprio[0:3] = (1.0, 2.0, 3.0)
    expected_proprio[27:30] = (0.0, 0.0, -1.0)
    expected_proprio[54:83] = current
    expected_proprio[315:344] = current + 9.0
    np.testing.assert_array_equal(inputs["proprioception"], expected_proprio[None])
    # HEFT reset semantics leave older history samples at zero.
    assert np.count_nonzero(bfm_state["joint_position"][:, 1:]) == 0


def test_boot_robot_history_and_clipped_action_history_advance_once_per_step() -> None:
    config = load_instance_config(CONFIG_PATH)
    state, motion_command = _scene(config)
    adapter = get_family("heft")(config)
    _, bfm_state = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )
    raw_action = np.linspace(-20.0, 20.0, 29, dtype=np.float32)[None]
    _, bfm_state = adapter.map_policy_outputs({"action": raw_action}, bfm_state)
    inputs, bfm_state = adapter.build_policy_inputs(state, motion_command, bfm_state)

    np.testing.assert_array_equal(inputs["context"], [[np.float32(23.0 / 25.0)]])
    proprio = inputs["proprioception"][0]
    np.testing.assert_array_equal(proprio[0:6], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    action_history = proprio[-8 * 29 :].reshape(8, 29)
    np.testing.assert_array_equal(
        action_history[0], np.clip(raw_action[0], -10.0, 10.0)
    )
    assert np.count_nonzero(action_history[1:]) == 0
    np.testing.assert_array_equal(
        bfm_state["previous_actions"][:, 0], np.clip(raw_action, -10.0, 10.0)
    )


def test_action_is_clipped_scaled_and_reordered_to_canonical_command() -> None:
    config = load_instance_config(CONFIG_PATH)
    adapter = get_family("heft")(config)
    bfm_state = adapter.initial_state(1)
    action = np.linspace(-12.0, 12.0, 29, dtype=np.float32)[None]
    command, next_state = adapter.map_policy_outputs({"action": action}, bfm_state)
    clipped = np.clip(action[0], -10.0, 10.0)
    default = np.asarray(config.family_config["default_joint_position"], np.float32)
    scale = np.asarray(config.family_config["action_scale"], np.float32)
    policy_target = default + clipped * scale
    order = [
        config.policy_joint_names.index(name)
        for name in config.command_joint_names
    ]
    np.testing.assert_array_equal(command.position_target[0], policy_target[order])
    np.testing.assert_array_equal(next_state["last_action"][0], clipped)
    np.testing.assert_array_equal(
        bfm_state["last_action"], np.zeros((1, 29), np.float32)
    )
