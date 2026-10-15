"""Released G1 SONIC serial observation, state, and command parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bfm_adapters import (
    DeviceBFMProvider,
    MotionCommand,
    RobotState,
    available,
    get_family,
    load_instance_config,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "bfms"
    / "sonic_release_g1.yaml"
)


def test_sonic_adapter_advertises_optional_device_execution() -> None:
    adapter = get_family("sonic")(load_instance_config(CONFIG_PATH))

    assert isinstance(adapter, DeviceBFMProvider)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], np.float32
    )


def _scene(
    config,
    *,
    batch_size: int = 1,
    robot_yaw: float = np.pi / 2.0,
    joint_shift: float = 0.0,
    velocity_shift: float = 0.0,
    angular_velocity: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> tuple[RobotState, MotionCommand]:
    policy_joints = config.policy_joint_names
    state_joint_names = tuple(reversed(policy_joints))
    positions = np.asarray(
        [policy_joints.index(name) + 1.0 + joint_shift for name in state_joint_names],
        np.float32,
    )
    velocities = np.asarray(
        [
            policy_joints.index(name) + 51.0 + velocity_shift
            for name in state_joint_names
        ],
        np.float32,
    )
    state_body_names = tuple(reversed(config.motion_command_body_names))
    bodies = len(state_body_names)
    state_orientation = np.zeros((batch_size, bodies, 4), np.float32)
    state_orientation[..., 0] = 1.0
    root = state_body_names.index("pelvis")
    state_orientation[:, root] = _yaw_quat(robot_yaw)
    state_angular_velocity = np.zeros((batch_size, bodies, 3), np.float32)
    state_angular_velocity[:, root] = angular_velocity
    state = RobotState(
        joint_names=state_joint_names,
        body_names=state_body_names,
        joint_position=np.broadcast_to(positions, (batch_size, 29)).copy(),
        joint_velocity=np.broadcast_to(velocities, (batch_size, 29)).copy(),
        body_position_world=np.zeros((batch_size, bodies, 3), np.float32),
        body_orientation_world=state_orientation,
        body_linear_velocity_world=np.zeros((batch_size, bodies, 3), np.float32),
        body_angular_velocity_world=state_angular_velocity,
    )

    offsets = config.motion_command_query_offsets
    queries = len(offsets)
    guide_joints = config.command_joint_names
    guide_bodies = tuple(reversed(config.motion_command_body_names))
    joint_position = np.empty((batch_size, queries, len(guide_joints)), np.float32)
    joint_velocity = np.empty_like(joint_position)
    for column, name in enumerate(guide_joints):
        policy_index = policy_joints.index(name)
        joint_position[:, :, column] = (
            100.0 * np.arange(queries, dtype=np.float32) + policy_index
        )
        joint_velocity[:, :, column] = (
            1000.0 * np.arange(queries, dtype=np.float32) + policy_index + 0.5
        )
    body_position = np.zeros((batch_size, queries, len(guide_bodies), 3), np.float32)
    body_orientation = np.zeros(
        (batch_size, queries, len(guide_bodies), 4), np.float32
    )
    body_orientation[..., 0] = 1.0
    motion_command = MotionCommand(
        query_offsets=offsets,
        joint_names=guide_joints,
        body_names=guide_bodies,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_position_world=body_position,
        body_orientation_world=body_orientation,
        body_linear_velocity_world=np.zeros_like(body_position),
        body_angular_velocity_world=np.zeros_like(body_position),
    )
    return state, motion_command


def test_sonic_is_registered_and_release_schema_is_exact() -> None:
    assert "sonic" in available()
    config = load_instance_config(CONFIG_PATH)
    assert config.motion_command_query_offsets == tuple(range(0, 50, 5))
    assert tuple(spec.non_batch_shape for spec in config.policy_inputs) == (
        (640,),
        (930,),
    )
    assert tuple(spec.name for spec in config.policy_outputs) == ("action",)
    other = load_instance_config(CONFIG_PATH.with_name("heft_pmg.yaml"))
    assert config.motion_command_body_names == other.motion_command_body_names


def test_first_observation_matches_independent_fixed_formula() -> None:
    config = load_instance_config(CONFIG_PATH)
    state, motion_command = _scene(config)
    adapter = get_family("sonic")(config)
    inputs, bfm_state = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )

    expected_command = np.concatenate(
        (
            np.concatenate(
                tuple(
                    np.arange(29, dtype=np.float32) + 100.0 * query
                    for query in range(10)
                )
            ),
            np.concatenate(
                tuple(
                    np.arange(29, dtype=np.float32) + 1000.0 * query + 0.5
                    for query in range(10)
                )
            ),
        )
    )
    expected_anchor = np.tile(
        np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], np.float32), 10
    )
    np.testing.assert_allclose(
        inputs["g1_input"][0],
        np.concatenate((expected_command, expected_anchor)),
        rtol=0.0,
        atol=1.0e-5,
    )

    default = np.asarray(config.family_config["default_joint_position"], np.float32)
    expected_proprioception = np.concatenate(
        (
            np.tile(np.asarray([2.0, -1.0, 3.0], np.float32), 10),
            np.tile(np.arange(1, 30, dtype=np.float32) - default, 10),
            np.tile(np.arange(51, 80, dtype=np.float32), 10),
            np.zeros(10 * 29, np.float32),
            np.tile(np.asarray([0.0, 0.0, -1.0], np.float32), 10),
        )
    )
    np.testing.assert_allclose(
        inputs["proprioception"][0], expected_proprioception, atol=1.0e-5
    )
    np.testing.assert_allclose(
        bfm_state["heading_offset"][0],
        _yaw_quat(np.pi / 2.0),
        atol=1.0e-6,
    )


def test_histories_are_oldest_to_newest_and_heading_is_fixed() -> None:
    config = load_instance_config(CONFIG_PATH)
    adapter = get_family("sonic")(config)
    first_state, motion_command = _scene(config)
    _, bfm_state = adapter.build_policy_inputs(
        first_state, motion_command, adapter.initial_state(1)
    )
    restored_state = adapter.clone_state(bfm_state)
    action = np.linspace(-12.0, 12.0, 29, dtype=np.float32)[None]
    _, bfm_state = adapter.map_policy_outputs({"action": action}, bfm_state)
    second_state, motion_command = _scene(
        config,
        robot_yaw=0.0,
        joint_shift=100.0,
        velocity_shift=200.0,
        angular_velocity=(4.0, 5.0, 6.0),
    )
    inputs, bfm_state = adapter.build_policy_inputs(
        second_state, motion_command, bfm_state
    )

    anchor = inputs["g1_input"][0, -60:].reshape(10, 6)
    np.testing.assert_allclose(
        anchor,
        np.tile(
            np.asarray([0.0, -1.0, 1.0, 0.0, 0.0, 0.0], np.float32),
            (10, 1),
        ),
        atol=1.0e-5,
    )
    proprioception = inputs["proprioception"][0]
    root_history = proprioception[:30].reshape(10, 3)
    np.testing.assert_allclose(
        root_history[:9],
        np.tile(np.asarray([2.0, -1.0, 3.0], np.float32), (9, 1)),
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(root_history[-1], [4.0, 5.0, 6.0])
    action_history = proprioception[30 + 290 + 290 : 30 + 3 * 290].reshape(
        10, 29
    )
    np.testing.assert_array_equal(action_history[:9], 0.0)
    np.testing.assert_array_equal(action_history[-1], action[0])
    np.testing.assert_array_equal(bfm_state["previous_actions"][0, -1], action[0])

    expanded = adapter.expand_state(restored_state, 2)
    np.testing.assert_allclose(
        expanded["heading_offset"],
        np.tile(_yaw_quat(np.pi / 2.0), (2, 1)),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        restored_state["heading_offset"][0],
        _yaw_quat(np.pi / 2.0),
        atol=1.0e-6,
    )


def test_action_is_scaled_and_reordered_to_canonical_command() -> None:
    config = load_instance_config(CONFIG_PATH)
    adapter = get_family("sonic")(config)
    bfm_state = adapter.initial_state(1)
    action = np.linspace(-12.0, 12.0, 29, dtype=np.float32)[None]
    command, next_state = adapter.map_policy_outputs({"action": action}, bfm_state)
    default = np.asarray(config.family_config["default_joint_position"], np.float32)
    scale = np.asarray(config.family_config["action_scale"], np.float32)
    policy_target = default + action[0] * scale
    order = [
        config.policy_joint_names.index(name) for name in config.command_joint_names
    ]
    np.testing.assert_array_equal(command.position_target[0], policy_target[order])
    np.testing.assert_array_equal(next_state["last_action"], action)
    np.testing.assert_array_equal(
        bfm_state["last_action"], np.zeros((1, 29), np.float32)
    )


def test_sonic_query_offsets_come_from_instance_config() -> None:
    config = replace(
        load_instance_config(CONFIG_PATH),
        motion_command_query_offsets=tuple(range(10)),
    )
    adapter = get_family("sonic")(config)
    state, motion_command = _scene(config)
    inputs, _ = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )
    assert motion_command.query_offsets == tuple(range(10))
    assert inputs["g1_input"].shape == (1, 640)


def test_sonic_requires_exactly_ten_query_offsets() -> None:
    config = load_instance_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="10"):
        get_family("sonic")(
            replace(config, motion_command_query_offsets=tuple(range(9)))
        )


def test_sonic_rejects_negative_query_offsets() -> None:
    config = load_instance_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="non-negative"):
        get_family("sonic")(
            replace(config, motion_command_query_offsets=(-1, *range(9)))
        )


@pytest.mark.parametrize(
    "offsets",
    (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (0, 1, 2, 3, 4, 5, 6, 8, 7, 9),
    ),
)
def test_sonic_query_offsets_start_at_zero_and_strictly_increase(
    offsets: tuple[int, ...],
) -> None:
    config = load_instance_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="zero and be strictly increasing"):
        get_family("sonic")(replace(config, motion_command_query_offsets=offsets))


def test_sonic_low_latency_instance_uses_dense_ten_step_offsets() -> None:
    config = load_instance_config(CONFIG_PATH.with_name("sonic_low_latency_g1.yaml"))
    adapter = get_family("sonic")(config)
    state, motion_command = _scene(config)
    inputs, _ = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(1)
    )

    assert config.instance == "sonic_low_latency_g1"
    assert config.motion_command_query_offsets == tuple(range(10))
    assert config.artifacts["policy"] == "sonic-low-latency-g1-policy.batched"
    assert isinstance(adapter, DeviceBFMProvider)
    assert inputs["g1_input"].shape == (1, 640)
    assert inputs["proprioception"].shape == (1, 930)


def test_sonic_motion_command_offsets_must_match_instance_config() -> None:
    config = load_instance_config(CONFIG_PATH)
    adapter = get_family("sonic")(config)
    state, motion_command = _scene(config)
    mismatched = replace(motion_command, query_offsets=tuple(range(10)))
    with pytest.raises(ValueError, match="offsets"):
        adapter.build_policy_inputs(state, mismatched, adapter.initial_state(1))
