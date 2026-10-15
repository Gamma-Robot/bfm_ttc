"""CUDA parity for TWIST2's prepared device execution path."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import twist2_family
from bfm_adapters import DevicePDCommand, MotionCommand, get_family
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import ensure_artifact

wp = pytest.importorskip("warp")
pytest.importorskip("onnxruntime")

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = twist2_family._CONFIG_PATH.parents[2]
PD_COMMAND_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)
MOTION_COMMAND_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)


def _timeline(batch_size: int, num_steps: int) -> MotionCommand:
    config = twist2_family.instance_config()
    rng = np.random.default_rng(1500 + batch_size)
    length = num_steps + 1
    joint_names = tuple(reversed(config.policy_joint_names))
    body_names = config.motion_command_body_names

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, length, *shape)).astype(np.float32)

    orientation = draw(len(body_names), 4)
    orientation[..., 0] = np.abs(orientation[..., 0]) + 1.0
    orientation /= np.linalg.norm(orientation, axis=-1, keepdims=True)
    return MotionCommand(
        query_offsets=tuple(range(length)),
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=orientation.astype(np.float32),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def _take(
    motion_command: MotionCommand, indices: list[int] | range, offsets: tuple[int, ...]
) -> MotionCommand:
    selected = np.asarray(indices, dtype=np.int64)
    return MotionCommand(
        query_offsets=offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **{
            name: np.ascontiguousarray(getattr(motion_command, name)[:, selected])
            for name in MOTION_COMMAND_FIELDS
        },
    )


def _device_view(state, device):
    return SimpleNamespace(
        joint_names=state.joint_names,
        body_names=state.body_names,
        joint_position=wp.array(state.joint_position, dtype=wp.float32, device=device),
        joint_velocity=wp.array(state.joint_velocity, dtype=wp.float32, device=device),
        body_position_world=wp.array(
            state.body_position_world, dtype=wp.vec3, device=device
        ),
        body_orientation_world=wp.array(
            state.body_orientation_world, dtype=wp.quat, device=device
        ),
        body_linear_velocity_world=wp.array(
            state.body_linear_velocity_world, dtype=wp.vec3, device=device
        ),
        body_angular_velocity_world=wp.array(
            state.body_angular_velocity_world, dtype=wp.vec3, device=device
        ),
    )


def _assign_view(view, state) -> None:
    for name in (
        "joint_position",
        "joint_velocity",
        "body_position_world",
        "body_orientation_world",
        "body_linear_velocity_world",
        "body_angular_velocity_world",
    ):
        getattr(view, name).assign(np.ascontiguousarray(getattr(state, name)))


def _device_command(config, batch_size: int, device) -> DevicePDCommand:
    arrays = {
        name: wp.empty(
            (batch_size, len(config.command_joint_names)),
            dtype=wp.float32,
            device=device,
        )
        for name in PD_COMMAND_FIELDS
    }
    return DevicePDCommand(joint_names=config.command_joint_names, **arrays)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_twist2_prepared_device_bfm_matches_host_oracle(batch_size: int) -> None:
    wp.init()
    assert wp.is_cuda_available()
    device = wp.get_device(
        f"cuda:{int(os.environ.get('BFM_TTC_TEST_DEVICE', '0'))}"
    )
    config = twist2_family.instance_config()
    adapter = get_family("twist2")(config)
    policy_path = ensure_artifact("twist2-1017-20k-policy", root=REPO)
    host_policy = BatchedOrtCudaPolicy(
        policy_path, config, device_id=int(device.ordinal)
    )
    states = [
        twist2_family.make_robot_state(config, batch_size, 1600 + step_index)
        for step_index in range(3)
    ]
    timeline = _timeline(batch_size, 3)
    candidate = _take(timeline, range(3), tuple(range(3)))
    right = _take(timeline, [3], (0,))
    view = _device_view(states[0], device)
    command_view = _device_command(config, batch_size, device)
    bfm = adapter.build_device_bfm(policy_path, view, command_view)
    initial = adapter.initial_state(batch_size)
    host_state = adapter.clone_state(initial)
    bfm.reset(initial)
    bfm.prepare(None, candidate, right, num_steps=3)

    try:
        for step_index, state in enumerate(states):
            _assign_view(view, state)
            query = _take(
                timeline, [step_index, step_index + 1], (0, 1)
            )
            host_inputs, observed = adapter.build_policy_inputs(
                state, query, host_state
            )
            host_outputs = host_policy.run(host_inputs)
            host_command, host_state = adapter.map_policy_outputs(
                host_outputs, observed
            )
            bfm.step_prepared(view, step_index)

            np.testing.assert_allclose(
                bfm._policy_inputs["input"].numpy(),
                host_inputs["input"],
                rtol=0.0,
                atol=3.0e-5,
            )
            for name in PD_COMMAND_FIELDS:
                np.testing.assert_allclose(
                    getattr(command_view, name).numpy(),
                    getattr(host_command, name),
                    rtol=1.0e-5,
                    atol=1.0e-5,
                )
            device_state = bfm.capture_state()
            assert device_state["history_fill_steps_remaining"].tobytes() == (
                host_state["history_fill_steps_remaining"].tobytes()
            )
            for name in ("history", "last_action"):
                np.testing.assert_allclose(
                    device_state[name], host_state[name], rtol=1.0e-5, atol=3.0e-5
                )

        captured_state = bfm.capture_state()
        bfm.reset(captured_state)
        restored = bfm.capture_state()
        for name in captured_state:
            np.testing.assert_array_equal(restored[name], captured_state[name])
    finally:
        bfm.close()
