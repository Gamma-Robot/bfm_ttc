"""CUDA parity for HEFT's prepared device execution path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import heft_family
from bfm_adapters import DevicePDCommand, MotionCommand, PreparedDeviceBFM
from bfm_adapters.families.heft.adapter import HeftAdapter
from bfm_adapters.families.heft.device import _map_action
from bfm_adapters.families.heft.observation import QUERY_OFFSETS
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import ensure_artifact

ort = pytest.importorskip("onnxruntime")
wp = pytest.importorskip("warp")

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMAND_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)
_COMMAND_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)
_PAST = 16
_FUTURE = 6


def _timeline(batch_size: int, num_steps: int) -> MotionCommand:
    config = heft_family.instance_config()
    rng = np.random.default_rng(500 + batch_size)
    length = _PAST + num_steps + _FUTURE
    joint_names = tuple(reversed(config.command_joint_names))
    body_names = tuple(reversed(config.motion_command_body_names))

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal((batch_size, length, *shape)).astype(np.float32)

    orientations = draw(len(body_names), 4)
    orientations[..., 0] = np.abs(orientations[..., 0]) + 1.0
    orientations /= np.linalg.norm(orientations, axis=-1, keepdims=True)
    return MotionCommand(
        query_offsets=tuple(range(length)),
        joint_names=joint_names,
        body_names=body_names,
        joint_position=draw(len(joint_names)),
        joint_velocity=draw(len(joint_names)),
        body_position_world=draw(len(body_names), 3),
        body_orientation_world=orientations.astype(np.float32),
        body_linear_velocity_world=draw(len(body_names), 3),
        body_angular_velocity_world=draw(len(body_names), 3),
    )


def _take(
    motion_command: MotionCommand,
    indices: list[int] | range,
    offsets: tuple[int, ...],
) -> MotionCommand:
    selected = np.asarray(indices, dtype=np.int64)
    return MotionCommand(
        query_offsets=offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **{
            name: np.ascontiguousarray(getattr(motion_command, name)[:, selected])
            for name in _COMMAND_FIELDS
        },
    )


def _contexts(
    batch_size: int, num_steps: int
) -> tuple[MotionCommand, MotionCommand, MotionCommand, MotionCommand]:
    timeline = _timeline(batch_size, num_steps)
    history = _take(timeline, range(_PAST), tuple(range(-_PAST, 0)))
    candidate = _take(
        timeline, range(_PAST, _PAST + num_steps), tuple(range(num_steps))
    )
    right = _take(
        timeline,
        range(_PAST + num_steps, _PAST + num_steps + _FUTURE),
        tuple(range(_FUTURE)),
    )
    return timeline, history, candidate, right


def _query(timeline: MotionCommand, step_index: int) -> MotionCommand:
    origin = _PAST + step_index
    return _take(
        timeline,
        [origin + offset for offset in QUERY_OFFSETS],
        QUERY_OFFSETS,
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
        for name in COMMAND_FIELDS
    }
    return DevicePDCommand(joint_names=config.command_joint_names, **arrays)


def _copy_device(value):
    copied = wp.empty(value.shape, dtype=value.dtype, device=value.device)
    wp.copy(copied, value)
    return copied


def _nontrivial_state(adapter: HeftAdapter, batch_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(700 + batch_size)
    state = adapter.initial_state(batch_size)
    state["boot"] = np.arange(25, 25 - batch_size, -1, dtype=np.int32)
    for name in (
        "root_angular_velocity",
        "projected_gravity",
        "joint_position",
        "joint_velocity",
        "previous_actions",
        "last_action",
    ):
        state[name] = rng.standard_normal(state[name].shape).astype(np.float32)
    return state


@pytest.mark.parametrize("batch_size", [1, 4])
def test_heft_prepared_device_bfm_matches_numpy_oracle(batch_size: int) -> None:
    assert wp.is_cuda_available()
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    config = heft_family.instance_config()
    model_path = ensure_artifact("heft-pmg-policy.batched", root=REPO_ROOT)
    device = wp.get_device("cuda:0")
    num_steps = 3
    states = [
        heft_family.make_robot_state(config, batch_size, 200 + step_index)
        for step_index in range(num_steps)
    ]
    timeline, history, motion_command, right = _contexts(batch_size, num_steps)
    view = _device_view(states[0], device)
    command_buffers = _device_command(config, batch_size, device)

    reference_adapter = HeftAdapter(config)
    reference_policy = BatchedOrtCudaPolicy(
        model_path, config, device_id=device.ordinal
    )
    reference_state = _nontrivial_state(reference_adapter, batch_size)
    bfm = reference_adapter.build_device_bfm(
        model_path, view, command_buffers
    )
    assert isinstance(bfm, PreparedDeviceBFM)
    assert bfm._policy.asynchronous
    bfm.reset(reference_state)
    bfm.prepare(history, motion_command, right, num_steps=num_steps)

    try:
        expected_inputs_by_step = []
        expected_commands = []
        actual_inputs_by_step = []
        actual_commands = []
        for step_index, state in enumerate(states):
            _assign_view(view, state)
            expected_inputs, observed_state = reference_adapter.build_policy_inputs(
                state, _query(timeline, step_index), reference_state
            )
            expected_outputs = reference_policy.run(expected_inputs)
            expected_command, reference_state = reference_adapter.map_policy_outputs(
                expected_outputs, observed_state
            )

            actual_command = bfm.step_prepared(view, step_index)
            assert actual_command is command_buffers
            expected_inputs_by_step.append(expected_inputs)
            expected_commands.append(expected_command)
            actual_inputs_by_step.append(
                {
                    name: _copy_device(value)
                    for name, value in bfm._policy_inputs.items()
                }
            )
            actual_commands.append(
                {
                    name: _copy_device(getattr(actual_command, name))
                    for name in COMMAND_FIELDS
                }
            )

        actual_state = bfm.capture_state()
        assert bfm._policy._pending == 0
        bfm.reset(actual_state)
        assert bfm._policy._pending == 0
        for expected_inputs, actual_inputs in zip(
            expected_inputs_by_step, actual_inputs_by_step, strict=True
        ):
            for name, expected in expected_inputs.items():
                np.testing.assert_allclose(
                    actual_inputs[name].numpy(), expected, rtol=0.0, atol=3.0e-5
                )
        for expected_command, actual_command in zip(
            expected_commands, actual_commands, strict=True
        ):
            for name in COMMAND_FIELDS:
                np.testing.assert_allclose(
                    actual_command[name].numpy(),
                    getattr(expected_command, name),
                    rtol=0.0,
                    atol=3.0e-5,
                )
        for name, expected in reference_state.items():
            np.testing.assert_allclose(
                actual_state[name], expected, rtol=0.0, atol=3.0e-5
            )
    finally:
        bfm.close()


def test_heft_device_action_mapping_clips_and_reorders() -> None:
    config = heft_family.instance_config()
    adapter = HeftAdapter(config)
    device = wp.get_device("cuda:0")
    batch_size = 2
    action = np.stack(
        (
            np.linspace(-12.0, 12.0, 29, dtype=np.float32),
            np.linspace(12.0, -12.0, 29, dtype=np.float32),
        )
    )
    command = _device_command(config, batch_size, device)
    last_action = wp.empty((batch_size, 29), dtype=wp.float32, device=device)
    command_to_policy = np.asarray(
        [
            config.policy_joint_names.index(name)
            for name in config.command_joint_names
        ],
        np.int32,
    )
    family = config.family_config
    wp.launch(
        _map_action,
        dim=(batch_size, 29),
        inputs=[
            wp.array(action, dtype=wp.float32, device=device),
            wp.array(command_to_policy, dtype=wp.int32, device=device),
            wp.array(family["default_joint_position"], dtype=wp.float32, device=device),
            wp.array(family["action_scale"], dtype=wp.float32, device=device),
            wp.array(family["proportional_gain"], dtype=wp.float32, device=device),
            wp.array(family["derivative_gain"], dtype=wp.float32, device=device),
            float(family["action_clip"]),
            command.position_target,
            command.velocity_target,
            command.feedforward_torque,
            command.proportional_gain,
            command.derivative_gain,
            last_action,
        ],
        device=device,
    )
    expected_command, expected_state = adapter.map_policy_outputs(
        {"action": action}, adapter.initial_state(batch_size)
    )
    for name in COMMAND_FIELDS:
        np.testing.assert_allclose(
            getattr(command, name).numpy(),
            getattr(expected_command, name),
            rtol=0.0,
            atol=1.0e-6,
        )
    np.testing.assert_array_equal(last_action.numpy(), expected_state["last_action"])
