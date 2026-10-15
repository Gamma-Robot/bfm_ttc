"""CUDA parity for SONIC's prepared device execution path."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sonic_family
from bfm_adapters import (
    DevicePDCommand,
    MotionCommand,
    PreparedDeviceBFM,
    load_instance_config,
)
from bfm_adapters.families.sonic.adapter import SonicAdapter
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import ensure_artifact

ort = pytest.importorskip("onnxruntime")
wp = pytest.importorskip("warp")

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO_ROOT = Path(__file__).resolve().parents[2]
SONIC_INSTANCES = (
    ("sonic_release_g1.yaml", "sonic-release-g1-policy.batched"),
    ("sonic_low_latency_g1.yaml", "sonic-low-latency-g1-policy.batched"),
)
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


def _load_instance(config_name: str):
    return load_instance_config(REPO_ROOT / "configs" / "bfms" / config_name)


def _timeline(config, batch_size: int, num_steps: int) -> MotionCommand:
    rng = np.random.default_rng(900 + batch_size)
    length = num_steps + max(config.motion_command_query_offsets)
    joint_names = config.command_joint_names
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
            for name in MOTION_COMMAND_FIELDS
        },
    )


def _contexts(
    config, batch_size: int, num_steps: int
) -> tuple[MotionCommand, MotionCommand, MotionCommand]:
    future_steps = max(config.motion_command_query_offsets)
    timeline = _timeline(config, batch_size, num_steps)
    candidate = _take(timeline, range(num_steps), tuple(range(num_steps)))
    right = _take(
        timeline,
        range(num_steps, num_steps + future_steps),
        tuple(range(future_steps)),
    )
    return timeline, candidate, right


def _query(config, timeline: MotionCommand, step_index: int) -> MotionCommand:
    offsets = config.motion_command_query_offsets
    return _take(
        timeline,
        [step_index + offset for offset in offsets],
        offsets,
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


def _copy_device(value):
    copied = wp.empty(value.shape, dtype=value.dtype, device=value.device)
    wp.copy(copied, value)
    return copied


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("config_name", "artifact_id"),
    SONIC_INSTANCES,
    ids=("release", "low_latency"),
)
def test_sonic_prepared_device_bfm_matches_numpy_oracle(
    batch_size: int, config_name: str, artifact_id: str
) -> None:
    assert wp.is_cuda_available()
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    config = _load_instance(config_name)
    model_path = ensure_artifact(artifact_id, root=REPO_ROOT)
    device_index = int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))
    device = wp.get_device(f"cuda:{device_index}")
    num_steps = 3
    states = [
        sonic_family.make_robot_state(config, batch_size, 1000 + step_index)
        for step_index in range(num_steps)
    ]
    timeline, motion_command, right = _contexts(config, batch_size, num_steps)
    view = _device_view(states[0], device)
    command_buffers = _device_command(config, batch_size, device)

    reference_adapter = SonicAdapter(config)
    reference_policy = BatchedOrtCudaPolicy(
        model_path, config, device_id=device.ordinal
    )
    reference_state = reference_adapter.initial_state(batch_size)
    bfm = reference_adapter.build_device_bfm(
        model_path, view, command_buffers
    )
    assert isinstance(bfm, PreparedDeviceBFM)
    assert not bfm._policy.asynchronous
    binding = bfm._policy._binding
    bfm.reset(reference_state)
    bfm.prepare(None, motion_command, right, num_steps=num_steps)

    try:
        expected_inputs_by_step = []
        expected_commands = []
        actual_inputs_by_step = []
        actual_commands = []
        for step_index, state in enumerate(states):
            _assign_view(view, state)
            expected_inputs, observed_state = reference_adapter.build_policy_inputs(
                state, _query(config, timeline, step_index), reference_state
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
                    for name in PD_COMMAND_FIELDS
                }
            )

        bfm._policy.enqueue = lambda: pytest.fail(
            "capture_state must not execute the policy"
        )
        actual_state = bfm.capture_state()
        assert bfm._policy._binding is binding
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
            for name in PD_COMMAND_FIELDS:
                np.testing.assert_allclose(
                    actual_command[name].numpy(),
                    getattr(expected_command, name),
                    rtol=0.0,
                    atol=1.0e-3,
                )
        for name, expected in reference_state.items():
            if expected.dtype == bool:
                np.testing.assert_array_equal(actual_state[name], expected)
            else:
                tolerance = 1.0e-3 if name == "last_action" else 3.0e-5
                np.testing.assert_allclose(
                    actual_state[name], expected, rtol=0.0, atol=tolerance
                )
    finally:
        bfm.close()
        bfm.close()
