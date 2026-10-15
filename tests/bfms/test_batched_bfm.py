"""BFM runtime tests through the public batched-bfm seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import toy_family
import heft_family
from bfm_adapters import (
    BatchedBFM,
    MotionCommand,
    PDCommand,
    RobotState,
    RobotStateView,
    SerialReferenceBFM,
    get_family,
)

COMMAND_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)


class _RecordingPolicy:
    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs):
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        return toy_family.fake_policy(self.config, inputs)


def _slice_state(state: RobotState, lane: int) -> RobotState:
    select = lambda value: value[lane : lane + 1]
    return RobotState(
        joint_names=state.joint_names,
        body_names=state.body_names,
        joint_position=select(state.joint_position),
        joint_velocity=select(state.joint_velocity),
        body_position_world=select(state.body_position_world),
        body_orientation_world=select(state.body_orientation_world),
        body_linear_velocity_world=select(state.body_linear_velocity_world),
        body_angular_velocity_world=select(state.body_angular_velocity_world),
    )


def _slice_motion_command(motion_command: MotionCommand, lane: int) -> MotionCommand:
    select = lambda value: value[lane : lane + 1]
    return MotionCommand(
        query_offsets=motion_command.query_offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        joint_position=select(motion_command.joint_position),
        joint_velocity=select(motion_command.joint_velocity),
        body_position_world=select(motion_command.body_position_world),
        body_orientation_world=select(motion_command.body_orientation_world),
        body_linear_velocity_world=select(motion_command.body_linear_velocity_world),
        body_angular_velocity_world=select(motion_command.body_angular_velocity_world),
    )


def _assert_commands_close(
    actual: PDCommand, expected: PDCommand, *, atol: float = 0.0
) -> None:
    assert actual.joint_names == expected.joint_names
    for field in COMMAND_FIELDS:
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=0.0, atol=atol
        )


def _stack_commands(commands: list[PDCommand]) -> PDCommand:
    values = {
        field: np.concatenate([getattr(command, field) for command in commands])
        for field in COMMAND_FIELDS
    }
    return PDCommand(joint_names=commands[0].joint_names, **values)


def _toy_bfm(batch_size: int, policy=None) -> SerialReferenceBFM:
    config = toy_family.instance_config()
    return SerialReferenceBFM(
        get_family(config.family)(config),
        policy if policy is not None else _RecordingPolicy(config),
        batch_size=batch_size,
    )


def test_batched_step_matches_independent_lanes_and_batches_policy_once() -> None:
    config = toy_family.instance_config()
    batch_size = 4
    state = toy_family.make_robot_state(config, batch_size, 7)
    motion_command = toy_family.make_motion_command(config, batch_size, 11)
    policy = _RecordingPolicy(config)
    batched = _toy_bfm(batch_size, policy)
    serial = [_toy_bfm(1) for _ in range(batch_size)]

    assert isinstance(state, RobotStateView)
    assert isinstance(batched, BatchedBFM)
    for _ in range(3):
        actual = batched.step(state, motion_command)
        expected = _stack_commands(
            [
                bfm.step(_slice_state(state, lane), _slice_motion_command(motion_command, lane))
                for lane, bfm in enumerate(serial)
            ]
        )
        _assert_commands_close(actual, expected)

    assert len(policy.calls) == 3
    assert all(next(iter(inputs.values())).shape[0] == batch_size for inputs in policy.calls)


def test_capture_and_reset_restore_opaque_bfm_state_without_aliasing() -> None:
    config = toy_family.instance_config()
    state = toy_family.make_robot_state(config, 2, 13)
    motion_command = toy_family.make_motion_command(config, 2, 17)
    bfm = _toy_bfm(2)

    bfm.step(state, motion_command)
    captured_state = bfm.capture_state()
    expected = bfm.step(state, motion_command)
    bfm.step(state, motion_command)

    bfm.reset(captured_state)
    _assert_commands_close(bfm.step(state, motion_command), expected)
    bfm.step(state, motion_command)
    bfm.reset(captured_state)
    _assert_commands_close(bfm.step(state, motion_command), expected)


def test_failed_step_does_not_advance_bfm_state() -> None:
    config = toy_family.instance_config()
    state = toy_family.make_robot_state(config, 2, 23)
    motion_command = toy_family.make_motion_command(config, 2, 29)

    class FailOncePolicy(_RecordingPolicy):
        def run(self, inputs):
            if not self.calls:
                self.calls.append({name: value.copy() for name, value in inputs.items()})
                raise RuntimeError("policy failed")
            return super().run(inputs)

    bfm = _toy_bfm(2, FailOncePolicy(config))
    with pytest.raises(RuntimeError, match="policy failed"):
        bfm.step(state, motion_command)

    actual = bfm.step(state, motion_command)
    expected = _toy_bfm(2).step(state, motion_command)
    _assert_commands_close(actual, expected)


def test_robot_state_view_materializes_device_like_arrays_on_host() -> None:
    config = toy_family.instance_config()
    state = toy_family.make_robot_state(config, 2, 31)
    motion_command = toy_family.make_motion_command(config, 2, 37)
    calls: list[int] = []

    class DeviceArray:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            calls.append(1)
            return self.value

    view = SimpleNamespace(
        joint_names=state.joint_names,
        body_names=state.body_names,
        joint_position=DeviceArray(state.joint_position),
        joint_velocity=DeviceArray(state.joint_velocity),
        body_position_world=DeviceArray(state.body_position_world),
        body_orientation_world=DeviceArray(state.body_orientation_world),
        body_linear_velocity_world=DeviceArray(state.body_linear_velocity_world),
        body_angular_velocity_world=DeviceArray(state.body_angular_velocity_world),
    )

    actual = _toy_bfm(2).step(view, motion_command)
    expected = _toy_bfm(2).step(state, motion_command)
    _assert_commands_close(actual, expected)
    assert len(calls) == 6


def test_rejects_invalid_batch_size_and_motion_command_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _toy_bfm(0)

    config = toy_family.instance_config()
    state = toy_family.make_robot_state(config, 2, 41)
    motion_command = toy_family.make_motion_command(config, 1, 43)
    with pytest.raises(ValueError, match="batch sizes differ"):
        _toy_bfm(2).step(state, motion_command)


@pytest.mark.gpu
@pytest.mark.conformance
def test_heft_batched_bfm_matches_batch1_cuda_oracle() -> None:
    """Qualification gate: missing CUDA or artifacts is a failure, never a skip."""

    import onnxruntime as ort

    from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
    from command_optimization.artifacts import ensure_artifact

    assert "CUDAExecutionProvider" in ort.get_available_providers()
    config = heft_family.instance_config()
    model_path = ensure_artifact("heft-pmg-policy.batched")
    policy = BatchedOrtCudaPolicy(model_path, config, device_id=0)
    batch_size = 4
    state = heft_family.make_robot_state(config, batch_size, 47)
    motion_command = heft_family.make_motion_command(config, batch_size, 53)
    batched = SerialReferenceBFM(
        get_family(config.family)(config), policy, batch_size=batch_size
    )
    serial = [
        SerialReferenceBFM(get_family(config.family)(config), policy, batch_size=1)
        for _ in range(batch_size)
    ]

    for _ in range(2):
        actual = batched.step(state, motion_command)
        expected = _stack_commands(
            [
                bfm.step(_slice_state(state, lane), _slice_motion_command(motion_command, lane))
                for lane, bfm in enumerate(serial)
            ]
        )
        _assert_commands_close(actual, expected, atol=2e-5)
