from __future__ import annotations

import numpy as np
import pytest

from bfm_adapters import DevicePDCommand, MotionCommand, PDCommand, RobotState
from command_optimization.simulation.command_execution import CommandExecutor
from command_optimization.simulation.measurements import MeasurementPrimitive
from command_optimization.simulation.execution_state import ExecutionState


JOINTS = ("joint",)
BODIES = ("pelvis",)


def _robot_state(qpos: np.ndarray, qvel: np.ndarray) -> RobotState:
    batch = len(qpos)
    orientation = np.zeros((batch, 1, 4))
    orientation[..., 0] = 1.0
    return RobotState(
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=qpos.copy(),
        joint_velocity=qvel.copy(),
        body_position_world=np.pad(qpos[:, :, None], ((0, 0), (0, 0), (0, 2))),
        body_orientation_world=orientation,
        body_linear_velocity_world=np.pad(
            qvel[:, :, None], ((0, 0), (0, 0), (0, 2))
        ),
        body_angular_velocity_world=np.zeros((batch, 1, 3)),
    )


class _FakeWorld:
    batch_size = 2
    physics_timestep = 0.05
    joint_names = JOINTS
    body_names = BODIES

    def __init__(self) -> None:
        self.qpos = np.zeros((2, 1))
        self.qvel = np.zeros((2, 1))
        self.time = np.zeros(2)
        self.command: PDCommand | DevicePDCommand | None = None
        self.substeps: list[int] = []
        self.physical_state_calls = 0
        self.state_view_calls = 0

    @property
    def state_view(self) -> RobotState:
        self.state_view_calls += 1
        return _robot_state(self.qpos, self.qvel)

    def reset(self, execution_state: ExecutionState) -> None:
        self.qpos = execution_state.qpos.copy()
        self.qvel = execution_state.qvel.copy()
        self.time.fill(execution_state.time)
        self.command = None

    def apply_pd(self, command: PDCommand | DevicePDCommand) -> None:
        self.command = command

    def host_command(self) -> PDCommand:
        assert self.command is not None
        return PDCommand(
            joint_names=self.command.joint_names,
            position_target=np.asarray(self.command.position_target).copy(),
            velocity_target=np.asarray(self.command.velocity_target).copy(),
            feedforward_torque=np.asarray(self.command.feedforward_torque).copy(),
            proportional_gain=np.asarray(self.command.proportional_gain).copy(),
            derivative_gain=np.asarray(self.command.derivative_gain).copy(),
        )

    def step(self, num_substeps: int) -> None:
        assert self.command is not None
        self.substeps.append(num_substeps)
        for _ in range(num_substeps):
            self.qvel = self.command.position_target.copy()
            self.qpos += self.physics_timestep * self.qvel
            self.time += self.physics_timestep

    def robot_state(self) -> RobotState:
        return _robot_state(self.qpos, self.qvel)

    def physical_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.physical_state_calls += 1
        return self.qpos.copy(), self.qvel.copy(), self.time.copy()

    def warmstart_acceleration(self) -> np.ndarray:
        return np.zeros_like(self.qvel)


class _FakeBFM:
    def __init__(self, offsets: tuple[int, ...] = (0, 2)) -> None:
        self.state = {"num_steps": 0}
        self.seen: list[np.ndarray] = []
        self.offsets = offsets

    def reset(self, state: object) -> None:
        self.state = dict(state)  # type: ignore[arg-type]

    def capture_state(self) -> object:
        return dict(self.state)

    def step(self, state_view: object, motion_command: MotionCommand) -> PDCommand:
        assert isinstance(state_view, RobotState)
        assert motion_command.query_offsets == self.offsets
        self.seen.append(motion_command.joint_position.copy())
        self.state["num_steps"] += 1
        target = motion_command.joint_position[:, motion_command.query_offsets.index(0)]
        zeros = np.zeros_like(target)
        return PDCommand(JOINTS, target, zeros, zeros, np.ones_like(target), zeros)


class _FakePreparedBFM(_FakeBFM):
    def __init__(self) -> None:
        super().__init__((0, 2))
        self.prepared: list[
            tuple[MotionCommand | None, MotionCommand, MotionCommand | None, int]
        ] = []

    def prepare(
        self,
        history: MotionCommand | None,
        motion_command: MotionCommand,
        right_context: MotionCommand | None,
        *,
        num_steps: int,
    ) -> None:
        self.prepared.append((history, motion_command, right_context, num_steps))

    def step(self, state_view: object, motion_command: MotionCommand) -> PDCommand:
        raise AssertionError("prepared bfm must not use the host step path")

    def step_prepared(
        self, state_view: object, step: int
    ) -> DevicePDCommand:
        assert isinstance(state_view, RobotState)
        motion_command = self.prepared[-1][1]
        self.state["num_steps"] += 1
        target = motion_command.joint_position[:, step].copy()
        zeros = np.zeros_like(target)
        return DevicePDCommand(
            JOINTS, target, zeros, zeros, np.ones_like(target), zeros
        )

    def close(self) -> None:
        pass


class _FakeReducer:
    def __init__(self) -> None:
        self.values: list[np.ndarray] = []

    def record(self, step: int, state_view: object) -> None:
        assert step == len(self.values)
        assert isinstance(state_view, RobotState)
        self.values.append(state_view.joint_position.copy())

    def finalize(self) -> np.ndarray:
        return np.stack(self.values).sum(axis=(0, 2))


def _motion_command() -> MotionCommand:
    batch, horizon = 2, 4
    return _motion_command_values(
        np.arange(1, batch * horizon + 1, dtype=np.float64).reshape(batch, horizon)
    )


def _motion_command_values(values: np.ndarray) -> MotionCommand:
    joint = np.asarray(values, dtype=np.float64)[..., None]
    batch, horizon, _ = joint.shape
    body_position = np.zeros((batch, horizon, 1, 3))
    body_orientation = np.zeros((batch, horizon, 1, 4))
    body_orientation[..., 0] = 1.0
    body_velocity = np.zeros_like(body_position)
    return MotionCommand(
        query_offsets=tuple(range(horizon)),
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body_position,
        body_orientation_world=body_orientation,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=body_velocity.copy(),
    )


def _history(values: np.ndarray) -> MotionCommand:
    motion_command = _motion_command_values(values)
    num_steps = len(motion_command.query_offsets)
    return MotionCommand(
        query_offsets=tuple(range(-num_steps, 0)),
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        joint_position=motion_command.joint_position,
        joint_velocity=motion_command.joint_velocity,
        body_position_world=motion_command.body_position_world,
        body_orientation_world=motion_command.body_orientation_world,
        body_linear_velocity_world=motion_command.body_linear_velocity_world,
        body_angular_velocity_world=motion_command.body_angular_velocity_world,
    )


def _execution_state(history: MotionCommand | None = None) -> ExecutionState:
    return ExecutionState(
        qpos=np.zeros((2, 1)),
        qvel=np.zeros((2, 1)),
        control_step=7,
        time=0.4,
        motion_command_index=11,
        command=None,
        bfm_state={"num_steps": 3},
        motion_command_history=history,
    )


def test_motion_command_executor_owns_the_complete_control_step_path() -> None:
    world = _FakeWorld()
    bfm = _FakeBFM()
    executor = CommandExecutor(
        world,  # type: ignore[arg-type]
        bfm,  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=(0, 2),
    )

    result = executor.execute(
        _execution_state(),
        _motion_command(),
        num_steps=2,
        requires=frozenset({MeasurementPrimitive.JOINT_POSITION}),
    )

    assert world.substeps == [2, 2]
    assert len(bfm.seen) == 2
    np.testing.assert_array_equal(bfm.seen[0][:, :, 0], [[1.0, 3.0], [5.0, 7.0]])
    np.testing.assert_array_equal(bfm.seen[1][:, :, 0], [[2.0, 4.0], [6.0, 8.0]])
    assert result.final_execution_state.control_step == 9
    assert result.final_execution_state.motion_command_index == 13
    assert result.final_execution_state.time == pytest.approx(0.6)
    assert result.final_execution_state.bfm_state == {"num_steps": 5}
    assert result.qpos.shape == (2, 2, 1)
    assert result.qvel.shape == (2, 2, 1)
    assert not result.qpos.flags.writeable
    assert result.measurements.arrays[MeasurementPrimitive.JOINT_POSITION].shape == (
        2,
        2,
        1,
    )
    assert world.physical_state_calls == 3
    assert world.state_view_calls == 1


def test_rollout_uses_the_same_control_step_path_without_trace_downloads() -> None:
    world = _FakeWorld()
    bfm = _FakeBFM()
    executor = CommandExecutor(
        world,  # type: ignore[arg-type]
        bfm,  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=(0, 2),
    )

    measurements = executor.rollout(
        _execution_state(),
        _motion_command(),
        num_steps=2,
        requires=frozenset({MeasurementPrimitive.JOINT_POSITION}),
    )

    assert measurements.num_steps == 2
    assert world.substeps == [2, 2]
    assert len(bfm.seen) == 2
    assert world.physical_state_calls == 0
    assert world.state_view_calls == 1


def test_prepared_device_bfm_binds_once_and_downloads_only_on_execute() -> None:
    world = _FakeWorld()
    bfm = _FakePreparedBFM()
    executor = CommandExecutor(
        world,  # type: ignore[arg-type]
        bfm,
        control_period=0.1,
        motion_command_query_offsets=(0, 2),
    )

    result = executor.execute(_execution_state(), _motion_command(), num_steps=2)

    assert len(bfm.prepared) == 1
    assert bfm.prepared[0][0] is None
    assert bfm.prepared[0][2] is None
    assert bfm.prepared[0][3] == 2
    assert bfm.state == {"num_steps": 5}
    assert world.state_view_calls == 2
    assert isinstance(result.final_execution_state.command, PDCommand)
    np.testing.assert_array_equal(
        result.final_execution_state.command.position_target[:, 0], [2.0, 6.0]
    )

    executor.rollout(_execution_state(), _motion_command(), num_steps=2)
    assert len(bfm.prepared) == 2
    assert bfm.prepared[-1][3] == 2


def test_prepared_reduced_rollout_never_downloads_per_step_host_state() -> None:
    world = _FakeWorld()
    bfm = _FakePreparedBFM()
    reducer = _FakeReducer()
    executor = CommandExecutor(
        world,  # type: ignore[arg-type]
        bfm,
        control_period=0.1,
        motion_command_query_offsets=(0, 2),
    )

    reduced = executor.rollout_reduced(
        _execution_state(), _motion_command(), reducer, num_steps=2
    )

    np.testing.assert_allclose(reduced, [0.4, 1.6])
    assert bfm.prepared[0][3] == 2
    assert world.physical_state_calls == 0
    assert world.state_view_calls == 4


def test_motion_command_executor_rejects_nonintegral_control_decimation() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        CommandExecutor(
            _FakeWorld(),  # type: ignore[arg-type]
            _FakeBFM(),  # type: ignore[arg-type]
            control_period=0.075,
            motion_command_query_offsets=(0,),
        )


def test_signed_queries_cross_history_candidate_and_reference_boundaries() -> None:
    offsets = (-2, 0, 1)
    bfm = _FakeBFM(offsets)
    executor = CommandExecutor(
        _FakeWorld(),  # type: ignore[arg-type]
        bfm,  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=offsets,
    )
    candidate = _motion_command_values([[20.0, 21.0, 22.0], [120.0, 121.0, 122.0]])
    execution_state = _execution_state(_history([[10.0, 11.0], [110.0, 111.0]]))

    first = executor.execute(execution_state, candidate, num_steps=2)

    np.testing.assert_array_equal(
        bfm.seen[0][:, :, 0], [[10.0, 20.0, 21.0], [110.0, 120.0, 121.0]]
    )
    np.testing.assert_array_equal(
        bfm.seen[1][:, :, 0], [[11.0, 21.0, 22.0], [111.0, 121.0, 122.0]]
    )
    assert first.final_execution_state.motion_command_history is not None
    np.testing.assert_array_equal(
        first.final_execution_state.motion_command_history.joint_position[:, :, 0],
        [[20.0, 21.0], [120.0, 121.0]],
    )

    next_candidate = _motion_command_values([[40.0, 41.0], [140.0, 141.0]])
    executor.execute(first.final_execution_state, next_candidate, num_steps=1)
    np.testing.assert_array_equal(
        bfm.seen[2][:, :, 0], [[20.0, 40.0, 41.0], [120.0, 140.0, 141.0]]
    )


def test_reference_right_context_is_observed_but_never_committed() -> None:
    bfm = _FakeBFM((0, 2))
    executor = CommandExecutor(
        _FakeWorld(),  # type: ignore[arg-type]
        bfm,  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=(0, 2),
    )
    candidate = _motion_command_values([[1.0, 2.0], [11.0, 12.0]])
    reference_right = _motion_command_values([[90.0, 91.0], [190.0, 191.0]])

    executor.execute(
        _execution_state(), candidate, num_steps=2, right_context=reference_right
    )

    np.testing.assert_array_equal(
        bfm.seen[0][:, :, 0], [[1.0, 90.0], [11.0, 190.0]]
    )
    np.testing.assert_array_equal(
        bfm.seen[1][:, :, 0], [[2.0, 91.0], [12.0, 191.0]]
    )


def test_zero_offset_bfm_needs_no_history_or_right_context() -> None:
    bfm = _FakeBFM((0,))
    executor = CommandExecutor(
        _FakeWorld(),  # type: ignore[arg-type]
        bfm,  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=(0,),
    )
    motion_command = _motion_command_values([[3.0, 4.0], [13.0, 14.0]])

    result = executor.execute(_execution_state(), motion_command)

    np.testing.assert_array_equal(
        [seen[:, 0, 0].tolist() for seen in bfm.seen],
        [[3.0, 13.0], [4.0, 14.0]],
    )
    assert result.final_execution_state.motion_command_history is None


def test_signed_executor_requires_sufficient_committed_history() -> None:
    executor = CommandExecutor(
        _FakeWorld(),  # type: ignore[arg-type]
        _FakeBFM((-2, 0)),  # type: ignore[arg-type]
        control_period=0.1,
        motion_command_query_offsets=(-2, 0),
    )
    with pytest.raises(ValueError, match="motion_command_history"):
        executor.execute(_execution_state(), _motion_command_values([[1.0], [2.0]]))
