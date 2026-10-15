"""The single motion_command -> bfm -> PD -> simulation execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bfm_adapters import (
    DevicePDCommand,
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
    PDCommand,
    PreparedDeviceBFM,
)
from bfm_adapters.batched_bfm import BatchedBFM, RobotStateView
from command_optimization.simulation.measurements import (
    MeasurementBuffer,
    MeasurementPrimitive,
    ControlStepMeasurements,
)
from command_optimization.simulation.mjwarp import MjwarpWorld
from command_optimization.simulation.execution_state import ExecutionState


def _owned(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class ExecutionResult:
    """Measurements, physical trace, and restartable state after one segment."""

    measurements: ControlStepMeasurements
    qpos: np.ndarray
    qvel: np.ndarray
    final_execution_state: ExecutionState


class RolloutReducer(Protocol):
    """Consume post-step device state and return one compact rollout result."""

    def record(self, step_index: int, state_view: RobotStateView) -> None: ...

    def finalize(self) -> np.ndarray: ...


class CommandExecutor:
    """Execute a dense candidate window through one bfm and one world.

    BFM queries may be signed and ordered arbitrarily. Past queries read
    the execution history carried by the execution state; queries beyond the
    candidate window read an explicit reference_motion right context. Neither context
    extends the number of simulated or evaluated Candidate Rollout Control Steps.
    """

    def __init__(
        self,
        world: MjwarpWorld,
        bfm: BatchedBFM | PreparedDeviceBFM,
        *,
        control_period: float,
        motion_command_query_offsets: tuple[int, ...],
    ) -> None:
        period = float(control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be finite and positive")
        ratio = period / float(world.physics_timestep)
        substeps = int(round(ratio))
        if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError(
                "control_period must be an integer multiple of physics_timestep"
            )
        offsets = tuple(motion_command_query_offsets)
        if (
            not offsets
            or any(not isinstance(offset, int) for offset in offsets)
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError("motion_command_query_offsets must be unique integers")
        self._world = world
        self._bfm = bfm
        self._control_period = period
        self._substeps = substeps
        self._query_offsets = offsets
        self._past_context_steps = max(0, -min(offsets))
        self._future_context_steps = max(0, max(offsets))

    @property
    def physics_substeps(self) -> int:
        """Number of physics increments in one control step."""
        return self._substeps

    @property
    def past_context_steps(self) -> int:
        """Executed motion_command frames required before a candidate window."""
        return self._past_context_steps

    @property
    def future_context_steps(self) -> int:
        """Nominal motion_command frames required after a full candidate window."""
        return self._future_context_steps

    def execute(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        *,
        num_steps: int | None = None,
        right_context: MotionCommand | None = None,
        requires: frozenset[MeasurementPrimitive] = frozenset(MeasurementPrimitive),
    ) -> ExecutionResult:
        """Execute and retain the restart state and physical trace."""
        measurements, qpos_trace, qvel_trace, command = self._run(
            execution_state,
            motion_command,
            num_steps=num_steps,
            right_context=right_context,
            requires=requires,
            capture_trace=True,
        )
        assert measurements is not None
        steps = measurements.num_steps
        qpos, qvel, times = self._world.physical_state()
        if not np.allclose(times, times[0], rtol=0.0, atol=1e-6):
            raise RuntimeError("simulation lanes disagree on time")
        host_command = (
            self._world.host_command()
            if isinstance(command, DevicePDCommand)
            else command
        )
        final_execution_state = ExecutionState(
            qpos=_owned(qpos),
            qvel=_owned(qvel),
            qacc_warmstart=self._world.warmstart_acceleration(),
            control_step=execution_state.control_step + steps,
            time=float(times[0]),
            motion_command_index=execution_state.motion_command_index + steps,
            command=host_command,
            bfm_state=self._bfm.capture_state(),
            motion_command_history=self._advance_history(
                execution_state.motion_command_history, motion_command, steps
            ),
        )
        return ExecutionResult(
            measurements=measurements,
            qpos=_owned(np.stack(qpos_trace, axis=1)),
            qvel=_owned(np.stack(qvel_trace, axis=1)),
            final_execution_state=final_execution_state,
        )

    def rollout(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        *,
        num_steps: int | None = None,
        right_context: MotionCommand | None = None,
        requires: frozenset[MeasurementPrimitive] = frozenset(MeasurementPrimitive),
    ) -> ControlStepMeasurements:
        """Evaluate a Candidate Rollout without downloading restart or traces."""
        measurements, _, _, _ = self._run(
            execution_state,
            motion_command,
            num_steps=num_steps,
            right_context=right_context,
            requires=requires,
            capture_trace=False,
        )
        assert measurements is not None
        return measurements

    def rollout_reduced(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        reducer: RolloutReducer,
        *,
        num_steps: int | None = None,
        right_context: MotionCommand | None = None,
    ) -> np.ndarray:
        """Roll out into a compact device reducer without collecting a host trace."""
        measurements, _, _, _ = self._run(
            execution_state,
            motion_command,
            num_steps=num_steps,
            right_context=right_context,
            requires=frozenset(),
            capture_trace=False,
            reducer=reducer,
        )
        assert measurements is None
        return reducer.finalize()

    def _run(
        self,
        execution_state: ExecutionState,
        motion_command: MotionCommand,
        *,
        num_steps: int | None,
        right_context: MotionCommand | None,
        requires: frozenset[MeasurementPrimitive],
        capture_trace: bool,
        reducer: RolloutReducer | None = None,
    ) -> tuple[
        ControlStepMeasurements | None,
        list[np.ndarray],
        list[np.ndarray],
        PDCommand | DevicePDCommand | None,
    ]:
        horizon = len(motion_command.query_offsets)
        if num_steps is not None and (
            not isinstance(num_steps, (int, np.integer)) or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer")
        steps = horizon if num_steps is None else int(num_steps)
        if steps > horizon:
            raise ValueError("num_steps cannot exceed the candidate motion_command horizon")
        if motion_command.query_offsets != tuple(range(horizon)):
            raise ValueError("motion_command must use a dense zero-based query grid")
        if motion_command.joint_position.shape[0] != self._world.batch_size:
            raise ValueError("motion_command batch size must match the world")
        if (
            motion_command.joint_names != self._world.joint_names
            or motion_command.body_names != self._world.body_names
        ):
            raise ValueError("motion_command joint/body names must match the world")
        self._validate_history(execution_state, motion_command)
        self._validate_right_context(right_context, motion_command, steps)

        self._world.reset(execution_state)
        self._bfm.reset(execution_state.bfm_state)
        prepared = isinstance(self._bfm, PreparedDeviceBFM)
        if prepared:
            self._bfm.prepare(
                execution_state.motion_command_history,
                motion_command,
                right_context,
                num_steps=steps,
            )
        buffer = (
            None
            if reducer is not None
            else MeasurementBuffer(
                frozenset(requires),
                self._world.joint_names,
                self._world.body_names,
                self._world.batch_size,
                steps,
            )
        )
        qpos_trace: list[np.ndarray] = []
        qvel_trace: list[np.ndarray] = []
        command = execution_state.command
        state_view = None if prepared else self._world.state_view
        for step_index in range(steps):
            if prepared:
                command = self._bfm.step_prepared(
                    self._world.state_view, step_index
                )
            else:
                assert state_view is not None
                control_step_motion_command = self._slice_motion_command(
                    execution_state.motion_command_history,
                    motion_command,
                    right_context,
                    step_index,
                )
                command = self._bfm.step(state_view, control_step_motion_command)
            self._world.apply_pd(command)
            self._world.step(self._substeps)
            if reducer is not None:
                reducer.record(step_index, self._world.state_view)
            if buffer is not None or not prepared:
                host_state = self._world.robot_state()
                if buffer is not None:
                    buffer.record(step_index, host_state)
                if not prepared:
                    state_view = host_state
            if capture_trace:
                qpos, qvel, _ = self._world.physical_state()
                qpos_trace.append(qpos)
                qvel_trace.append(qvel)

        measurements = None if buffer is None else buffer.finalize()
        return measurements, qpos_trace, qvel_trace, command

    def _validate_history(
        self, execution_state: ExecutionState, motion_command: MotionCommand
    ) -> None:
        if self._past_context_steps == 0:
            return
        history = execution_state.motion_command_history
        if history is None or len(history.query_offsets) < self._past_context_steps:
            raise ValueError(
                "execution_state motion_command_history must contain at least "
                f"{self._past_context_steps} steps"
            )
        if (
            history.joint_position.shape[0] != self._world.batch_size
            or history.joint_names != motion_command.joint_names
            or history.body_names != motion_command.body_names
        ):
            raise ValueError("motion_command_history must match motion_command batch and names")

    def _validate_right_context(
        self,
        right_context: MotionCommand | None,
        motion_command: MotionCommand,
        steps: int,
    ) -> None:
        required = max(
            0,
            steps + self._future_context_steps - len(motion_command.query_offsets),
        )
        if required == 0:
            return
        if right_context is None or len(right_context.query_offsets) < required:
            raise ValueError(
                f"right_context must contain at least {required} reference_motion steps"
            )
        context_steps = len(right_context.query_offsets)
        if right_context.query_offsets != tuple(range(context_steps)):
            raise ValueError("right_context must use a dense zero-based query grid")
        if (
            right_context.joint_position.shape[0] != self._world.batch_size
            or right_context.joint_names != motion_command.joint_names
            or right_context.body_names != motion_command.body_names
        ):
            raise ValueError("right_context must match motion_command batch and names")

    def _slice_motion_command(
        self,
        history: MotionCommand | None,
        motion_command: MotionCommand,
        right_context: MotionCommand | None,
        step_index: int,
    ) -> MotionCommand:
        sources: list[tuple[MotionCommand, int]] = []
        horizon = len(motion_command.query_offsets)
        for offset in self._query_offsets:
            index = step_index + offset
            if index < 0:
                assert history is not None
                sources.append((history, len(history.query_offsets) + index))
            elif index < horizon:
                sources.append((motion_command, index))
            else:
                assert right_context is not None
                sources.append((right_context, index - horizon))
        return MotionCommand(
            query_offsets=self._query_offsets,
            joint_names=motion_command.joint_names,
            body_names=motion_command.body_names,
            **{
                name: np.stack(
                    [getattr(source, name)[:, index] for source, index in sources],
                    axis=1,
                )
                for name in _COMMAND_ARRAYS
            },
        )

    def _advance_history(
        self,
        history: MotionCommand | None,
        motion_command: MotionCommand,
        steps: int,
    ) -> MotionCommand | None:
        if self._past_context_steps == 0:
            return None
        assert history is not None
        arrays = {}
        for name in _COMMAND_ARRAYS:
            combined = np.concatenate(
                (getattr(history, name), getattr(motion_command, name)[:, :steps]), axis=1
            )
            arrays[name] = _owned(combined[:, -self._past_context_steps :])
        return MotionCommand(
            query_offsets=tuple(range(-self._past_context_steps, 0)),
            joint_names=motion_command.joint_names,
            body_names=motion_command.body_names,
            **arrays,
        )
