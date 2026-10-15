"""Stateful bfm execution over host observations and one batched policy call."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from bfm_adapters.adapter import BFMAdapter
from bfm_adapters.contracts import (
    DevicePDCommand,
    MotionCommand,
    NamedTensors,
    PDCommand,
    RobotState,
    BFMStateBatch,
)


@runtime_checkable
class RobotStateView(Protocol):
    """Canonical physical arrays exposed by a simulation world.

    Arrays may live on a device, but must either support ``numpy()`` or be
    convertible with ``numpy.asarray``. Quaternion fields use ``(w, x, y, z)``.
    """

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_position: object
    joint_velocity: object
    body_position_world: object
    body_orientation_world: object
    body_linear_velocity_world: object
    body_angular_velocity_world: object


@runtime_checkable
class BatchedBFM(Protocol):
    """Stateful pretrained bfm; the caller owns physics and applies commands."""

    def reset(self, state: BFMStateBatch) -> None:
        """Restore an independently owned bfm-state copy."""

    def capture_state(self) -> BFMStateBatch:
        """Return an independent copy of the current bfm state."""

    def step(
        self, state_view: RobotStateView, motion_command: MotionCommand
    ) -> PDCommand:
        """Advance bfm state by one control step and return a PD command."""


@runtime_checkable
class PreparedDeviceBFM(Protocol):
    """Optional device runtime that binds a complete motion_command segment once."""

    def reset(self, state: BFMStateBatch) -> None:
        """Complete queued policy work, then restore an owned bfm-state copy."""

    def capture_state(self) -> BFMStateBatch:
        """Return an independent copy of the current bfm state."""

    def prepare(
        self,
        history: MotionCommand | None,
        motion_command: MotionCommand,
        right_context: MotionCommand | None,
        *,
        num_steps: int,
    ) -> None:
        """Bind motion_command and the exact number of control steps to execute."""

    def step_prepared(
        self, state_view: RobotStateView, step_index: int
    ) -> DevicePDCommand:
        """Advance one control step using the motion_command bound by ``prepare``."""

    def close(self) -> None:
        """Release policy runtime resources after all queued work completes."""


@runtime_checkable
class DeviceBFMProvider(Protocol):
    """Optional adapter capability for one persistent device bfm."""

    def build_device_bfm(
        self,
        policy_path: str | Path,
        state_view: RobotStateView,
        command_view: DevicePDCommand,
    ) -> PreparedDeviceBFM:
        """Bind family semantics to one fixed simulation batch."""


class _BatchedPolicy(Protocol):
    def run(self, inputs: NamedTensors) -> NamedTensors: ...


class SerialReferenceBFM:
    """Host-NumPy bfm path used as the reference implementation.

    The family adapter owns observation, action, and opaque-state semantics.
    Only policy inference is delegated, once per control step for the complete batch.
    State advances atomically after policy execution and output mapping succeed.
    """

    def __init__(
        self,
        adapter: BFMAdapter,
        policy: _BatchedPolicy,
        *,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._adapter = adapter
        self._policy = policy
        self._state = adapter.initial_state(batch_size)

    def reset(self, state: BFMStateBatch) -> None:
        self._state = self._adapter.clone_state(state)

    def capture_state(self) -> BFMStateBatch:
        return self._adapter.clone_state(self._state)

    def step(
        self, state_view: RobotStateView, motion_command: MotionCommand
    ) -> PDCommand:
        state = _copy_to_host(state_view)
        batch_size = state.joint_position.shape[0]
        if motion_command.joint_position.shape[0] != batch_size:
            raise ValueError("RobotStateView and MotionCommand batch sizes differ")

        inputs, observed_state = self._adapter.build_policy_inputs(
            state, motion_command, self._state
        )
        outputs = self._policy.run(inputs)
        command, next_state = self._adapter.map_policy_outputs(
            outputs, observed_state
        )
        if command.position_target.shape[0] != batch_size:
            raise ValueError("bfm command batch size differs from RobotStateView")
        self._state = next_state
        return command


def _copy_to_host(view: RobotStateView) -> RobotState:
    if isinstance(view, RobotState):
        return view

    def array(field: str) -> np.ndarray:
        value = getattr(view, field)
        to_numpy = getattr(value, "numpy", None)
        host = to_numpy() if callable(to_numpy) else value
        return np.asarray(host).copy()

    return RobotState(
        joint_names=tuple(view.joint_names),
        body_names=tuple(view.body_names),
        joint_position=array("joint_position"),
        joint_velocity=array("joint_velocity"),
        body_position_world=array("body_position_world"),
        body_orientation_world=array("body_orientation_world"),
        body_linear_velocity_world=array("body_linear_velocity_world"),
        body_angular_velocity_world=array("body_angular_velocity_world"),
    )
