"""Device-resident TWIST2 observation, policy, history, and command path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bfm_adapters.contracts import DevicePDCommand, MotionCommand
from bfm_adapters.device_abi import (
    normalize_bfm_state,
    validate_bound_state_view,
    validate_device_layout,
)
from bfm_adapters.device_policy import PersistentOrtCudaPolicy
from bfm_adapters.families.twist2.adapter import Twist2Adapter, _order
from bfm_adapters.families.twist2.observation import (
    ACTION_DIM,
    CURRENT_OFFSET,
    FRAME_DIM,
    FUTURE_OFFSET,
    HISTORY_LENGTH,
    INPUT_DIM,
    MIMIC_DIM,
)

try:  # Keep the host family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without runtime extras
    wp = None


_PROPRIOCEPTION = MIMIC_DIM
_HISTORY = FRAME_DIM
_FUTURE = _HISTORY + HISTORY_LENGTH * FRAME_DIM


if wp is not None:

    @wp.func
    def _safe_quat(q: wp.quat) -> wp.quat:
        norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm < 1.0e-8:
            return wp.quat(1.0, 0.0, 0.0, 0.0)
        return wp.quat(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


    @wp.func
    def _rotate_inverse(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        scalar = q[0]
        xyz = wp.vec3(q[1], q[2], q[3])
        return (
            vector * (2.0 * scalar * scalar - 1.0)
            - wp.cross(xyz, vector) * (2.0 * scalar)
            + xyz * (2.0 * wp.dot(xyz, vector))
        )


    @wp.func
    def _roll(q_raw: wp.quat) -> float:
        q = _safe_quat(q_raw)
        return wp.atan2(
            2.0 * (q[0] * q[1] + q[2] * q[3]),
            1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]),
        )


    @wp.func
    def _pitch(q_raw: wp.quat) -> float:
        q = _safe_quat(q_raw)
        value = wp.clamp(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)
        return wp.asin(value)


    @wp.func
    def _mimic_value(
        feature: int,
        lane: int,
        step_index: int,
        joint_position: wp.array3d(dtype=wp.float32),
        root_position: wp.array2d(dtype=wp.vec3),
        root_orientation: wp.array2d(dtype=wp.quat),
        root_linear_velocity: wp.array2d(dtype=wp.vec3),
        root_angular_velocity: wp.array2d(dtype=wp.vec3),
    ) -> float:
        orientation = root_orientation[lane, step_index]
        if feature < 2:
            velocity = _rotate_inverse(
                orientation, root_linear_velocity[lane, step_index]
            )
            return velocity[feature]
        if feature == 2:
            return root_position[lane, step_index][2]
        if feature == 3:
            return _roll(orientation)
        if feature == 4:
            return _pitch(orientation)
        if feature == 5:
            velocity = _rotate_inverse(
                orientation, root_angular_velocity[lane, step_index]
            )
            return velocity[2]
        return joint_position[lane, step_index, feature - 6]


    @wp.kernel
    def _pack_input(
        step_index: int,
        current_offset: int,
        future_offset: int,
        joint_position: wp.array3d(dtype=wp.float32),
        root_position: wp.array2d(dtype=wp.vec3),
        root_orientation: wp.array2d(dtype=wp.quat),
        root_linear_velocity: wp.array2d(dtype=wp.vec3),
        root_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_position: wp.array2d(dtype=wp.float32),
        state_joint_velocity: wp.array2d(dtype=wp.float32),
        state_root_orientation: wp.array2d(dtype=wp.quat),
        state_root_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        default_joint_position: wp.array(dtype=wp.float32),
        joint_velocity_scale: wp.array(dtype=wp.float32),
        angular_velocity_scale: float,
        joint_position_scale: float,
        last_action: wp.array2d(dtype=wp.float32),
        history: wp.array3d(dtype=wp.float32),
        policy_input: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        value = float(0.0)
        if feature < _PROPRIOCEPTION:
            value = _mimic_value(
                feature,
                lane,
                step_index + current_offset,
                joint_position,
                root_position,
                root_orientation,
                root_linear_velocity,
                root_angular_velocity,
            )
        elif feature < _HISTORY:
            local = feature - _PROPRIOCEPTION
            robot_orientation = state_root_orientation[lane, root_body_index]
            if local < 3:
                angular_velocity = _rotate_inverse(
                    robot_orientation,
                    state_root_angular_velocity[lane, root_body_index],
                )
                value = angular_velocity[local] * angular_velocity_scale
            elif local == 3:
                value = _roll(robot_orientation)
            elif local == 4:
                value = _pitch(robot_orientation)
            elif local < 34:
                joint = local - 5
                state_joint = state_joint_indices[joint]
                value = (
                    state_joint_position[lane, state_joint]
                    - default_joint_position[joint]
                ) * joint_position_scale
            elif local < 63:
                joint = local - 34
                value = (
                    state_joint_velocity[lane, state_joint_indices[joint]]
                    * joint_velocity_scale[joint]
                )
            else:
                value = last_action[lane, local - 63]
        elif feature < _FUTURE:
            local = feature - _HISTORY
            slot = local // FRAME_DIM
            item = local - slot * FRAME_DIM
            value = history[lane, slot, item]
        else:
            value = _mimic_value(
                feature - _FUTURE,
                lane,
                step_index + future_offset,
                joint_position,
                root_position,
                root_orientation,
                root_linear_velocity,
                root_angular_velocity,
            )
        policy_input[lane, feature] = value


    @wp.kernel
    def _advance_history(
        policy_input: wp.array2d(dtype=wp.float32),
        fill_steps_remaining: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        if fill_steps_remaining[lane] > 0:
            for slot in range(HISTORY_LENGTH):
                history[lane, slot, feature] = policy_input[lane, feature]
        else:
            for slot in range(HISTORY_LENGTH - 1):
                history[lane, slot, feature] = history[lane, slot + 1, feature]
            history[lane, HISTORY_LENGTH - 1, feature] = policy_input[lane, feature]


    @wp.kernel
    def _decrement_fill_steps(fill_steps_remaining: wp.array(dtype=wp.int32)):
        lane = wp.tid()
        fill_steps_remaining[lane] = wp.max(fill_steps_remaining[lane] - 1, 0)


    @wp.kernel
    def _map_action(
        action: wp.array2d(dtype=wp.float32),
        command_to_policy: wp.array(dtype=wp.int32),
        default_joint_position: wp.array(dtype=wp.float32),
        action_scale: wp.array(dtype=wp.float32),
        proportional_gain: wp.array(dtype=wp.float32),
        derivative_gain: wp.array(dtype=wp.float32),
        action_clip: float,
        position_target: wp.array2d(dtype=wp.float32),
        velocity_target: wp.array2d(dtype=wp.float32),
        feedforward_torque: wp.array2d(dtype=wp.float32),
        command_kp: wp.array2d(dtype=wp.float32),
        command_kd: wp.array2d(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
    ):
        lane, command_joint = wp.tid()
        policy_joint = command_to_policy[command_joint]
        raw = action[lane, policy_joint]
        clipped = wp.clamp(raw, -action_clip, action_clip)
        position_target[lane, command_joint] = (
            default_joint_position[policy_joint]
            + clipped * action_scale[policy_joint]
        )
        velocity_target[lane, command_joint] = 0.0
        feedforward_torque[lane, command_joint] = 0.0
        command_kp[lane, command_joint] = proportional_gain[policy_joint]
        command_kd[lane, command_joint] = derivative_gain[policy_joint]
        last_action[lane, policy_joint] = raw


@dataclass(frozen=True)
class _MotionCommandBuffers:
    joint_position: Any
    root_position: Any
    root_orientation: Any
    root_linear_velocity: Any
    root_angular_velocity: Any


class Twist2DeviceBFM:
    """Prepared TWIST2 bfm bound to one CUDA simulation batch."""

    def __init__(
        self,
        adapter: Twist2Adapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("TWIST2 device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="TWIST2",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        state_joint_names = tuple(getattr(state_view, "joint_names"))
        state_body_names = tuple(getattr(state_view, "body_names"))
        if self._family.root_body not in state_body_names:
            raise ValueError("state_view does not contain the TWIST2 root body")
        self._root_body_index = state_body_names.index(self._family.root_body)
        state_joint_indices = np.asarray(
            _order(state_joint_names, self._config.policy_joint_names), np.int32
        )
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            np.int32,
        )
        velocity_scale = np.full(
            ACTION_DIM, self._family.joint_velocity_scale, dtype=np.float32
        )
        velocity_scale[adapter._ankle_indices] = 0.0
        self._stream = wp.get_stream(self._device)

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._joint_velocity_scale = self._array(velocity_scale, wp.float32)
            self._default_joint_position = self._array(
                self._family.default_joint_position, wp.float32
            )
            self._action_scale = self._array(self._family.action_scale, wp.float32)
            self._proportional_gain = self._array(
                self._family.proportional_gain, wp.float32
            )
            self._derivative_gain = self._array(
                self._family.derivative_gain, wp.float32
            )
            self._history = wp.empty(
                (self._batch_size, HISTORY_LENGTH, FRAME_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._history_fill_steps_remaining = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._policy_inputs = {
                "input": wp.empty(
                    (self._batch_size, INPUT_DIM),
                    dtype=wp.float32,
                    device=self._device,
                )
            }
            self._policy_outputs = {
                "output": wp.empty(
                    (self._batch_size, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                )
            }
        self._policy = PersistentOrtCudaPolicy(
            policy_path,
            self._config,
            input_buffers=self._policy_inputs,
            output_buffers=self._policy_outputs,
            stream=self._stream,
        )
        self._motion_command_buffers: dict[int, _MotionCommandBuffers] = {}
        self._prepared: _MotionCommandBuffers | None = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        self._state_template = adapter.initial_state(self._batch_size)

    def reset(self, state: object) -> None:
        values = normalize_bfm_state(state, self._state_template, family="TWIST2")
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        with wp.ScopedStream(self._stream):
            self._history.assign(values["history"])
            self._history_fill_steps_remaining.assign(
                values["history_fill_steps_remaining"]
            )
            self._last_action.assign(values["last_action"])

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        return {
            "history": np.asarray(self._history.numpy()).copy(),
            "history_fill_steps_remaining": np.asarray(
                self._history_fill_steps_remaining.numpy(), dtype=np.int32
            ).copy(),
            "last_action": np.asarray(self._last_action.numpy()).copy(),
        }

    def prepare(
        self,
        history: MotionCommand | None,
        motion_command: MotionCommand,
        right_context: MotionCommand | None,
        *,
        num_steps: int,
    ) -> None:
        del history
        horizon = len(motion_command.query_offsets)
        if not isinstance(num_steps, (int, np.integer)) or not (
            1 <= int(num_steps) <= horizon
        ):
            raise ValueError("num_steps must be within prepared motion_command")
        self._validate_motion_command(motion_command, "motion_command")
        if right_context is not None:
            self._validate_motion_command(right_context, "right_context")
            if (
                right_context.joint_names != motion_command.joint_names
                or right_context.body_names != motion_command.body_names
            ):
                raise ValueError("right_context names differ from motion_command")

        def combined(field: str) -> np.ndarray:
            candidate = getattr(motion_command, field)
            if right_context is None:
                return candidate
            return np.concatenate((candidate, getattr(right_context, field)), axis=1)

        total_steps = horizon + (
            0 if right_context is None else len(right_context.query_offsets)
        )
        if total_steps <= horizon + CURRENT_OFFSET - 1:
            raise ValueError(
                "TWIST2 prepared motion_command requires one right-context step"
            )
        joint_order = _order(motion_command.joint_names, self._config.policy_joint_names)
        root_index = motion_command.body_names.index(self._family.root_body)
        arrays = {
            "joint_position": combined("joint_position")[:, :, joint_order],
            "root_position": combined("body_position_world")[:, :, root_index],
            "root_orientation": combined("body_orientation_world")[:, :, root_index],
            "root_linear_velocity": combined("body_linear_velocity_world")[:, :, root_index],
            "root_angular_velocity": combined("body_angular_velocity_world")[:, :, root_index],
        }

        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        buffers = self._motion_command_buffers.get(total_steps)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _MotionCommandBuffers(
                    joint_position=wp.empty(
                        (self._batch_size, total_steps, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    root_position=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_orientation=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    root_linear_velocity=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_angular_velocity=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                )
            self._motion_command_buffers[total_steps] = buffers
        with wp.ScopedStream(self._stream):
            for name, value in arrays.items():
                getattr(buffers, name).assign(
                    np.ascontiguousarray(value, dtype=np.float32)
                )
        self._prepared = buffers
        self._horizon = horizon
        self._execution_steps = int(num_steps)
        self._total_steps = total_steps

    def step_prepared(
        self, state_view: object, step_index: int
    ) -> DevicePDCommand:
        if self._prepared is None:
            raise RuntimeError("prepare() must be called before step_prepared()")
        if not isinstance(step_index, (int, np.integer)) or not (
            0 <= int(step_index) < self._horizon
        ):
            raise ValueError("step_index is outside prepared motion_command")
        if int(step_index) + CURRENT_OFFSET >= self._total_steps:
            raise ValueError("prepared motion_command lacks TWIST2 current-motion context")
        validate_bound_state_view(state_view, self._state_view, family="TWIST2")
        buffers = self._prepared
        wp.launch(
            _pack_input,
            dim=(self._batch_size, INPUT_DIM),
            inputs=[
                int(step_index),
                CURRENT_OFFSET,
                FUTURE_OFFSET,
                buffers.joint_position,
                buffers.root_position,
                buffers.root_orientation,
                buffers.root_linear_velocity,
                buffers.root_angular_velocity,
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                self._default_joint_position,
                self._joint_velocity_scale,
                self._family.angular_velocity_scale,
                self._family.joint_position_scale,
                self._last_action,
                self._history,
                self._policy_inputs["input"],
            ],
            device=self._device,
            stream=self._stream,
        )
        self._policy.enqueue(
            synchronize=int(step_index) == self._execution_steps - 1
        )
        wp.launch(
            _advance_history,
            dim=(self._batch_size, FRAME_DIM),
            inputs=[
                self._policy_inputs["input"],
                self._history_fill_steps_remaining,
                self._history,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _decrement_fill_steps,
            dim=self._batch_size,
            inputs=[self._history_fill_steps_remaining],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _map_action,
            dim=(self._batch_size, ACTION_DIM),
            inputs=[
                self._policy_outputs["output"],
                self._command_to_policy,
                self._default_joint_position,
                self._action_scale,
                self._proportional_gain,
                self._derivative_gain,
                self._family.action_clip,
                self._command.position_target,
                self._command.velocity_target,
                self._command.feedforward_torque,
                self._command.proportional_gain,
                self._command.derivative_gain,
                self._last_action,
            ],
            device=self._device,
            stream=self._stream,
        )
        return self._command

    def close(self) -> None:
        self._policy.close()

    def _array(self, value: np.ndarray, dtype: Any) -> Any:
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self._device)

    def _validate_motion_command(self, motion_command: MotionCommand, label: str) -> None:
        steps = len(motion_command.query_offsets)
        if motion_command.query_offsets != tuple(range(steps)):
            raise ValueError(f"{label} must be dense and zero-based")
        if motion_command.joint_position.shape[0] != self._batch_size:
            raise ValueError(f"{label} batch size differs from the device bfm")


__all__ = ["Twist2DeviceBFM"]
