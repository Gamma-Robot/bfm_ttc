"""Device-resident SONIC observation, policy, history, and command execution."""

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
from bfm_adapters.families.sonic.adapter import SonicAdapter, _order
from bfm_adapters.families.sonic.observation import (
    ACTION_DIM,
    G1_INPUT_DIM,
    HISTORY_LENGTH,
    PROPRIOCEPTION_DIM,
    QUERY_COUNT,
)

try:  # Keep the serial family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without runtime extras
    wp = None


_ROOT_ANGULAR_VELOCITY = 0
_JOINT_POSITION = 3
_JOINT_VELOCITY = _JOINT_POSITION + ACTION_DIM
_PREVIOUS_ACTION = _JOINT_VELOCITY + ACTION_DIM
_PROJECTED_GRAVITY = _PREVIOUS_ACTION + ACTION_DIM
_HISTORY_DIM = _PROJECTED_GRAVITY + 3
_JOINT_COMMAND_DIM = QUERY_COUNT * ACTION_DIM


if wp is not None:

    @wp.func
    def _safe_quat(q: wp.quat) -> wp.quat:
        norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm < 1.0e-8:
            return wp.quat(1.0, 0.0, 0.0, 0.0)
        return wp.quat(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


    @wp.func
    def _quat_conjugate(q: wp.quat) -> wp.quat:
        return wp.quat(q[0], -q[1], -q[2], -q[3])


    @wp.func
    def _quat_mul(left: wp.quat, right: wp.quat) -> wp.quat:
        w1 = left[0]
        x1 = left[1]
        y1 = left[2]
        z1 = left[3]
        w2 = right[0]
        x2 = right[1]
        y2 = right[2]
        z2 = right[3]
        return wp.quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    @wp.func
    def _rotate(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        scalar = q[0]
        xyz = wp.vec3(q[1], q[2], q[3])
        return (
            vector * (2.0 * scalar * scalar - 1.0)
            + wp.cross(xyz, vector) * (2.0 * scalar)
            + xyz * (2.0 * wp.dot(xyz, vector))
        )


    @wp.func
    def _rotate_inverse(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        return _rotate(_quat_conjugate(_safe_quat(q_raw)), vector)


    @wp.func
    def _projected_yaw_quat(q_raw: wp.quat) -> wp.quat:
        q = _safe_quat(q_raw)
        x_world = _rotate(q, wp.vec3(1.0, 0.0, 0.0))
        z_world = _rotate(q, wp.vec3(0.0, 0.0, 1.0))
        heading_x = x_world[0]
        heading_y = x_world[1]
        if wp.sqrt(heading_x * heading_x + heading_y * heading_y) <= 0.1:
            if x_world[2] < 0.0:
                heading_x = z_world[0]
                heading_y = z_world[1]
            else:
                heading_x = -z_world[0]
                heading_y = -z_world[1]
        half_yaw = 0.5 * wp.atan2(heading_y, heading_x)
        return wp.quat(wp.cos(half_yaw), 0.0, 0.0, wp.sin(half_yaw))


    @wp.func
    def _heading_offset(robot: wp.quat, reference: wp.quat) -> wp.quat:
        return _quat_mul(
            _projected_yaw_quat(robot),
            _quat_conjugate(_projected_yaw_quat(reference)),
        )


    @wp.func
    def _rotation_6d_value(q_raw: wp.quat, component: int) -> float:
        q = _safe_quat(q_raw)
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        if component == 0:
            return 1.0 - 2.0 * (y * y + z * z)
        if component == 1:
            return 2.0 * (x * y - z * w)
        if component == 2:
            return 2.0 * (x * y + z * w)
        if component == 3:
            return 1.0 - 2.0 * (x * x + z * z)
        if component == 4:
            return 2.0 * (x * z - y * w)
        return 2.0 * (y * z + x * w)


    @wp.kernel
    def _observe(
        step_index: int,
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        motion_command_root_orientation: wp.array2d(dtype=wp.quat),
        default_joint_position: wp.array(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
        history_initialized: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
        heading_initialized: wp.array(dtype=wp.int32),
        fixed_heading_offset: wp.array(dtype=wp.quat),
    ):
        lane, feature = wp.tid()
        robot_orientation = body_orientation[lane, root_body_index]
        value = float(0.0)
        if feature < _JOINT_POSITION:
            angular_velocity = _rotate_inverse(
                robot_orientation, body_angular_velocity[lane, root_body_index]
            )
            value = angular_velocity[feature]
        elif feature < _JOINT_VELOCITY:
            joint = feature - _JOINT_POSITION
            value = (
                joint_position[lane, state_joint_indices[joint]]
                - default_joint_position[joint]
            )
        elif feature < _PREVIOUS_ACTION:
            joint = feature - _JOINT_VELOCITY
            value = joint_velocity[lane, state_joint_indices[joint]]
        elif feature < _PROJECTED_GRAVITY:
            value = last_action[lane, feature - _PREVIOUS_ACTION]
        else:
            gravity = _rotate_inverse(
                robot_orientation, wp.vec3(0.0, 0.0, -1.0)
            )
            value = gravity[feature - _PROJECTED_GRAVITY]

        if history_initialized[lane] == 0:
            for slot in range(HISTORY_LENGTH):
                history[lane, slot, feature] = value
        else:
            for slot in range(HISTORY_LENGTH - 1):
                history[lane, slot, feature] = history[lane, slot + 1, feature]
            history[lane, HISTORY_LENGTH - 1, feature] = value

        if feature == 0 and heading_initialized[lane] == 0:
            fixed_heading_offset[lane] = _heading_offset(
                robot_orientation, motion_command_root_orientation[lane, step_index]
            )
            heading_initialized[lane] = 1


    @wp.kernel
    def _pack_g1_input(
        step_index: int,
        query_offsets: wp.array(dtype=wp.int32),
        motion_command_joint_position: wp.array3d(dtype=wp.float32),
        motion_command_joint_velocity: wp.array3d(dtype=wp.float32),
        motion_command_root_orientation: wp.array2d(dtype=wp.quat),
        body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        fixed_heading_offset: wp.array(dtype=wp.quat),
        g1_input: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        if feature < 2 * _JOINT_COMMAND_DIM:
            local = feature
            velocity = 0
            if feature >= _JOINT_COMMAND_DIM:
                local = feature - _JOINT_COMMAND_DIM
                velocity = 1
            query = local // ACTION_DIM
            joint = local - query * ACTION_DIM
            reference_step = step_index + query_offsets[query]
            if velocity == 0:
                g1_input[lane, feature] = motion_command_joint_position[
                    lane, reference_step, joint
                ]
            else:
                g1_input[lane, feature] = motion_command_joint_velocity[
                    lane, reference_step, joint
                ]
        else:
            local = feature - 2 * _JOINT_COMMAND_DIM
            query = local // 6
            component = local - query * 6
            reference_step = step_index + query_offsets[query]
            aligned_reference = _quat_mul(
                fixed_heading_offset[lane],
                motion_command_root_orientation[lane, reference_step],
            )
            robot = _safe_quat(body_orientation[lane, root_body_index])
            relative = _quat_mul(_quat_conjugate(robot), aligned_reference)
            g1_input[lane, feature] = _rotation_6d_value(relative, component)


    @wp.kernel
    def _pack_proprioception(
        history: wp.array3d(dtype=wp.float32),
        history_initialized: wp.array(dtype=wp.int32),
        proprioception: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        local = feature
        item_dim = 3
        history_offset = _ROOT_ANGULAR_VELOCITY
        if feature >= HISTORY_LENGTH * 3:
            local = feature - HISTORY_LENGTH * 3
            item_dim = ACTION_DIM
            history_offset = _JOINT_POSITION
        if feature >= HISTORY_LENGTH * (3 + ACTION_DIM):
            local = feature - HISTORY_LENGTH * (3 + ACTION_DIM)
            history_offset = _JOINT_VELOCITY
        if feature >= HISTORY_LENGTH * (3 + 2 * ACTION_DIM):
            local = feature - HISTORY_LENGTH * (3 + 2 * ACTION_DIM)
            history_offset = _PREVIOUS_ACTION
        if feature >= HISTORY_LENGTH * (3 + 3 * ACTION_DIM):
            local = feature - HISTORY_LENGTH * (3 + 3 * ACTION_DIM)
            item_dim = 3
            history_offset = _PROJECTED_GRAVITY
        slot = local // item_dim
        item = local - slot * item_dim
        proprioception[lane, feature] = history[
            lane, slot, history_offset + item
        ]
        if feature == 0:
            history_initialized[lane] = 1


    @wp.kernel
    def _map_action(
        action: wp.array2d(dtype=wp.float32),
        command_to_policy: wp.array(dtype=wp.int32),
        default_joint_position: wp.array(dtype=wp.float32),
        action_scale: wp.array(dtype=wp.float32),
        proportional_gain: wp.array(dtype=wp.float32),
        derivative_gain: wp.array(dtype=wp.float32),
        position_target: wp.array2d(dtype=wp.float32),
        velocity_target: wp.array2d(dtype=wp.float32),
        feedforward_torque: wp.array2d(dtype=wp.float32),
        command_kp: wp.array2d(dtype=wp.float32),
        command_kd: wp.array2d(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
    ):
        lane, command_joint = wp.tid()
        policy_joint = command_to_policy[command_joint]
        value = action[lane, policy_joint]
        position_target[lane, command_joint] = (
            default_joint_position[policy_joint]
            + value * action_scale[policy_joint]
        )
        velocity_target[lane, command_joint] = 0.0
        feedforward_torque[lane, command_joint] = 0.0
        command_kp[lane, command_joint] = proportional_gain[policy_joint]
        command_kd[lane, command_joint] = derivative_gain[policy_joint]
        last_action[lane, policy_joint] = value


@dataclass(frozen=True)
class _MotionCommandBuffers:
    joint_position: Any
    joint_velocity: Any
    root_orientation: Any


class SonicDeviceBFM:
    """Prepared SONIC bfm bound to one CUDA simulation batch."""

    def __init__(
        self,
        adapter: SonicAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("SONIC device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="SONIC",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        self._stream = wp.get_stream(self._device)

        state_joints = tuple(getattr(state_view, "joint_names"))
        state_bodies = tuple(getattr(state_view, "body_names"))
        state_joint_indices = np.asarray(
            _order(state_joints, self._config.policy_joint_names), dtype=np.int32
        )
        state_body_indices = _order(state_bodies, self._config.motion_command_body_names)
        self._root_body_index = state_body_indices[adapter._root_index]
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            dtype=np.int32,
        )
        offsets = self._config.motion_command_query_offsets
        self._max_query_offset = max(offsets)

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._query_offsets = self._array(
                np.asarray(offsets, dtype=np.int32), wp.int32
            )
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
                (self._batch_size, HISTORY_LENGTH, _HISTORY_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM), dtype=wp.float32, device=self._device
            )
            self._history_initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._heading_offset = wp.empty(
                self._batch_size, dtype=wp.quat, device=self._device
            )
            self._heading_initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._policy_inputs = {
                "g1_input": wp.empty(
                    (self._batch_size, G1_INPUT_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "proprioception": wp.empty(
                    (self._batch_size, PROPRIOCEPTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
            }
            self._policy_outputs = {
                "action": wp.empty(
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
        values = normalize_bfm_state(state, self._state_template, family="SONIC")
        history = np.concatenate(
            [
                values["root_angular_velocity"],
                values["joint_position"],
                values["joint_velocity"],
                values["previous_actions"],
                values["projected_gravity"],
            ],
            axis=2,
        )
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        with wp.ScopedStream(self._stream):
            self._history.assign(np.ascontiguousarray(history, dtype=np.float32))
            self._last_action.assign(values["last_action"])
            self._history_initialized.assign(
                values["history_initialized"].astype(np.int32)
            )
            self._heading_offset.assign(values["heading_offset"])
            self._heading_initialized.assign(
                values["heading_initialized"].astype(np.int32)
            )

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        history = np.asarray(self._history.numpy()).copy()
        return {
            "root_angular_velocity": history[
                :, :, _ROOT_ANGULAR_VELOCITY:_JOINT_POSITION
            ],
            "joint_position": history[:, :, _JOINT_POSITION:_JOINT_VELOCITY],
            "joint_velocity": history[:, :, _JOINT_VELOCITY:_PREVIOUS_ACTION],
            "previous_actions": history[
                :, :, _PREVIOUS_ACTION:_PROJECTED_GRAVITY
            ],
            "projected_gravity": history[:, :, _PROJECTED_GRAVITY:],
            "last_action": np.asarray(self._last_action.numpy()).copy(),
            "history_initialized": np.asarray(
                self._history_initialized.numpy(), dtype=bool
            ),
            "heading_offset": np.asarray(self._heading_offset.numpy()).copy(),
            "heading_initialized": np.asarray(
                self._heading_initialized.numpy(), dtype=bool
            ),
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
        context_steps = (
            0 if right_context is None else len(right_context.query_offsets)
        )
        total_steps = horizon + context_steps
        if total_steps <= self._max_query_offset:
            raise ValueError(
                f"SONIC prepared motion_command must cover offset {self._max_query_offset}"
            )

        joint_order = _order(
            motion_command.joint_names, self._config.policy_joint_names
        )
        root_index = motion_command.body_names.index(self._family.root_body)

        def combined(field: str) -> np.ndarray:
            candidate = getattr(motion_command, field)
            if right_context is None:
                return candidate
            return np.concatenate((candidate, getattr(right_context, field)), axis=1)

        joint_position = combined("joint_position")[:, :, joint_order]
        joint_velocity = combined("joint_velocity")[:, :, joint_order]
        root_orientation = combined("body_orientation_world")[:, :, root_index]

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
                    joint_velocity=wp.empty(
                        (self._batch_size, total_steps, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    root_orientation=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                )
            self._motion_command_buffers[total_steps] = buffers
        with wp.ScopedStream(self._stream):
            buffers.joint_position.assign(
                np.ascontiguousarray(joint_position, dtype=np.float32)
            )
            buffers.joint_velocity.assign(
                np.ascontiguousarray(joint_velocity, dtype=np.float32)
            )
            buffers.root_orientation.assign(
                np.ascontiguousarray(root_orientation, dtype=np.float32)
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
        if int(step_index) + self._max_query_offset >= self._total_steps:
            raise ValueError("prepared motion_command does not cover SONIC future queries")
        validate_bound_state_view(state_view, self._state_view, family="SONIC")
        buffers = self._prepared
        wp.launch(
            _observe,
            dim=(self._batch_size, _HISTORY_DIM),
            inputs=[
                int(step_index),
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                buffers.root_orientation,
                self._default_joint_position,
                self._last_action,
                self._history_initialized,
                self._history,
                self._heading_initialized,
                self._heading_offset,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_g1_input,
            dim=(self._batch_size, G1_INPUT_DIM),
            inputs=[
                int(step_index),
                self._query_offsets,
                buffers.joint_position,
                buffers.joint_velocity,
                buffers.root_orientation,
                state_view.body_orientation_world,
                self._root_body_index,
                self._heading_offset,
                self._policy_inputs["g1_input"],
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_proprioception,
            dim=(self._batch_size, PROPRIOCEPTION_DIM),
            inputs=[
                self._history,
                self._history_initialized,
                self._policy_inputs["proprioception"],
            ],
            device=self._device,
            stream=self._stream,
        )
        self._policy.enqueue(
            synchronize=int(step_index) == self._execution_steps - 1
        )
        wp.launch(
            _map_action,
            dim=(self._batch_size, ACTION_DIM),
            inputs=[
                self._policy_outputs["action"],
                self._command_to_policy,
                self._default_joint_position,
                self._action_scale,
                self._proportional_gain,
                self._derivative_gain,
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

__all__ = ["SonicDeviceBFM"]
