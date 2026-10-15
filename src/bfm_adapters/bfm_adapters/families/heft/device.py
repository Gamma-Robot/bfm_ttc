"""Device-resident HEFT observation, history, policy, and PD command path."""

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
from bfm_adapters.families.heft.adapter import HeftAdapter, _order
from bfm_adapters.families.heft.observation import (
    ACTION_DIM,
    CONTEXT_DIM,
    HISTORY_LAGS,
    HISTORY_LENGTH,
    MOTION_COMMAND_DIM,
    PREVIOUS_ACTIONS,
    PROPRIOCEPTION_DIM,
    QUERY_OFFSETS,
    TARGET_MOTION_DIM,
)

try:  # Keep the serial family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without runtime extras
    wp = None


_ANGULAR_VELOCITY = 0
_GRAVITY = 3
_JOINT_POSITION = 6
_JOINT_VELOCITY = 35
_HISTORY_DIM = 64
_PAST_STEPS = max(0, -min(QUERY_OFFSETS))
_FUTURE_STEPS = max(0, max(QUERY_OFFSETS))


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
        imaginary = wp.vec3(q[1], q[2], q[3])
        return (
            vector * (2.0 * scalar * scalar - 1.0)
            - wp.cross(imaginary, vector) * (2.0 * scalar)
            + imaginary * (2.0 * wp.dot(imaginary, vector))
        )


    @wp.func
    def _relative_quat(anchor_raw: wp.quat, target: wp.quat) -> wp.quat:
        anchor = _safe_quat(anchor_raw)
        w1 = anchor[0]
        x1 = -anchor[1]
        y1 = -anchor[2]
        z1 = -anchor[3]
        w2 = target[0]
        x2 = target[1]
        y2 = target[2]
        z2 = target[3]
        return wp.quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    @wp.func
    def _rotation_6d_value(q: wp.quat, component: int) -> float:
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        two_s = 2.0 / (w * w + x * x + y * y + z * z)
        if component == 0:
            return 1.0 - two_s * (y * y + z * z)
        if component == 1:
            return two_s * (x * y + z * w)
        if component == 2:
            return two_s * (x * z - y * w)
        if component == 3:
            return two_s * (x * y - z * w)
        if component == 4:
            return 1.0 - two_s * (x * x + z * z)
        return two_s * (y * z + x * w)


    @wp.kernel
    def _update_state(
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        last_action: wp.array2d(dtype=wp.float32),
        boot: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
        previous_actions: wp.array3d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        root_orientation = body_orientation[lane, root_body_index]
        value = float(0.0)
        if feature < _GRAVITY:
            angular_velocity = _rotate_inverse(
                root_orientation, body_angular_velocity[lane, root_body_index]
            )
            value = angular_velocity[feature]
        elif feature < _JOINT_POSITION:
            component = feature - _GRAVITY
            gravity = _rotate_inverse(root_orientation, wp.vec3(0.0, 0.0, -1.0))
            norm = wp.sqrt(wp.dot(gravity, gravity))
            if norm > 1.0e-8:
                value = gravity[component] / norm
        elif feature < _JOINT_VELOCITY:
            policy_joint = feature - _JOINT_POSITION
            value = joint_position[lane, state_joint_indices[policy_joint]]
        else:
            policy_joint = feature - _JOINT_VELOCITY
            value = joint_velocity[lane, state_joint_indices[policy_joint]]

        for index in range(HISTORY_LENGTH - 1):
            slot = HISTORY_LENGTH - 1 - index
            history[lane, slot, feature] = history[lane, slot - 1, feature]
        history[lane, 0, feature] = value

        if feature < ACTION_DIM:
            for index in range(PREVIOUS_ACTIONS - 1):
                slot = PREVIOUS_ACTIONS - 1 - index
                previous_actions[lane, slot, feature] = previous_actions[
                    lane, slot - 1, feature
                ]
            previous_actions[lane, 0, feature] = last_action[lane, feature]
        if feature == 0 and boot[lane] > 0:
            boot[lane] = boot[lane] - 1


    @wp.kernel
    def _pack_inputs(
        step_index: int,
        timeline_start: int,
        query_offsets: wp.array(dtype=wp.int32),
        history_lags: wp.array(dtype=wp.int32),
        state_body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        motion_command_joint_position: wp.array3d(dtype=wp.float32),
        motion_command_root_position: wp.array2d(dtype=wp.vec3),
        motion_command_root_orientation: wp.array2d(dtype=wp.quat),
        boot: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
        previous_actions: wp.array3d(dtype=wp.float32),
        boot_indicator_max: int,
        context: wp.array2d(dtype=wp.float32),
        motion_command: wp.array2d(dtype=wp.float32),
        target_motion: wp.array2d(dtype=wp.float32),
        proprioception: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        origin = timeline_start + step_index

        if feature < CONTEXT_DIM:
            context[lane, feature] = float(boot[lane]) / float(boot_indicator_max)

        if feature < MOTION_COMMAND_DIM:
            value = float(0.0)
            if feature < 36:
                query = feature // 3 + 1
                component = feature - (query - 1) * 3
                origin_position = motion_command_root_position[
                    lane, origin + query_offsets[0]
                ]
                target_position = motion_command_root_position[
                    lane, origin + query_offsets[query]
                ]
                anchor_orientation = motion_command_root_orientation[
                    lane, origin + query_offsets[0]
                ]
                delta = _rotate_inverse(
                    anchor_orientation, target_position - origin_position
                )
                value = delta[component]
            else:
                local = feature - 36
                query = local // 6
                component = local - query * 6
                relative = _relative_quat(
                    state_body_orientation[lane, root_body_index],
                    motion_command_root_orientation[lane, origin + query_offsets[query]],
                )
                value = _rotation_6d_value(relative, component)
            motion_command[lane, feature] = value

        if feature < TARGET_MOTION_DIM:
            value = float(0.0)
            if feature < 13 * ACTION_DIM:
                query = feature // ACTION_DIM
                joint = feature - query * ACTION_DIM
                value = motion_command_joint_position[
                    lane, origin + query_offsets[query], joint
                ]
            elif feature < 26 * ACTION_DIM:
                local = feature - 13 * ACTION_DIM
                query = local // ACTION_DIM
                joint = local - query * ACTION_DIM
                value = (
                    motion_command_joint_position[
                        lane, origin + query_offsets[query], joint
                    ]
                    - history[lane, 0, _JOINT_POSITION + joint]
                )
            elif feature < 26 * ACTION_DIM + 13:
                query = feature - 26 * ACTION_DIM
                value = motion_command_root_position[
                    lane, origin + query_offsets[query]
                ][2]
            else:
                local = feature - (26 * ACTION_DIM + 13)
                query = local // 3
                component = local - query * 3
                gravity = _rotate_inverse(
                    motion_command_root_orientation[
                        lane, origin + query_offsets[query]
                    ],
                    wp.vec3(0.0, 0.0, -1.0),
                )
                value = gravity[component]
            target_motion[lane, feature] = value

        value = float(0.0)
        if feature < 9 * 3:
            lag_index = feature // 3
            component = feature - lag_index * 3
            value = history[
                lane, history_lags[lag_index], _ANGULAR_VELOCITY + component
            ]
        elif feature < 18 * 3:
            local = feature - 9 * 3
            lag_index = local // 3
            component = local - lag_index * 3
            value = history[lane, history_lags[lag_index], _GRAVITY + component]
        elif feature < 18 * 3 + 9 * ACTION_DIM:
            local = feature - 18 * 3
            lag_index = local // ACTION_DIM
            joint = local - lag_index * ACTION_DIM
            value = history[
                lane, history_lags[lag_index], _JOINT_POSITION + joint
            ]
        elif feature < 18 * 3 + 18 * ACTION_DIM:
            local = feature - (18 * 3 + 9 * ACTION_DIM)
            lag_index = local // ACTION_DIM
            joint = local - lag_index * ACTION_DIM
            value = history[
                lane, history_lags[lag_index], _JOINT_VELOCITY + joint
            ]
        else:
            local = feature - (18 * 3 + 18 * ACTION_DIM)
            action_slot = local // ACTION_DIM
            joint = local - action_slot * ACTION_DIM
            value = previous_actions[lane, action_slot, joint]
        proprioception[lane, feature] = value


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
        value = wp.clamp(action[lane, policy_joint], -action_clip, action_clip)
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
    root_position: Any
    root_orientation: Any
    length: int


class HeftDeviceBFM:
    """Prepared HEFT bfm whose per-Control-Step path stays on one CUDA stream."""

    def __init__(
        self,
        adapter: HeftAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("HEFT device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="HEFT",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        if self._family.root_body not in tuple(getattr(state_view, "body_names")):
            raise ValueError("state_view does not contain the HEFT root body")
        self._stream = wp.get_stream(self._device)

        state_joint_names = tuple(getattr(state_view, "joint_names"))
        state_body_names = tuple(getattr(state_view, "body_names"))
        state_joint_indices = np.asarray(
            _order(state_joint_names, self._config.policy_joint_names), np.int32
        )
        root_body_index = state_body_names.index(self._family.root_body)
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            np.int32,
        )

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._root_body_index = root_body_index
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._query_offsets = self._array(
                np.asarray(QUERY_OFFSETS, np.int32), wp.int32
            )
            self._history_lags = self._array(
                np.asarray(HISTORY_LAGS, np.int32), wp.int32
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
            self._boot = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._history = wp.empty(
                (self._batch_size, HISTORY_LENGTH, _HISTORY_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._previous_actions = wp.empty(
                (self._batch_size, PREVIOUS_ACTIONS, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._policy_inputs = {
                "context": wp.empty(
                    (self._batch_size, CONTEXT_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "motion_command": wp.empty(
                    (self._batch_size, MOTION_COMMAND_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "target_motion": wp.empty(
                    (self._batch_size, TARGET_MOTION_DIM),
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
        self._state_template = adapter.initial_state(self._batch_size)

    def reset(self, state: object) -> None:
        values = normalize_bfm_state(state, self._state_template, family="HEFT")
        history = np.concatenate(
            (
                values["root_angular_velocity"],
                values["projected_gravity"],
                values["joint_position"],
                values["joint_velocity"],
            ),
            axis=2,
        )
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        with wp.ScopedStream(self._stream):
            self._boot.assign(values["boot"])
            self._history.assign(np.ascontiguousarray(history, dtype=np.float32))
            self._previous_actions.assign(values["previous_actions"])
            self._last_action.assign(values["last_action"])

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        history = np.asarray(self._history.numpy()).copy()
        return {
            "boot": np.asarray(self._boot.numpy(), dtype=np.int32).copy(),
            "root_angular_velocity": history[:, :, _ANGULAR_VELOCITY:_GRAVITY],
            "projected_gravity": history[:, :, _GRAVITY:_JOINT_POSITION],
            "joint_position": history[:, :, _JOINT_POSITION:_JOINT_VELOCITY],
            "joint_velocity": history[:, :, _JOINT_VELOCITY:],
            "previous_actions": np.asarray(self._previous_actions.numpy()).copy(),
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
        horizon = len(motion_command.query_offsets)
        if not isinstance(num_steps, (int, np.integer)) or not (
            1 <= int(num_steps) <= horizon
        ):
            raise ValueError("num_steps must be within prepared motion_command")
        if motion_command.query_offsets != tuple(range(horizon)):
            raise ValueError("HEFT prepared motion_command must be dense and zero-based")
        if history is None or len(history.query_offsets) < _PAST_STEPS:
            raise ValueError(
                f"HEFT prepared motion_command requires {_PAST_STEPS} history steps"
            )
        history_steps = len(history.query_offsets)
        if history.query_offsets != tuple(range(-history_steps, 0)):
            raise ValueError("HEFT history must use a dense negative query grid")
        self._validate_motion_command(history, motion_command, "history")
        if right_context is not None:
            context_steps = len(right_context.query_offsets)
            if right_context.query_offsets != tuple(range(context_steps)):
                raise ValueError("HEFT right_context must be dense and zero-based")
            self._validate_motion_command(right_context, motion_command, "right_context")

        joint_order = _order(motion_command.joint_names, self._config.policy_joint_names)
        root_column = motion_command.body_names.index(self._family.root_body)
        future = 0 if right_context is None else min(
            _FUTURE_STEPS, len(right_context.query_offsets)
        )
        parts: list[tuple[MotionCommand, slice]] = [
            (history, slice(history_steps - _PAST_STEPS, history_steps)),
            (motion_command, slice(0, horizon)),
        ]
        if future:
            assert right_context is not None
            parts.append((right_context, slice(0, future)))
        joint_position = np.concatenate(
            [
                source.joint_position[:, section, :][:, :, joint_order]
                for source, section in parts
            ],
            axis=1,
        )
        root_position = np.concatenate(
            [
                source.body_position_world[:, section, root_column]
                for source, section in parts
            ],
            axis=1,
        )
        root_orientation = np.concatenate(
            [
                source.body_orientation_world[:, section, root_column]
                for source, section in parts
            ],
            axis=1,
        )
        length = joint_position.shape[1]

        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        buffers = self._motion_command_buffers.get(length)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _MotionCommandBuffers(
                    joint_position=wp.empty(
                        (self._batch_size, length, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    root_position=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_orientation=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    length=length,
                )
            self._motion_command_buffers[length] = buffers
        with wp.ScopedStream(self._stream):
            buffers.joint_position.assign(
                np.ascontiguousarray(joint_position, dtype=np.float32)
            )
            buffers.root_position.assign(
                np.ascontiguousarray(root_position, dtype=np.float32)
            )
            buffers.root_orientation.assign(
                np.ascontiguousarray(root_orientation, dtype=np.float32)
            )
        self._prepared = buffers
        self._horizon = horizon
        self._execution_steps = int(num_steps)

    def step_prepared(
        self, state_view: object, step_index: int
    ) -> DevicePDCommand:
        if self._prepared is None:
            raise RuntimeError("prepare() must be called before step_prepared()")
        if not isinstance(step_index, (int, np.integer)) or not (
            0 <= int(step_index) < self._horizon
        ):
            raise ValueError("step_index is outside prepared motion_command")
        if (
            _PAST_STEPS + int(step_index) + _FUTURE_STEPS
            >= self._prepared.length
        ):
            raise ValueError(
                "prepared motion_command lacks HEFT future context for this step"
            )
        validate_bound_state_view(state_view, self._state_view, family="HEFT")
        buffers = self._prepared
        wp.launch(
            _update_state,
            dim=(self._batch_size, _HISTORY_DIM),
            inputs=[
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                self._last_action,
                self._boot,
                self._history,
                self._previous_actions,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_inputs,
            dim=(self._batch_size, PROPRIOCEPTION_DIM),
            inputs=[
                int(step_index),
                _PAST_STEPS,
                self._query_offsets,
                self._history_lags,
                state_view.body_orientation_world,
                self._root_body_index,
                buffers.joint_position,
                buffers.root_position,
                buffers.root_orientation,
                self._boot,
                self._history,
                self._previous_actions,
                self._family.boot_indicator_max,
                self._policy_inputs["context"],
                self._policy_inputs["motion_command"],
                self._policy_inputs["target_motion"],
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

    def _validate_motion_command(
        self,
        value: MotionCommand,
        reference: MotionCommand,
        label: str,
    ) -> None:
        if (
            value.joint_position.shape[0] != self._batch_size
            or value.joint_names != reference.joint_names
            or value.body_names != reference.body_names
        ):
            raise ValueError(f"HEFT {label} must match motion_command batch and names")

    def _array(self, value: np.ndarray, dtype: Any) -> Any:
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self._device)

__all__ = ["HeftDeviceBFM"]
