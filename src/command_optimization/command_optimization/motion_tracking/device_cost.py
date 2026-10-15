"""Device-resident reduction for the motion-tracking rollout objective."""

from __future__ import annotations

from typing import Any

import numpy as np

from bfm_adapters import MOTION_COMMAND_ARRAY_FIELDS, MotionCommand
from bfm_adapters.batched_bfm import RobotStateView
from command_optimization.motion_tracking.objective import (
    MotionTrackingObjective,
    _deformation_terms,
)

try:  # Keep CPU-only imports usable.
    import warp as wp
except ImportError:  # pragma: no cover - runtime construction reports the error
    wp = None


if wp is not None:

    @wp.kernel
    def _accumulate_motion_tracking_cost(
        step_index: int,
        horizon: int,
        joint_count: int,
        body_count: int,
        body_indices: wp.array(dtype=wp.int32),
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        reference_joint_position: wp.array2d(dtype=wp.float64),
        reference_joint_velocity: wp.array2d(dtype=wp.float64),
        reference_body_position: wp.array3d(dtype=wp.float64),
        reference_body_orientation: wp.array3d(dtype=wp.float64),
        joint_position_weight: wp.float64,
        joint_velocity_weight: wp.float64,
        body_position_weight: wp.float64,
        body_orientation_weight: wp.float64,
        joint_speed_weight: wp.float64,
        terminal_joint_weight: wp.float64,
        terminal_body_weight: wp.float64,
        total_cost: wp.array(dtype=wp.float64),
        invalid: wp.array(dtype=wp.int32),
    ):
        lane = wp.tid()
        joint_position_error = wp.float64(0.0)
        joint_velocity_error = wp.float64(0.0)
        joint_speed = wp.float64(0.0)
        body_position_error = wp.float64(0.0)
        body_orientation_error = wp.float64(0.0)
        valid = int(1)

        for joint in range(joint_count):
            joint_position_value = wp.float64(joint_position[lane, joint])
            joint_velocity_value = wp.float64(joint_velocity[lane, joint])
            target_position = reference_joint_position[step_index, joint]
            target_velocity = reference_joint_velocity[step_index, joint]
            if (
                not wp.isfinite(joint_position_value)
                or not wp.isfinite(joint_velocity_value)
                or not wp.isfinite(target_position)
                or not wp.isfinite(target_velocity)
            ):
                valid = 0
            position_delta = joint_position_value - target_position
            velocity_delta = joint_velocity_value - target_velocity
            joint_position_error += position_delta * position_delta
            joint_velocity_error += velocity_delta * velocity_delta
            joint_speed += joint_velocity_value * joint_velocity_value

        for task_body in range(body_count):
            body = body_indices[task_body]
            body_position_value = body_position[lane, body]
            dx = (
                wp.float64(body_position_value[0])
                - reference_body_position[step_index, body, 0]
            )
            dy = (
                wp.float64(body_position_value[1])
                - reference_body_position[step_index, body, 1]
            )
            dz = (
                wp.float64(body_position_value[2])
                - reference_body_position[step_index, body, 2]
            )
            body_position_error += dx * dx + dy * dy + dz * dz

            orientation = body_orientation[lane, body]
            dot = wp.float64(0.0)
            actual_norm_squared = wp.float64(0.0)
            target_norm_squared = wp.float64(0.0)
            for component in range(4):
                actual = wp.float64(orientation[component])
                target = reference_body_orientation[step_index, body, component]
                dot += actual * target
                actual_norm_squared += actual * actual
                target_norm_squared += target * target
            if (
                not wp.isfinite(dx)
                or not wp.isfinite(dy)
                or not wp.isfinite(dz)
                or not wp.isfinite(dot)
                or not wp.isfinite(actual_norm_squared)
                or not wp.isfinite(target_norm_squared)
                or actual_norm_squared <= wp.float64(1.0e-24)
                or target_norm_squared <= wp.float64(1.0e-24)
            ):
                valid = 0
            else:
                dot = dot / wp.sqrt(actual_norm_squared) / wp.sqrt(target_norm_squared)
                angle = wp.float64(2.0) * wp.acos(
                    wp.clamp(wp.abs(dot), wp.float64(0.0), wp.float64(1.0))
                )
                body_orientation_error += angle * angle

        joints = wp.float64(joint_count)
        bodies = wp.float64(body_count)
        running = (
            joint_position_weight * joint_position_error / joints
            + joint_velocity_weight * joint_velocity_error / joints
            + body_position_weight * body_position_error / bodies
            + body_orientation_weight * body_orientation_error / bodies
            + joint_speed_weight * joint_speed / joints
        )
        if step_index == horizon - 1:
            running += (
                terminal_joint_weight * joint_position_error / joints
                + terminal_body_weight * body_position_error / bodies
            )
        updated = total_cost[lane] + running
        if valid == 0 or not wp.isfinite(updated):
            invalid[lane] = 1
        else:
            total_cost[lane] = updated


class MotionTrackingCostReducer:
    """Reusable per-capacity objective state with boundary-only readback."""

    def __init__(
        self,
        objective: MotionTrackingObjective,
        state_view: RobotStateView,
    ) -> None:
        if wp is None:
            raise RuntimeError("device motion-tracking cost requires warp-lang")
        self._objective = objective
        self._state_view = state_view
        self._joint_names = tuple(state_view.joint_names)
        self._body_names = tuple(state_view.body_names)
        self._batch, self._device = self._validate_state_view(state_view)
        self._horizon: int | None = None
        self._expected_step = 0
        self._prepared = False
        self._static_cost = np.empty(self._batch, dtype=np.float64)
        with wp.ScopedDevice(self._device):
            self._body_indices = wp.array(
                objective.body_indices(self._body_names),
                dtype=wp.int32,
                device=self._device,
            )
            self._total = wp.empty(self._batch, dtype=wp.float64, device=self._device)
            self._invalid = wp.empty(self._batch, dtype=wp.int32, device=self._device)

    def prepare(
        self,
        reference_motion: MotionCommand,
        candidate: MotionCommand,
    ) -> None:
        """Upload one wave's targets and reset its compact accumulators."""
        horizon = self._validate_motion_command(reference_motion, candidate)
        if self._horizon is None:
            self._allocate_targets(horizon)
        elif horizon != self._horizon:
            raise ValueError(
                f"device cost horizon changed from {self._horizon} to {horizon}"
            )

        deformation = _deformation_terms(self._objective, reference_motion, candidate)
        self._static_cost = np.sum(
            np.stack(tuple(deformation.values()), axis=0), axis=0
        )
        if not np.isfinite(self._static_cost).all():
            raise ValueError("deformation cost must be finite")

        with wp.ScopedDevice(self._device):
            self._reference_joint_position.assign(
                np.ascontiguousarray(reference_motion.joint_position[0], dtype=np.float64)
            )
            self._reference_joint_velocity.assign(
                np.ascontiguousarray(reference_motion.joint_velocity[0], dtype=np.float64)
            )
            self._reference_body_position.assign(
                np.ascontiguousarray(reference_motion.body_position_world[0], dtype=np.float64)
            )
            self._reference_body_orientation.assign(
                np.ascontiguousarray(
                    reference_motion.body_orientation_world[0], dtype=np.float64
                )
            )
            self._total.zero_()
            self._invalid.zero_()
        self._expected_step = 0
        self._prepared = True

    def record(self, step_index: int, state_view: RobotStateView) -> None:
        """Accumulate one post-step frame; steps must be complete and ordered."""
        if not self._prepared or self._horizon is None:
            raise RuntimeError("prepare must be called before record")
        if state_view is not self._state_view:
            raise ValueError("device cost reducer is bound to one RobotStateView")
        if step_index != self._expected_step or step_index >= self._horizon:
            raise ValueError(
                f"expected objective step {self._expected_step}, got {step_index}"
            )
        o = self._objective
        with wp.ScopedDevice(self._device):
            wp.launch(
                _accumulate_motion_tracking_cost,
                dim=self._batch,
                inputs=[
                    step_index,
                    self._horizon,
                    len(self._joint_names),
                    len(o.body_names),
                    self._body_indices,
                    state_view.joint_position,
                    state_view.joint_velocity,
                    state_view.body_position_world,
                    state_view.body_orientation_world,
                    self._reference_joint_position,
                    self._reference_joint_velocity,
                    self._reference_body_position,
                    self._reference_body_orientation,
                    o.tracking_joint_position_weight,
                    o.tracking_joint_velocity_weight,
                    o.tracking_body_position_weight,
                    o.tracking_body_orientation_weight,
                    o.joint_speed_weight,
                    o.terminal_joint_position_weight,
                    o.terminal_body_position_weight,
                    self._total,
                    self._invalid,
                ],
                device=self._device,
            )
        self._expected_step += 1

    def finalize(self) -> np.ndarray:
        """Read only the total and invalid vectors after a complete horizon."""
        if not self._prepared or self._horizon is None:
            raise RuntimeError("prepare must be called before finalize")
        if self._expected_step != self._horizon:
            raise RuntimeError(
                f"objective recorded {self._expected_step}/{self._horizon} steps"
            )
        wp.synchronize_device(self._device)
        invalid = np.asarray(self._invalid.numpy(), dtype=np.int32)
        dynamic = np.asarray(self._total.numpy(), dtype=np.float64)
        self._prepared = False
        if invalid.shape != (self._batch,) or dynamic.shape != (self._batch,):
            raise RuntimeError("device objective returned invalid batch shapes")
        result = dynamic + self._static_cost
        bad = (invalid != 0) | ~np.isfinite(dynamic) | ~np.isfinite(result)
        result[bad] = np.nan
        return result

    def _allocate_targets(self, horizon: int) -> None:
        with wp.ScopedDevice(self._device):
            self._reference_joint_position = wp.empty(
                (horizon, len(self._joint_names)),
                dtype=wp.float64,
                device=self._device,
            )
            self._reference_joint_velocity = wp.empty(
                (horizon, len(self._joint_names)),
                dtype=wp.float64,
                device=self._device,
            )
            self._reference_body_position = wp.empty(
                (horizon, len(self._body_names), 3),
                dtype=wp.float64,
                device=self._device,
            )
            self._reference_body_orientation = wp.empty(
                (horizon, len(self._body_names), 4),
                dtype=wp.float64,
                device=self._device,
            )
        self._horizon = horizon

    def _validate_motion_command(
        self, reference_motion: MotionCommand, candidate: MotionCommand
    ) -> int:
        horizon = len(reference_motion.query_offsets)
        dense = tuple(range(horizon))
        if horizon < 1 or reference_motion.query_offsets != dense or candidate.query_offsets != dense:
            raise ValueError("reference_motion and candidate motion_command must share one dense horizon")
        if reference_motion.joint_position.shape[0] != 1:
            raise ValueError("reference_motion motion_command must contain one lane")
        if candidate.joint_position.shape[0] != self._batch:
            raise ValueError(
                f"candidate batch must be {self._batch}, got "
                f"{candidate.joint_position.shape[0]}"
            )
        for label, motion_command in (("reference_motion", reference_motion), ("candidate", candidate)):
            if (
                motion_command.joint_names != self._joint_names
                or motion_command.body_names != self._body_names
            ):
                raise ValueError(f"{label} names must match the bound RobotStateView")
            for field in MOTION_COMMAND_ARRAY_FIELDS:
                if not np.isfinite(getattr(motion_command, field)).all():
                    raise ValueError(f"{label} {field} must be finite")
        return horizon

    def _validate_state_view(self, state_view: RobotStateView) -> tuple[int, Any]:
        joint_shape = (int(state_view.joint_position.shape[0]), len(self._joint_names))
        if joint_shape[0] < 1:
            raise ValueError("device state batch must be positive")
        body_shape = (joint_shape[0], len(self._body_names))
        specs = (
            (state_view.joint_position, joint_shape, wp.float32, "joint_position"),
            (state_view.joint_velocity, joint_shape, wp.float32, "joint_velocity"),
            (state_view.body_position_world, body_shape, wp.vec3, "body_position_world"),
            (
                state_view.body_orientation_world,
                body_shape,
                wp.quat,
                "body_orientation_world",
            ),
        )
        device = getattr(state_view.joint_position, "device", None)
        if device is None or not getattr(device, "is_cuda", False):
            raise ValueError("device cost reducer requires CUDA Warp state arrays")
        for value, shape, dtype, field in specs:
            if (
                getattr(value, "shape", None) != shape
                or getattr(value, "dtype", None) != dtype
                or getattr(value, "device", None) != device
                or not getattr(value, "is_contiguous", False)
            ):
                raise ValueError(
                    f"{field} must be contiguous {dtype} with shape {shape} on {device}"
                )
        return joint_shape[0], device
