from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

wp = pytest.importorskip("warp")

from bfm_adapters import MotionCommand
from command_optimization.motion_tracking.device_cost import MotionTrackingCostReducer
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.simulation.measurements import MeasurementPrimitive, ControlStepMeasurements


JOINTS = ("joint_a", "joint_b")
BODIES = ("pelvis", "hand")
WEIGHTS = (
    "tracking_joint_position_weight",
    "tracking_joint_velocity_weight",
    "tracking_body_position_weight",
    "tracking_body_orientation_weight",
    "deformation_joint_position_weight",
    "deformation_body_position_weight",
    "deformation_body_orientation_weight",
    "joint_speed_weight",
    "terminal_joint_position_weight",
    "terminal_body_position_weight",
)


def _quaternion(angle: np.ndarray) -> np.ndarray:
    result = np.zeros((*angle.shape, 4), dtype=np.float64)
    result[..., 0] = np.cos(angle / 2.0)
    result[..., 3] = np.sin(angle / 2.0)
    return result


def _motion_command(batch: int, num_steps: int, *, shift: float) -> MotionCommand:
    lane = np.arange(batch, dtype=np.float64)[:, None, None]
    time = np.arange(num_steps, dtype=np.float64)[None, :, None]
    joint = shift + 0.07 * lane + 0.03 * time + np.asarray([0.1, -0.2])
    body_position = np.zeros((batch, num_steps, len(BODIES), 3), dtype=np.float64)
    body_position[..., 0] = shift + 0.02 * lane + 0.04 * time
    body_position[..., 1] = np.asarray([0.0, 0.2])[None, None]
    angles = shift + 0.01 * lane[..., 0] + 0.03 * time[..., 0]
    orientation = np.repeat(_quaternion(angles)[:, :, None], len(BODIES), axis=2)
    return MotionCommand(
        query_offsets=tuple(range(num_steps)),
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=joint,
        joint_velocity=0.5 * joint,
        body_position_world=body_position,
        body_orientation_world=orientation,
        body_linear_velocity_world=np.zeros_like(body_position),
        body_angular_velocity_world=np.zeros_like(body_position),
    )


def _executed_trajectory(batch: int, num_steps: int) -> dict[MeasurementPrimitive, np.ndarray]:
    target = _motion_command(batch, num_steps, shift=0.11)
    return {
        MeasurementPrimitive.JOINT_POSITION: np.asarray(
            target.joint_position, dtype=np.float32
        ),
        MeasurementPrimitive.JOINT_VELOCITY: np.asarray(
            target.joint_velocity, dtype=np.float32
        ),
        MeasurementPrimitive.BODY_POSITION_WORLD: np.asarray(
            target.body_position_world, dtype=np.float32
        ),
        MeasurementPrimitive.BODY_ORIENTATION_WORLD: np.asarray(
            target.body_orientation_world, dtype=np.float32
        ),
    }


def _device_view(arrays: dict[MeasurementPrimitive, np.ndarray], device):
    batch = next(iter(arrays.values())).shape[0]
    return SimpleNamespace(
        joint_names=JOINTS,
        body_names=BODIES,
        joint_position=wp.empty(
            (batch, len(JOINTS)), dtype=wp.float32, device=device
        ),
        joint_velocity=wp.empty(
            (batch, len(JOINTS)), dtype=wp.float32, device=device
        ),
        body_position_world=wp.empty(
            (batch, len(BODIES)), dtype=wp.vec3, device=device
        ),
        body_orientation_world=wp.empty(
            (batch, len(BODIES)), dtype=wp.quat, device=device
        ),
    )


def _assign_step(view, arrays, step: int) -> None:
    for primitive in MeasurementPrimitive:
        getattr(view, primitive.value).assign(
            np.ascontiguousarray(arrays[primitive][:, step])
        )


def _objective(active: str, *, body_names=BODIES) -> MotionTrackingObjective:
    return MotionTrackingObjective(
        body_names=body_names,
        **{name: float(name == active) for name in WEIGHTS}
    )


@pytest.mark.parametrize("active", WEIGHTS)
def test_device_cost_matches_each_host_objective_term(active: str) -> None:
    assert wp.is_cuda_available()
    batch, num_steps = 3, 4
    reference_motion = _motion_command(1, num_steps, shift=0.02)
    candidate = _motion_command(batch, num_steps, shift=0.06)
    executed_trajectory = _executed_trajectory(batch, num_steps)
    reference_motion.body_orientation_world[...] *= 0.9999999
    candidate.body_orientation_world[...] *= -3.0
    executed_trajectory[MeasurementPrimitive.BODY_ORIENTATION_WORLD] *= 0.9999999
    candidate.body_position_world[:, :, 0] += 3.0
    candidate.body_orientation_world[:, :, 0] = (0.0, 1.0, 0.0, 0.0)
    executed_trajectory[MeasurementPrimitive.BODY_POSITION_WORLD][:, :, 0] += 5.0
    executed_trajectory[MeasurementPrimitive.BODY_ORIENTATION_WORLD][:, :, 0] = (0.0, 1.0, 0.0, 0.0)
    objective = _objective(active, body_names=("hand",))
    expected = objective.evaluate(
        ControlStepMeasurements(JOINTS, BODIES, executed_trajectory), reference_motion, candidate
    ).total
    view = _device_view(executed_trajectory, wp.get_device("cuda:0"))
    reducer = MotionTrackingCostReducer(objective, view)
    reducer.prepare(reference_motion, candidate)
    for step in range(num_steps):
        _assign_step(view, executed_trajectory, step)
        reducer.record(step, view)

    actual = reducer.finalize()

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-10)


@pytest.mark.parametrize("bad_quaternion", (None, 0.0, np.nan, np.inf))
def test_device_cost_rejects_incomplete_and_isolates_nonfinite_lane(
    bad_quaternion: float | None,
) -> None:
    assert wp.is_cuda_available()
    batch, num_steps = 2, 3
    reference_motion = _motion_command(1, num_steps, shift=0.0)
    candidate = _motion_command(batch, num_steps, shift=0.1)
    executed_trajectory = _executed_trajectory(batch, num_steps)
    view = _device_view(executed_trajectory, wp.get_device("cuda:0"))
    reducer = MotionTrackingCostReducer(_objective(WEIGHTS[0]), view)
    reducer.prepare(reference_motion, candidate)
    _assign_step(view, executed_trajectory, 0)
    reducer.record(0, view)
    with pytest.raises(RuntimeError, match="1/3 steps"):
        reducer.finalize()
    with pytest.raises(ValueError, match="expected objective step 1"):
        reducer.record(0, view)

    reducer.prepare(reference_motion, candidate)
    invalid = {key: value.copy() for key, value in executed_trajectory.items()}
    if bad_quaternion is None:
        invalid[MeasurementPrimitive.JOINT_POSITION][0, 1, 0] = np.nan
    else:
        invalid[MeasurementPrimitive.BODY_ORIENTATION_WORLD][0, 1, 0] = (
            bad_quaternion, 0.0, 0.0, 0.0
        )
    for step in range(num_steps):
        _assign_step(view, invalid, step)
        reducer.record(step, view)
    actual = reducer.finalize()

    assert np.isnan(actual[0])
    assert np.isfinite(actual[1])
