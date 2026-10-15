from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from bfm_adapters import MotionCommand, RobotState
from command_optimization.config import dataclass_from_dict
from command_optimization.motion_tracking.objective import CostResult, MotionTrackingObjective
from command_optimization.simulation.measurements import (
    MeasurementBuffer,
    MeasurementPrimitive,
    ControlStepMeasurements,
)

JOINTS = ("hip", "knee")
BODIES = ("pelvis", "foot")
ALL_PRIMITIVES = frozenset(MeasurementPrimitive)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
WEIGHTS = tuple(
    field.name for field in fields(MotionTrackingObjective) if field.name != "body_names"
)


def _quats(quat, shape):
    return np.broadcast_to(np.asarray(quat, dtype=float), (*shape, 4)).copy()


def _state(
    batch=2,
    jp=0.0,
    jv=0.0,
    bp=0.0,
    quat=IDENTITY,
    joints=JOINTS,
    bodies=BODIES,
):
    nj, nb = len(joints), len(bodies)
    return RobotState(
        joint_names=joints,
        body_names=bodies,
        joint_position=np.full((batch, nj), float(jp)),
        joint_velocity=np.full((batch, nj), float(jv)),
        body_position_world=np.broadcast_to(
            np.asarray(bp, dtype=float), (batch, nb, 3)
        ).copy(),
        body_orientation_world=_quats(quat, (batch, nb)),
        body_linear_velocity_world=np.zeros((batch, nb, 3)),
        body_angular_velocity_world=np.zeros((batch, nb, 3)),
    )


def _arrays(batch=2, num_steps=3, jp=0.0, jv=0.0, bp=0.0, quat=IDENTITY, bodies=BODIES):
    nj, nb = len(JOINTS), len(bodies)
    return {
        MeasurementPrimitive.JOINT_POSITION: np.full((batch, num_steps, nj), float(jp)),
        MeasurementPrimitive.JOINT_VELOCITY: np.full((batch, num_steps, nj), float(jv)),
        MeasurementPrimitive.BODY_POSITION_WORLD: np.broadcast_to(
            np.asarray(bp, dtype=float), (batch, num_steps, nb, 3)
        ).copy(),
        MeasurementPrimitive.BODY_ORIENTATION_WORLD: _quats(
            quat, (batch, num_steps, nb)
        ),
    }


def _measurements(*, bodies=BODIES, **kwargs):
    return ControlStepMeasurements(JOINTS, bodies, _arrays(bodies=bodies, **kwargs))


def _motion_command(
    batch=1,
    num_steps=3,
    jp=0.0,
    jv=0.0,
    bp=0.0,
    quat=IDENTITY,
    offsets=None,
    joints=JOINTS,
    bodies=BODIES,
):
    offsets = tuple(range(num_steps)) if offsets is None else offsets
    num_steps = len(offsets)
    nj, nb = len(joints), len(bodies)
    return MotionCommand(
        query_offsets=offsets,
        joint_names=joints,
        body_names=bodies,
        joint_position=np.full((batch, num_steps, nj), float(jp)),
        joint_velocity=np.full((batch, num_steps, nj), float(jv)),
        body_position_world=np.broadcast_to(
            np.asarray(bp, dtype=float), (batch, num_steps, nb, 3)
        ).copy(),
        body_orientation_world=_quats(quat, (batch, num_steps, nb)),
        body_linear_velocity_world=np.zeros((batch, num_steps, nb, 3)),
        body_angular_velocity_world=np.zeros((batch, num_steps, nb, 3)),
    )


def _objective(*, body_names=BODIES, **weights):
    values = dict.fromkeys(WEIGHTS, 0.0)
    values.update(weights)
    return MotionTrackingObjective(body_names=body_names, **values)


def _evaluate(objective, measurements=None, reference_motion=None, candidate=None):
    measurements = measurements or _measurements()
    reference_motion = reference_motion or _motion_command()
    candidate = candidate or _motion_command(batch=measurements.batch)
    return objective.evaluate(measurements, reference_motion, candidate)


def test_buffer_allocates_only_required_primitives() -> None:
    requires = frozenset(
        {
            MeasurementPrimitive.JOINT_POSITION,
            MeasurementPrimitive.BODY_ORIENTATION_WORLD,
        }
    )
    buffer = MeasurementBuffer(requires, JOINTS, BODIES, batch=2, num_steps=2)
    for step in range(2):
        buffer.record(step, _state())
    measurements = buffer.finalize()
    assert set(measurements.arrays) == requires
    assert measurements.arrays[MeasurementPrimitive.JOINT_POSITION].shape == (
        2,
        2,
        2,
    )
    assert measurements.arrays[
        MeasurementPrimitive.BODY_ORIENTATION_WORLD
    ].shape == (2, 2, 2, 4)


def test_buffer_rejects_incomplete_or_duplicate_steps() -> None:
    buffer = MeasurementBuffer(ALL_PRIMITIVES, JOINTS, BODIES, batch=1, num_steps=2)
    state = _state(batch=1)
    buffer.record(0, state)
    with pytest.raises(ValueError, match="already recorded"):
        buffer.record(0, state)
    with pytest.raises(ValueError, match="unrecorded"):
        buffer.finalize()


def test_buffer_record_rejects_name_and_batch_mismatch() -> None:
    buffer = MeasurementBuffer(ALL_PRIMITIVES, JOINTS, BODIES, batch=2, num_steps=1)
    with pytest.raises(ValueError, match="names"):
        buffer.record(0, _state(joints=("a", "b")))
    with pytest.raises(ValueError, match="batch"):
        buffer.record(0, _state(batch=3))


def test_buffer_defensively_copies_recorded_state() -> None:
    buffer = MeasurementBuffer(ALL_PRIMITIVES, JOINTS, BODIES, batch=1, num_steps=1)
    state = _state(batch=1, jp=0.25, bp=0.5)
    buffer.record(0, state)
    state.joint_position[:] = 99.0
    state.body_position_world[:] = 99.0
    measurements = buffer.finalize()
    np.testing.assert_array_equal(
        measurements.arrays[MeasurementPrimitive.JOINT_POSITION], 0.25
    )
    np.testing.assert_array_equal(
        measurements.arrays[MeasurementPrimitive.BODY_POSITION_WORLD], 0.5
    )


def test_measurements_validate_shape_and_finiteness() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ControlStepMeasurements(JOINTS, BODIES, {})
    bad_shape = _arrays()
    bad_shape[MeasurementPrimitive.JOINT_POSITION] = np.zeros((2, 3, 5))
    with pytest.raises(ValueError, match="shape"):
        ControlStepMeasurements(JOINTS, BODIES, bad_shape)
    non_finite = _arrays()
    non_finite[MeasurementPrimitive.JOINT_VELOCITY][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ControlStepMeasurements(JOINTS, BODIES, non_finite)


def test_perfect_reference_and_candidate_have_zero_cost() -> None:
    objective = _objective(
        tracking_joint_position_weight=1.0,
        tracking_joint_velocity_weight=1.0,
        tracking_body_position_weight=1.0,
        tracking_body_orientation_weight=1.0,
        deformation_joint_position_weight=1.0,
        deformation_body_position_weight=1.0,
        deformation_body_orientation_weight=1.0,
        joint_speed_weight=1.0,
        terminal_joint_position_weight=1.0,
        terminal_body_position_weight=1.0,
    )
    result = _evaluate(
        objective,
        measurements=_measurements(jp=0.3, bp=0.7),
        reference_motion=_motion_command(jp=0.3, bp=0.7),
        candidate=_motion_command(batch=2, jp=0.3, bp=0.7),
    )
    np.testing.assert_array_equal(result.total, np.zeros(2))


def test_task_fidelity_uses_reference_not_candidate() -> None:
    objective = _objective(
        tracking_joint_position_weight=2.0,
        deformation_joint_position_weight=3.0,
    )
    result = _evaluate(
        objective,
        measurements=_measurements(jp=0.0),
        reference_motion=_motion_command(jp=0.0),
        candidate=_motion_command(batch=2, jp=0.5),
    )
    np.testing.assert_array_equal(result.terms["tracking/joint_position"], 0.0)
    np.testing.assert_allclose(
        result.terms["deformation/joint_position"], np.full(2, 3 * 3 * 0.25)
    )


def test_tracking_joint_terms_sum_over_horizon() -> None:
    objective = _objective(
        tracking_joint_position_weight=2.0,
        tracking_joint_velocity_weight=4.0,
    )
    result = _evaluate(
        objective,
        measurements=_measurements(jp=0.5, jv=0.4),
        reference_motion=_motion_command(jv=0.1),
    )
    np.testing.assert_allclose(
        result.terms["tracking/joint_position"], np.full(2, 2 * 3 * 0.25)
    )
    np.testing.assert_allclose(
        result.terms["tracking/joint_velocity"], np.full(2, 4 * 3 * 0.09)
    )


def test_tracking_body_position_sums_xyz_and_time() -> None:
    objective = _objective(tracking_body_position_weight=10.0)
    result = _evaluate(
        objective, measurements=_measurements(bp=(0.1, 0.2, 0.2))
    )
    np.testing.assert_allclose(
        result.terms["tracking/body_position"], np.full(2, 10 * 3 * 0.09)
    )


@pytest.mark.parametrize("scale", (1.0, -1.0, -7.0))
def test_orientation_uses_sign_invariant_geodesic_error(scale: float) -> None:
    angle = np.pi / 3
    quat = (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))
    objective = _objective(tracking_body_orientation_weight=3.0)
    expected = np.full(2, 3 * 3 * angle**2)
    result = _evaluate(
        objective, measurements=_measurements(quat=np.asarray(quat) * scale)
    )
    np.testing.assert_allclose(
        result.terms["tracking/body_orientation"], expected
    )


def test_same_pose_has_zero_orientation_cost_with_nonunit_quaternions() -> None:
    objective = _objective(
        tracking_body_orientation_weight=1.0,
        deformation_body_orientation_weight=1.0,
    )
    result = _evaluate(
        objective,
        measurements=_measurements(quat=(0.9999999, 0.0, 0.0, 0.0)),
        reference_motion=_motion_command(quat=(0.9999999, 0.0, 0.0, 0.0)),
        candidate=_motion_command(batch=2, quat=(-3.0, 0.0, 0.0, 0.0)),
    )
    np.testing.assert_array_equal(result.total, 0.0)


@pytest.mark.parametrize("component", (0.0, np.nan, np.inf))
def test_orientation_cost_rejects_invalid_quaternions(component: float) -> None:
    measurements = _measurements()
    measurements.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD][0, 0, 0] = (
        component, 0.0, 0.0, 0.0
    )
    with pytest.raises(ValueError, match="quaternions"):
        _evaluate(_objective(tracking_body_orientation_weight=1.0), measurements)


@pytest.mark.parametrize("changed_body", ("foot", "policy_only"))
def test_body_terms_use_task_subset_in_declared_order(changed_body: str) -> None:
    bodies = ("pelvis", "policy_only", "foot")
    objective = _objective(
        body_names=("foot", "pelvis"),
        tracking_body_position_weight=1.0,
        tracking_body_orientation_weight=1.0,
        deformation_body_position_weight=1.0,
        deformation_body_orientation_weight=1.0,
        terminal_body_position_weight=1.0,
    )
    assert objective.body_indices(bodies) == [2, 0]
    measurements = _measurements(bodies=bodies)
    reference_motion = _motion_command(bodies=bodies)
    candidate = _motion_command(batch=2, bodies=bodies)
    index = bodies.index(changed_body)
    angle = np.pi / 3
    quat = (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))
    measurements.arrays[MeasurementPrimitive.BODY_POSITION_WORLD][:, :, index, 0] = 1.0
    measurements.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD][:, :, index] = quat
    candidate.body_position_world[:, :, index, 0] = 1.0
    candidate.body_orientation_world[:, :, index] = quat

    result = _evaluate(objective, measurements, reference_motion, candidate)

    expected = 0.5 if changed_body == "foot" else 0.0
    for name in ("tracking", "deformation"):
        np.testing.assert_allclose(result.terms[f"{name}/body_position"], 3 * expected)
        np.testing.assert_allclose(
            result.terms[f"{name}/body_orientation"], 3 * expected * angle**2
        )
    np.testing.assert_allclose(result.terms["terminal/body_position"], expected)


def test_joint_speed_is_a_stability_term_not_tracking_error() -> None:
    objective = _objective(joint_speed_weight=4.0)
    result = _evaluate(
        objective,
        measurements=_measurements(jv=0.3),
        reference_motion=_motion_command(jv=0.3),
    )
    np.testing.assert_allclose(
        result.terms["stability/joint_speed"], np.full(2, 4 * 3 * 0.09)
    )


def test_terminal_terms_use_last_step_against_reference_motion() -> None:
    objective = _objective(
        terminal_joint_position_weight=2.0,
        terminal_body_position_weight=4.0,
    )
    arrays = _arrays()
    arrays[MeasurementPrimitive.JOINT_POSITION][:, 0] = 7.0
    arrays[MeasurementPrimitive.JOINT_POSITION][:, -1] = 0.5
    arrays[MeasurementPrimitive.BODY_POSITION_WORLD][:, 0] = 9.0
    arrays[MeasurementPrimitive.BODY_POSITION_WORLD][:, -1] = (0.1, 0.2, 0.2)
    result = _evaluate(
        objective, measurements=ControlStepMeasurements(JOINTS, BODIES, arrays)
    )
    np.testing.assert_allclose(
        result.terms["terminal/joint_position"], np.full(2, 2 * 0.25)
    )
    np.testing.assert_allclose(
        result.terms["terminal/body_position"], np.full(2, 4 * 0.09)
    )


def test_cost_result_is_float64_immutable_and_does_not_alias() -> None:
    total = np.array([1.0, 2.0], dtype=np.float32)
    term = np.array([1.0, 2.0], dtype=np.float32)
    result = CostResult(total, {"x": term})
    total[:] = 9.0
    term[:] = 9.0
    assert result.total.dtype == np.float64
    np.testing.assert_array_equal(result.total, (1.0, 2.0))
    np.testing.assert_array_equal(result.terms["x"], (1.0, 2.0))
    with pytest.raises(ValueError):
        result.total[0] = 0.0
    with pytest.raises(TypeError):
        result.terms["y"] = np.zeros(2)
    with pytest.raises(ValueError, match="sum"):
        CostResult(np.array([3.0]), {"x": np.array([2.0])})


def test_rejects_reference_candidate_and_metadata_mismatches() -> None:
    objective = _objective()
    measurements = _measurements()
    incomplete = ControlStepMeasurements(
        JOINTS,
        BODIES,
        {
            MeasurementPrimitive.JOINT_POSITION: np.zeros(
                (measurements.batch, measurements.num_steps, len(JOINTS))
            )
        },
    )
    with pytest.raises(ValueError, match="missing required"):
        objective.evaluate(incomplete, _motion_command(), _motion_command(batch=2))
    with pytest.raises(ValueError, match="reference_motion batch"):
        objective.evaluate(
            measurements, _motion_command(batch=2), _motion_command(batch=2)
        )
    with pytest.raises(ValueError, match="candidate batch"):
        objective.evaluate(
            measurements, _motion_command(), _motion_command(batch=3)
        )
    with pytest.raises(ValueError, match="names"):
        objective.evaluate(
            measurements,
            _motion_command(joints=("a", "b")),
            _motion_command(batch=2),
        )
    with pytest.raises(ValueError, match="query_offsets"):
        objective.evaluate(
            measurements,
            _motion_command(offsets=(0, 2, 4)),
            _motion_command(batch=2),
        )


def test_rejects_nonfinite_motion_command_and_bad_weights() -> None:
    objective = _objective()
    candidate = _motion_command(batch=2)
    candidate.joint_position[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="candidate joint_position"):
        objective.evaluate(_measurements(), _motion_command(), candidate)
    with pytest.raises(ValueError, match="joint_speed_weight"):
        _objective(joint_speed_weight=-0.1)
    with pytest.raises(ValueError, match="tracking_body_position_weight"):
        _objective(tracking_body_position_weight=np.nan)


def test_default_yaml_binds_directly_without_registry() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "motion_tracking"
        / "default.yaml"
    )
    objective = dataclass_from_dict(
        MotionTrackingObjective, yaml.safe_load(path.read_text())
    )
    assert objective.tracking_joint_position_weight == 1.0
    assert objective.deformation_joint_position_weight == 0.1
    assert objective.joint_speed_weight == 0.01
    assert objective.terminal_joint_position_weight == 5.0


def test_requires_exact_simulation_primitives() -> None:
    assert MotionTrackingObjective.requires == ALL_PRIMITIVES
