from __future__ import annotations

import numpy as np
import pytest

from bfm_adapters import MotionCommand, PDCommand
from command_optimization.simulation.execution_state import ExecutionState, physical_state_digest


class _CountingAdapter:
    """Counts clone_state calls and returns distinguishable opaque clones."""

    def __init__(self) -> None:
        self.clone_calls = 0

    def clone_state(self, state: object) -> object:
        self.clone_calls += 1
        return ("clone", self.clone_calls, state)


def _history(values: np.ndarray) -> MotionCommand:
    joint = np.asarray(values, dtype=np.float64)[..., None]
    batch, num_steps, _ = joint.shape
    body_position = np.zeros((batch, num_steps, 1, 3))
    body_orientation = np.zeros((batch, num_steps, 1, 4))
    body_orientation[..., 0] = 1.0
    body_velocity = np.zeros_like(body_position)
    return MotionCommand(
        query_offsets=tuple(range(-num_steps, 0)),
        joint_names=("joint_0",),
        body_names=("pelvis",),
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body_position,
        body_orientation_world=body_orientation,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=body_velocity.copy(),
    )


def _capture(adapter=None, **overrides):
    kwargs = dict(
        qpos=np.arange(6, dtype=np.float64).reshape(2, 3),
        qvel=np.arange(4, dtype=np.float64).reshape(2, 2),
        control_step=7,
        time=0.14,
        motion_command_index=3,
        command=None,
        bfm_state="opaque-token",
        adapter=adapter or _CountingAdapter(),
    )
    kwargs.update(overrides)
    return ExecutionState.capture(**kwargs)


def test_capture_round_trips_physical_arrays_bit_exact() -> None:
    qpos = np.array([[0.1, -2.5, 3e-8], [np.pi, 1e12, -0.0]])
    qvel = np.array([[7.25, -1e-300], [0.5, 2.0]])

    execution_state = _capture(qpos=qpos, qvel=qvel)

    np.testing.assert_array_equal(execution_state.qpos, qpos, strict=True)
    np.testing.assert_array_equal(execution_state.qvel, qvel, strict=True)
    assert execution_state.qpos.flags.c_contiguous and execution_state.qvel.flags.c_contiguous
    assert execution_state.control_step == 7
    assert execution_state.time == 0.14
    assert execution_state.motion_command_index == 3


def test_physical_state_digest_is_stable_across_dtype_copy_and_layout() -> None:
    qpos = np.array([[0.0, 0.5, -2.0], [4.0, 8.0, 16.0]], dtype=np.float64)
    qvel = np.array([[1.0, -0.25], [2.0, 32.0]], dtype=np.float64)

    expected = physical_state_digest(qpos, qvel)
    actual = physical_state_digest(
        np.asfortranarray(qpos.astype(np.float32)),
        np.asfortranarray(qvel.astype(np.float32)),
    )

    assert actual == expected
    assert len(actual) == 64


@pytest.mark.parametrize("field", ("qpos", "qvel"))
def test_physical_state_digest_changes_with_physical_content(field: str) -> None:
    qpos = np.zeros((1, 3), dtype=np.float64)
    qvel = np.zeros((1, 2), dtype=np.float64)
    expected = physical_state_digest(qpos, qvel)
    changed = {"qpos": qpos.copy(), "qvel": qvel.copy()}
    changed[field][0, 0] = 0.25

    assert physical_state_digest(changed["qpos"], changed["qvel"]) != expected


def test_physical_state_digest_includes_array_shapes() -> None:
    qvel = np.zeros((2, 1), dtype=np.float64)

    flat = physical_state_digest(np.zeros((2, 2)), qvel)
    reshaped = physical_state_digest(np.zeros((1, 4)), np.zeros((1, 2)))

    assert flat != reshaped


def test_mutating_source_after_capture_does_not_affect_execution_state() -> None:
    qpos = np.ones((1, 3))
    qvel = np.ones((1, 2))
    warmstart = np.full_like(qvel, 3.0)
    execution_state = _capture(qpos=qpos, qvel=qvel, qacc_warmstart=warmstart)

    qpos[:] = -9.0
    qvel[:] = -9.0
    warmstart[:] = -9.0

    np.testing.assert_array_equal(execution_state.qpos, np.ones((1, 3)))
    np.testing.assert_array_equal(execution_state.qvel, np.ones((1, 2)))
    np.testing.assert_array_equal(execution_state.qacc_warmstart, np.full((1, 2), 3.0))
    assert not execution_state.qacc_warmstart.flags.writeable


@pytest.mark.parametrize("warmstart", (np.zeros((1, 2)), np.full((2, 2), np.nan)))
def test_warmstart_must_match_finite_generalized_velocities(warmstart) -> None:
    with pytest.raises(ValueError, match="qacc_warmstart"):
        _capture(qacc_warmstart=warmstart)


def test_capture_owns_immutable_applied_command_history() -> None:
    source = _history([[1.0, 2.0], [3.0, 4.0]])

    execution_state = _capture(motion_command_history=source)
    source.joint_position[:] = -9.0

    assert execution_state.motion_command_history is not None
    np.testing.assert_array_equal(
        execution_state.motion_command_history.joint_position[:, :, 0], [[1.0, 2.0], [3.0, 4.0]]
    )
    assert not execution_state.motion_command_history.joint_position.flags.writeable
    with pytest.raises(ValueError):
        execution_state.motion_command_history.joint_position[0, 0, 0] = 8.0


def test_execution_state_arrays_are_non_writeable() -> None:
    execution_state = _capture()

    assert not execution_state.qpos.flags.writeable
    assert not execution_state.qvel.flags.writeable
    with pytest.raises(ValueError):
        execution_state.qpos[0, 0] = 1.0


def test_capture_stores_command_identity() -> None:
    joints = ("joint_0",)
    ones = np.ones((2, 1))
    command = PDCommand(joints, ones, ones, ones, ones, ones)

    execution_state = _capture(command=command)

    assert execution_state.command is command


def test_capture_clones_bfm_state_exactly_once() -> None:
    adapter = _CountingAdapter()

    execution_state = _capture(adapter=adapter)

    assert adapter.clone_calls == 1
    assert execution_state.bfm_state == ("clone", 1, "opaque-token")


def test_restore_clones_once_per_invocation_and_is_independent() -> None:
    adapter = _CountingAdapter()
    execution_state = _capture(adapter=adapter)

    first = execution_state.restore_bfm_state(adapter)
    second = execution_state.restore_bfm_state(adapter)

    assert adapter.clone_calls == 3
    assert first == ("clone", 2, ("clone", 1, "opaque-token"))
    assert second == ("clone", 3, ("clone", 1, "opaque-token"))
    assert first != second
    assert execution_state.bfm_state == ("clone", 1, "opaque-token")


def test_rejects_mismatched_batch_sizes() -> None:
    with pytest.raises(ValueError, match="batch"):
        _capture(qpos=np.zeros((2, 3)), qvel=np.zeros((3, 2)))


def test_rejects_non_2d_arrays() -> None:
    with pytest.raises(ValueError, match="qpos"):
        _capture(qpos=np.zeros(3))
    with pytest.raises(ValueError, match="qvel"):
        _capture(qvel=np.zeros((1, 2, 2)))


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _capture(qpos=np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        _capture(qvel=np.array([[np.inf, 0.0], [0.0, 0.0]]))


def test_rejects_negative_counters() -> None:
    with pytest.raises(ValueError, match="control_step"):
        _capture(control_step=-1)
    with pytest.raises(ValueError, match="time"):
        _capture(time=-0.01)
    with pytest.raises(ValueError, match="time"):
        _capture(time=float("nan"))
    with pytest.raises(ValueError, match="motion_command_index"):
        _capture(motion_command_index=-1)


def test_rejects_noncanonical_or_mismatched_motion_command_history() -> None:
    history = _history([[1.0], [2.0]])
    noncanonical = MotionCommand(
        query_offsets=(0,),
        joint_names=history.joint_names,
        body_names=history.body_names,
        joint_position=history.joint_position,
        joint_velocity=history.joint_velocity,
        body_position_world=history.body_position_world,
        body_orientation_world=history.body_orientation_world,
        body_linear_velocity_world=history.body_linear_velocity_world,
        body_angular_velocity_world=history.body_angular_velocity_world,
    )
    with pytest.raises(ValueError, match="dense negative"):
        _capture(motion_command_history=noncanonical)
    with pytest.raises(ValueError, match="batch size"):
        _capture(motion_command_history=_history([[1.0]]))
