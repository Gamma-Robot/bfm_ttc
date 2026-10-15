"""Registry-enumerated bfm conformance suite.

BFM state stays opaque here: it flows only through adapter methods and
its internal structure is never asserted on. Advanced states are compared
behaviorally, through the tensors a subsequent Control Step produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pytest

import heft_family
import sonic_family
import toy_family
import twist2_family
from bfm_adapters import (
    MotionCommand,
    NamedTensors,
    RobotState,
    TensorSpec,
    BFMAdapter,
    BFMInstanceConfig,
    BFMStateBatch,
    available,
    get_family,
)

# Registry state captured after conformance-participating imports; throwaway families
# from test_registry are registered later, at test run time.
_FAMILIES_AT_IMPORT = frozenset(available())

# Family modules that contribute a ConformanceCase. Built-in families append
# here when they join the suite; the registry check below still owns coverage.
_CONFORMANCE_MODULES = (
    toy_family,
    heft_family,
    sonic_family,
    twist2_family,
)

BATCH = 3
COMMAND_FIELDS = (
    "position_target",
    "velocity_target",
    "feedforward_torque",
    "proportional_gain",
    "derivative_gain",
)


@dataclass(frozen=True)
class ConformanceCase:
    """One family's config plus family-supplied synthetic-scene builders."""

    config: BFMInstanceConfig
    make_robot_state: Callable[[BFMInstanceConfig, int, int], RobotState]
    make_motion_command: Callable[[BFMInstanceConfig, int, int], MotionCommand]
    fake_policy: Callable[[BFMInstanceConfig, NamedTensors], NamedTensors]
    # Artifact identity -> declared I/O layout that identity claims to expose.
    artifact_io: Mapping[str, Mapping[str, tuple[TensorSpec, ...]]]

    def build_adapter(self) -> BFMAdapter:
        return get_family(self.config.family)(self.config)

    def build_scene(self) -> tuple[RobotState, MotionCommand]:
        return (
            self.make_robot_state(self.config, BATCH, 7),
            self.make_motion_command(self.config, BATCH, 11),
        )


def cases() -> tuple[ConformanceCase, ...]:
    """One contribution per participating family module."""

    return tuple(
        ConformanceCase(**module.conformance_case()) for module in _CONFORMANCE_MODULES
    )


CASES = cases()
CASE_IDS = [case.config.family for case in CASES]


def _step(
    case: ConformanceCase,
    adapter: BFMAdapter,
    state: RobotState,
    motion_command: MotionCommand,
    bfm_state: BFMStateBatch,
) -> tuple[NamedTensors, object, BFMStateBatch]:
    inputs, bfm_state = adapter.build_policy_inputs(state, motion_command, bfm_state)
    outputs = case.fake_policy(case.config, inputs)
    command, bfm_state = adapter.map_policy_outputs(outputs, bfm_state)
    return inputs, command, bfm_state


def _assert_tensors_identical(actual: NamedTensors, expected: NamedTensors) -> None:
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype
        assert actual[name].tobytes() == tensor.tobytes()


def _assert_matches_specs(
    tensors: NamedTensors, specs: tuple[TensorSpec, ...], batch_size: int
) -> None:
    assert set(tensors) == {spec.name for spec in specs}
    for spec in specs:
        tensor = tensors[spec.name]
        assert tensor.dtype == np.dtype(spec.dtype)
        assert tensor.shape == (batch_size, *spec.non_batch_shape)


def _spec_fingerprint(specs: tuple[TensorSpec, ...]) -> tuple[tuple, ...]:
    return tuple((spec.name, spec.dtype, spec.non_batch_shape) for spec in specs)


def _repeat_state(state: RobotState, batch_size: int) -> RobotState:
    repeat = lambda value: np.repeat(value, batch_size, axis=0)
    return RobotState(
        joint_names=state.joint_names,
        body_names=state.body_names,
        joint_position=repeat(state.joint_position),
        joint_velocity=repeat(state.joint_velocity),
        body_position_world=repeat(state.body_position_world),
        body_orientation_world=repeat(state.body_orientation_world),
        body_linear_velocity_world=repeat(state.body_linear_velocity_world),
        body_angular_velocity_world=repeat(state.body_angular_velocity_world),
    )


def _repeat_motion_command(motion_command: MotionCommand, batch_size: int) -> MotionCommand:
    repeat = lambda value: np.repeat(value, batch_size, axis=0)
    return MotionCommand(
        query_offsets=motion_command.query_offsets,
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        joint_position=repeat(motion_command.joint_position),
        joint_velocity=repeat(motion_command.joint_velocity),
        body_position_world=repeat(motion_command.body_position_world),
        body_orientation_world=repeat(motion_command.body_orientation_world),
        body_linear_velocity_world=repeat(motion_command.body_linear_velocity_world),
        body_angular_velocity_world=repeat(motion_command.body_angular_velocity_world),
    )


def test_every_family_has_a_conformance_case() -> None:
    covered = {case.config.family for case in CASES}
    assert covered <= set(available())
    assert _FAMILIES_AT_IMPORT <= covered


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_observation_determinism(case: ConformanceCase) -> None:
    state, motion_command = case.build_scene()
    runs = []
    for _ in range(2):
        adapter = case.build_adapter()
        bfm_state = adapter.initial_state(BATCH)
        first, _, bfm_state = _step(case, adapter, state, motion_command, bfm_state)
        second, _ = adapter.build_policy_inputs(state, motion_command, bfm_state)
        runs.append((first, second))
    for actual, expected in zip(*runs):
        _assert_tensors_identical(actual, expected)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_policy_io_matches_declared_specs(case: ConformanceCase) -> None:
    state, motion_command = case.build_scene()
    adapter = case.build_adapter()
    inputs, _ = adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(BATCH)
    )
    _assert_matches_specs(inputs, case.config.policy_inputs, BATCH)
    outputs = case.fake_policy(case.config, inputs)
    _assert_matches_specs(outputs, case.config.policy_outputs, BATCH)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_named_tensor_specs_match_artifact_claims(case: ConformanceCase) -> None:
    """Config I/O names/shapes must match what each artifact identity claims."""

    claimed_identities = set(case.artifact_io)
    config_identities = set(case.config.artifacts.values())
    assert claimed_identities == config_identities
    assert set(case.config.artifacts)  # at least one artifact key
    # Toy / v1: one policy artifact owns the full declared I/O surface.
    policy_identity = case.config.artifacts["policy"]
    claim = case.artifact_io[policy_identity]
    assert _spec_fingerprint(claim["inputs"]) == _spec_fingerprint(
        case.config.policy_inputs
    )
    assert _spec_fingerprint(claim["outputs"]) == _spec_fingerprint(
        case.config.policy_outputs
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_command_mapping(case: ConformanceCase) -> None:
    state, motion_command = case.build_scene()
    adapter = case.build_adapter()
    _, command, _ = _step(case, adapter, state, motion_command, adapter.initial_state(BATCH))
    assert command.joint_names == case.config.command_joint_names
    expected_shape = (BATCH, len(case.config.command_joint_names))
    for field in COMMAND_FIELDS:
        array = getattr(command, field)
        assert array.shape == expected_shape
        assert np.isfinite(array).all()


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_clone_survives_advancing_the_original(case: ConformanceCase) -> None:
    state, motion_command = case.build_scene()
    adapter = case.build_adapter()
    _, _, bfm_state = _step(
        case, adapter, state, motion_command, adapter.initial_state(BATCH)
    )
    clone = adapter.clone_state(bfm_state)
    expected_inputs, expected_command, _ = _step(
        case, adapter, state, motion_command, adapter.clone_state(clone)
    )
    for _ in range(3):
        _, _, bfm_state = _step(case, adapter, state, motion_command, bfm_state)
    actual_inputs, actual_command, _ = _step(case, adapter, state, motion_command, clone)
    _assert_tensors_identical(actual_inputs, expected_inputs)
    for field in COMMAND_FIELDS:
        assert (
            getattr(actual_command, field).tobytes()
            == getattr(expected_command, field).tobytes()
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expand_state_repeats_one_execution_lane(case: ConformanceCase) -> None:
    adapter = case.build_adapter()
    state = case.make_robot_state(case.config, 1, 17)
    motion_command = case.make_motion_command(case.config, 1, 19)
    _, _, bfm_state = _step(
        case, adapter, state, motion_command, adapter.initial_state(1)
    )
    expected, _ = adapter.build_policy_inputs(
        state, motion_command, adapter.clone_state(bfm_state)
    )

    expanded = adapter.expand_state(bfm_state, BATCH)
    actual, _ = adapter.build_policy_inputs(
        _repeat_state(state, BATCH),
        _repeat_motion_command(motion_command, BATCH),
        expanded,
    )
    for name, one_lane in expected.items():
        np.testing.assert_array_equal(
            actual[name], np.repeat(one_lane, BATCH, axis=0)
        )

    # Expanding and advancing rollout lanes must not mutate the execution lane.
    source_after, _ = adapter.build_policy_inputs(
        state, motion_command, adapter.clone_state(bfm_state)
    )
    _assert_tensors_identical(source_after, expected)
