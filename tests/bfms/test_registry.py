from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from bfm_adapters import (
    MotionCommand,
    PDCommand,
    RobotState,
    TensorSpec,
    BFMAdapter,
    BFMInstanceConfig,
    available,
    get_family,
    instance_config_from_mapping,
    load_instance_config,
    register,
)

JOINTS = tuple(f"joint_{index}" for index in range(29))
TASK_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

SCHEMAS = (
    (
        "heft",
        (
            ("context", (1,)),
            ("motion_command", (114,)),
            ("target_motion", (806,)),
            ("proprioception", (808,)),
        ),
        (("action", (29,)),),
        (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16),
        TASK_BODIES,
    ),
    (
        "sonic",
        (("g1_input", (640,)), ("proprioception", (930,))),
        (("action", (29,)),),
        (0, 5, 10, 15, 20, 25, 30, 35, 40, 45),
        TASK_BODIES,
    ),
    (
        "twist2",
        (("input", (1432,)),),
        (("output", (29,)),),
        (0, 1),
        ("pelvis",),
    ),
)


def _specs(values: tuple[tuple[str, tuple[int, ...]], ...]) -> tuple[TensorSpec, ...]:
    return tuple(TensorSpec(name, "float32", shape) for name, shape in values)


def _config(
    family: str = "fake",
    inputs: tuple[tuple[str, tuple[int, ...]], ...] = (("observation", (10,)),),
    outputs: tuple[tuple[str, tuple[int, ...]], ...] = (("action", (29,)),),
    bodies: tuple[str, ...] = ("pelvis",),
    query_offsets: tuple[int, ...] = (0,),
) -> BFMInstanceConfig:
    return BFMInstanceConfig(
        family=family,
        instance=f"{family}_instance",
        artifacts={"policy": "sha256:abc"},
        control_period=0.02,
        motion_command_query_offsets=query_offsets,
        policy_inputs=_specs(inputs),
        policy_outputs=_specs(outputs),
        policy_joint_names=JOINTS,
        command_joint_names=JOINTS,
        motion_command_body_names=bodies,
        family_config={},
    )


@pytest.mark.parametrize(
    ("family", "inputs", "outputs", "query_offsets", "bodies"), SCHEMAS
)
def test_representative_bfm_schemas(
    family: str,
    inputs: tuple[tuple[str, tuple[int, ...]], ...],
    outputs: tuple[tuple[str, tuple[int, ...]], ...],
    query_offsets: tuple[int, ...],
    bodies: tuple[str, ...],
) -> None:
    config = _config(family, inputs, outputs, bodies, query_offsets)
    batch, queries, joints, body_count = 2, len(query_offsets), len(JOINTS), len(bodies)
    motion_command = MotionCommand(
        query_offsets=query_offsets,
        joint_names=JOINTS,
        body_names=bodies,
        joint_position=np.zeros((batch, queries, joints), dtype=np.float32),
        joint_velocity=np.zeros((batch, queries, joints), dtype=np.float32),
        body_position_world=np.zeros(
            (batch, queries, body_count, 3), dtype=np.float32
        ),
        body_orientation_world=np.zeros(
            (batch, queries, body_count, 4), dtype=np.float32
        ),
        body_linear_velocity_world=np.zeros(
            (batch, queries, body_count, 3), dtype=np.float32
        ),
        body_angular_velocity_world=np.zeros(
            (batch, queries, body_count, 3), dtype=np.float32
        ),
    )

    assert tuple(spec.name for spec in config.policy_inputs) == tuple(
        name for name, _ in inputs
    )
    assert tuple(spec.name for spec in config.policy_outputs) == tuple(
        name for name, _ in outputs
    )
    assert config.motion_command_body_names == bodies
    assert config.motion_command_query_offsets == query_offsets
    assert motion_command.query_offsets == query_offsets


def test_load_instance_config(tmp_path: Path) -> None:
    expected = _config()
    data = {
        "family": expected.family,
        "instance": expected.instance,
        "artifacts": dict(expected.artifacts),
        "control_period": expected.control_period,
        "motion_command_query_offsets": list(expected.motion_command_query_offsets),
        "policy_inputs": [
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "non_batch_shape": list(spec.non_batch_shape),
            }
            for spec in expected.policy_inputs
        ],
        "policy_outputs": [
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "non_batch_shape": list(spec.non_batch_shape),
            }
            for spec in expected.policy_outputs
        ],
        "policy_joint_names": list(expected.policy_joint_names),
        "command_joint_names": list(expected.command_joint_names),
        "motion_command_body_names": list(expected.motion_command_body_names),
        "family_config": {},
    }
    path = tmp_path / "bfm.yaml"
    path.write_text(yaml.safe_dump(data))

    actual = load_instance_config(path)

    assert actual == expected
    assert instance_config_from_mapping(data) == expected


def test_config_rejects_incomplete_joint_mapping() -> None:
    with pytest.raises(ValueError, match="same joints"):
        replace(_config(), command_joint_names=JOINTS[:-1])


def test_config_rejects_invalid_motion_command_offsets_and_freezes_family_config() -> None:
    with pytest.raises(ValueError, match="motion_command_query_offsets"):
        replace(_config(), motion_command_query_offsets=())
    with pytest.raises(ValueError, match="motion_command_query_offsets"):
        replace(_config(), motion_command_query_offsets=(0, 0))

    config = replace(_config(), family_config={"history": [0, 1]})
    assert config.family_config["history"] == (0, 1)
    with pytest.raises(TypeError):
        config.family_config["history"] = (2,)  # type: ignore[index]


class _FakeAdapter:
    def initial_state(self, batch_size: int) -> object:
        return np.zeros((batch_size, 1), dtype=np.float32)

    def clone_state(self, state: object) -> object:
        return np.asarray(state).copy()

    def expand_state(self, state: object, batch_size: int) -> object:
        value = np.asarray(state)
        if value.shape[0] != 1 or batch_size < 1:
            raise ValueError
        return np.repeat(value, batch_size, axis=0)

    def build_policy_inputs(
        self,
        state: RobotState,
        motion_command: MotionCommand,
        bfm_state: object,
    ) -> tuple[dict[str, np.ndarray], object]:
        return {"observation": state.joint_position}, self.clone_state(bfm_state)

    def map_policy_outputs(
        self, outputs: dict[str, np.ndarray], bfm_state: object
    ) -> tuple[PDCommand, object]:
        action = outputs["action"]
        zeros = np.zeros_like(action)
        command = PDCommand(JOINTS, action, zeros, zeros, zeros, zeros)
        return command, self.clone_state(bfm_state)


def test_registry_is_explicit_and_closed() -> None:
    family = "test_registry_family"

    @register(family)
    def factory(config: BFMInstanceConfig) -> BFMAdapter:
        assert config.family == family
        return _FakeAdapter()

    adapter = get_family(family)(_config(family))

    assert isinstance(adapter, BFMAdapter)
    assert family in available()
    with pytest.raises(ValueError, match="already registered"):
        register(family)(factory)
    with pytest.raises(KeyError, match="available"):
        get_family("missing_family")


def test_currency_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="body_position_world"):
        RobotState(
            joint_names=("joint",),
            body_names=("pelvis",),
            joint_position=np.zeros((2, 1), dtype=np.float32),
            joint_velocity=np.zeros((2, 1), dtype=np.float32),
            body_position_world=np.zeros((1, 1, 3), dtype=np.float32),
            body_orientation_world=np.zeros((2, 1, 4), dtype=np.float32),
            body_linear_velocity_world=np.zeros((2, 1, 3), dtype=np.float32),
            body_angular_velocity_world=np.zeros((2, 1, 3), dtype=np.float32),
        )
