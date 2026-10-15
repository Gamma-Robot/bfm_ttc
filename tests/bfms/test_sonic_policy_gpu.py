from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from bfm_adapters import MotionCommand, RobotState
from bfm_adapters.families.sonic.adapter import SonicAdapter
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from bfm_adapters.registry import load_instance_config
from command_optimization.artifacts import ensure_artifact


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
SONIC_POLICIES = (
    (
        "sonic_release_g1.yaml",
        "sonic-release-g1-policy",
        "sonic-release-g1-policy.batched",
    ),
    (
        "sonic_low_latency_g1.yaml",
        "sonic-low-latency-g1-policy",
        "sonic-low-latency-g1-policy.batched",
    ),
)


def _adapter_inputs(adapter: SonicAdapter, config, batch: int):
    joints = len(config.policy_joint_names)
    bodies = len(config.motion_command_body_names)
    queries = len(config.motion_command_query_offsets)
    default = np.asarray(config.family_config["default_joint_position"], np.float32)
    lanes = np.arange(batch, dtype=np.float32)[:, None]
    joint_phase = np.linspace(0.0, 1.0, joints, dtype=np.float32)[None]

    orientation = np.zeros((batch, bodies, 4), np.float32)
    orientation[..., 0] = 1.0
    root = config.motion_command_body_names.index(str(config.family_config["root_body"]))
    yaw = lanes[:, 0] * 0.03
    orientation[:, root, 0] = np.cos(yaw / 2.0)
    orientation[:, root, 3] = np.sin(yaw / 2.0)
    angular_velocity = np.zeros((batch, bodies, 3), np.float32)
    angular_velocity[:, root, 2] = lanes[:, 0] * 0.02
    state = RobotState(
        joint_names=config.policy_joint_names,
        body_names=config.motion_command_body_names,
        joint_position=default + 0.02 * np.sin(joint_phase + lanes * 0.2),
        joint_velocity=0.04 * np.cos(joint_phase + lanes * 0.2),
        body_position_world=np.zeros((batch, bodies, 3), np.float32),
        body_orientation_world=orientation,
        body_linear_velocity_world=np.zeros((batch, bodies, 3), np.float32),
        body_angular_velocity_world=angular_velocity,
    )

    time = np.linspace(0.0, 0.5, queries, dtype=np.float32)[None, :, None]
    phase = time + joint_phase[:, None] + lanes[:, None] * 0.2
    guide_position = default[None, None] + 0.10 * np.sin(phase)
    guide_velocity = 0.05 * np.cos(phase)
    guide_orientation = np.zeros((batch, queries, bodies, 4), np.float32)
    guide_orientation[..., 0] = 1.0
    guide_yaw = yaw[:, None] + np.linspace(
        0.0, 0.1, queries, dtype=np.float32
    )[None]
    guide_orientation[:, :, root, 0] = np.cos(guide_yaw / 2.0)
    guide_orientation[:, :, root, 3] = np.sin(guide_yaw / 2.0)
    motion_command = MotionCommand(
        query_offsets=config.motion_command_query_offsets,
        joint_names=config.policy_joint_names,
        body_names=config.motion_command_body_names,
        joint_position=guide_position.astype(np.float32),
        joint_velocity=guide_velocity.astype(np.float32),
        body_position_world=np.zeros((batch, queries, bodies, 3), np.float32),
        body_orientation_world=guide_orientation,
        body_linear_velocity_world=np.zeros((batch, queries, bodies, 3), np.float32),
        body_angular_velocity_world=np.zeros((batch, queries, bodies, 3), np.float32),
    )
    return adapter.build_policy_inputs(
        state, motion_command, adapter.initial_state(batch)
    )


@pytest.mark.parametrize(
    ("config_name", "source_id", "prepared_id"),
    SONIC_POLICIES,
    ids=("release", "low_latency"),
)
def test_sonic_prepared_policy_matches_source_for_batch_one_and_many(
    config_name: str, source_id: str, prepared_id: str
) -> None:
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    device = int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))
    config = load_instance_config(REPO / "configs" / "bfms" / config_name)
    adapter = SonicAdapter(config)
    source = ensure_artifact(source_id, root=REPO)
    prepared = ensure_artifact(prepared_id, root=REPO)
    ort.preload_dlls(directory="")
    raw = ort.InferenceSession(
        str(source),
        providers=[("CUDAExecutionProvider", {"device_id": str(device), "use_tf32": "0"})],
    )
    batched = BatchedOrtCudaPolicy(prepared, config, device_id=device)
    assert raw.get_providers()[0] == "CUDAExecutionProvider"
    assert batched._session.get_providers()[0] == "CUDAExecutionProvider"
    inputs, bfm_state = _adapter_inputs(adapter, config, batch=4)

    expected = np.stack(
        [
            raw.run(
                ["action"],
                {name: value[lane] for name, value in inputs.items()},
            )[0]
            for lane in range(4)
        ]
    )
    one = batched.run({name: value[:1] for name, value in inputs.items()})[
        "action"
    ]
    actual = batched.run(inputs)["action"]
    perturbed_inputs = {name: value.copy() for name, value in inputs.items()}
    for value in perturbed_inputs.values():
        value[1:] = value[:0:-1]
    perturbed = batched.run(perturbed_inputs)["action"]
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    actual_command, _ = adapter.map_policy_outputs(
        {"action": actual}, bfm_state
    )
    expected_command, _ = adapter.map_policy_outputs(
        {"action": expected}, bfm_state
    )

    assert np.max(np.abs(one[0] - expected[0])) <= 1.0e-6
    np.testing.assert_array_equal(perturbed[0], actual[0])
    assert np.max(np.abs(perturbed[1:] - actual[1:])) > 1.0e-2
    assert np.max(np.abs(difference)) <= 1.0e-3
    assert np.sqrt(np.mean(difference**2)) <= 3.0e-4
    assert (
        np.max(
            np.abs(
                actual_command.position_target - expected_command.position_target
            )
        )
        <= 5.0e-4
    )
