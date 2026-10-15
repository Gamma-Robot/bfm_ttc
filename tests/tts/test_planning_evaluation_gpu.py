from __future__ import annotations

import gc
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from bfm_adapters import MotionCommand, PDCommand
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.motion_tracking.device_cost import MotionTrackingCostReducer
from command_optimization.motion_tracking.search_space import KnotScales, QposKnotsSpace
from command_optimization.planning.evaluation import EvaluationLedger, RolloutEvaluator
from command_optimization.simulation.command_execution import CommandExecutor
from command_optimization.simulation.mjwarp import MjwarpWorld
from command_optimization.simulation.execution_state import ExecutionState


JOINTS = ("joint_a", "joint_b")
BODIES = ("root", "link_a", "link_b")
HORIZON = 2
NUM_CANDIDATES = 1024
BEST_ID = 713
MJCF = """
<mujoco model="planning_g2">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="root" pos="0 0 0.5">
      <freejoint name="root_free"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <body name="link_a" pos="0 0 0.2">
        <joint name="joint_a" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0.1" mass="0.2" diaginertia="0.001 0.001 0.001"/>
        <body name="link_b" pos="0 0 0.2">
          <joint name="joint_b" type="hinge" axis="1 0 0"/>
          <inertial pos="0 0 0.1" mass="0.2" diaginertia="0.001 0.001 0.001"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _Projector:
    def project(
        self,
        *,
        query_offsets,
        joint_names,
        body_names,
        root_position_world,
        root_orientation_world,
        joint_position,
        control_period,
    ) -> MotionCommand:
        batch, horizon = joint_position.shape[:2]
        joint_velocity = np.gradient(joint_position, control_period, axis=1)
        body_position = np.repeat(root_position_world[:, :, None], len(BODIES), axis=2)
        body_position[:, :, 1, 2] += 0.2
        body_position[:, :, 2, 2] += 0.4
        body_orientation = np.repeat(
            root_orientation_world[:, :, None], len(BODIES), axis=2
        )
        body_linear_velocity = np.gradient(body_position, control_period, axis=1)
        body_angular_velocity = np.zeros((batch, horizon, len(BODIES), 3))
        return MotionCommand(
            query_offsets=query_offsets,
            joint_names=joint_names,
            body_names=body_names,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            body_position_world=body_position,
            body_orientation_world=body_orientation,
            body_linear_velocity_world=body_linear_velocity,
            body_angular_velocity_world=body_angular_velocity,
        )


class _StateAdapter:
    def expand_state(self, state: object, batch_size: int) -> object:
        assert state == ("initial",)
        return tuple(0 for _ in range(batch_size))


class _PdBFM:
    def __init__(self) -> None:
        self._state: object = ()

    def reset(self, state: object) -> None:
        self._state = tuple(state)  # type: ignore[arg-type]

    def capture_state(self) -> object:
        return tuple(self._state)  # type: ignore[arg-type]

    def step(self, state_view: object, motion_command: MotionCommand) -> PDCommand:
        del state_view
        target = motion_command.joint_position[:, 0]
        zeros = np.zeros_like(target)
        return PDCommand(
            JOINTS,
            target,
            zeros,
            zeros,
            np.full_like(target, 20.0),
            np.full_like(target, 1.0),
        )


def _reference_motion() -> MotionCommand:
    root_position = np.zeros((1, HORIZON, 3))
    root_position[..., 2] = 0.5
    root_orientation = np.zeros((1, HORIZON, 4))
    root_orientation[..., 0] = 1.0
    return _Projector().project(
        query_offsets=tuple(range(HORIZON)),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=root_position,
        root_orientation_world=root_orientation,
        joint_position=np.zeros((1, HORIZON, len(JOINTS))),
        control_period=0.002,
    )


def _objective() -> MotionTrackingObjective:
    return MotionTrackingObjective(
        body_names=BODIES,
        tracking_joint_position_weight=1.0,
        tracking_joint_velocity_weight=0.0,
        tracking_body_position_weight=0.0,
        tracking_body_orientation_weight=0.0,
        deformation_joint_position_weight=0.1,
        deformation_body_position_weight=0.0,
        deformation_body_orientation_weight=0.0,
        joint_speed_weight=0.0,
        terminal_joint_position_weight=1.0,
        terminal_body_position_weight=0.0,
    )


def _execution_state(model_path: Path) -> ExecutionState:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return ExecutionState(
        qpos=model.qpos0[None].copy(),
        qvel=np.zeros((1, model.nv)),
        control_step=0,
        time=0.0,
        motion_command_index=0,
        command=None,
        bfm_state=("initial",),
    )


def _evaluate(
    model_path: Path,
    search_space: QposKnotsSpace,
    reference_motion: MotionCommand,
    candidates: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, EvaluationLedger]:
    worlds: dict[int, MjwarpWorld] = {}
    objective = _objective()

    def executor_factory(batch_size: int) -> CommandExecutor:
        world = MjwarpWorld(
            model_path,
            batch_size=batch_size,
            joint_names=JOINTS,
            body_names=BODIES,
            physics_timestep=0.002,
            joint_effort_limit=np.full(len(JOINTS), 100.0),
            joint_armature=np.zeros(len(JOINTS)),
            joint_frictionloss=np.zeros(len(JOINTS)),
            nconmax=1,
            njmax=8,
        )
        worlds[batch_size] = world
        return CommandExecutor(
            world,
            _PdBFM(),  # type: ignore[arg-type]
            control_period=0.002,
            motion_command_query_offsets=(0,),
        )

    evaluator = RolloutEvaluator(
        search_space=search_space,
        objective=objective,
        reference_motion=reference_motion,
        execution_state=_execution_state(model_path),
        bfm_adapter=_StateAdapter(),  # type: ignore[arg-type]
        executor_factory=executor_factory,
        max_worlds=capacity,
        failure_penalty=1e12,
        reducer_factory=lambda batch: MotionTrackingCostReducer(
            objective, worlds[batch].state_view
        ),
    )
    result = evaluator.evaluate(candidates)
    return result.costs, result.ledger


@pytest.mark.gpu
@pytest.mark.conformance
def test_g2_candidate_costs_and_identity_are_wave_size_invariant(
    tmp_path: Path,
) -> None:
    """G2 uses real MJWarp execution; lack of CUDA is a gate failure, not a skip."""
    wp.init()
    assert wp.is_cuda_available(), "G2 requires a CUDA device"
    model_path = tmp_path / "planning_g2.xml"
    model_path.write_text(MJCF)
    reference_motion = _reference_motion()
    search_space = QposKnotsSpace(
        reference_motion,
        num_knots=HORIZON,
        control_period=0.002,
        scales=KnotScales(0.0, 0.0, 0.0, 0.2),
        projector=_Projector(),
        pelvis_body="root",
    )
    candidates = np.random.default_rng(47).uniform(
        -0.8,
        0.8,
        (NUM_CANDIDATES, *search_space.latent_shape),
    ).astype(np.float64)
    candidates[BEST_ID] = 0.0

    one_wave, one_ledger = _evaluate(
        model_path, search_space, reference_motion, candidates, NUM_CANDIDATES
    )
    gc.collect()
    two_waves, two_ledger = _evaluate(
        model_path, search_space, reference_motion, candidates, NUM_CANDIDATES // 2
    )

    np.testing.assert_allclose(one_wave, two_waves, rtol=1e-6, atol=1e-8)
    assert int(np.argmin(one_wave)) == BEST_ID
    assert int(np.argmin(two_waves)) == BEST_ID
    assert one_ledger.logical_rollouts == two_ledger.logical_rollouts == 1024
    assert one_ledger.physical_rollouts == two_ledger.physical_rollouts == 1024
    one_wave_ranges = [
        (wave.candidate_start, wave.candidate_stop)
        for wave in one_ledger.waves
    ]
    two_wave_ranges = [
        (wave.candidate_start, wave.candidate_stop)
        for wave in two_ledger.waves
    ]
    assert one_wave_ranges == [
        (0, 1024)
    ]
    assert two_wave_ranges == [
        (0, 512),
        (512, 1024),
    ]
    assert not any(wave.failure for wave in (*one_ledger.waves, *two_ledger.waves))
