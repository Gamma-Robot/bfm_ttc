from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

mujoco = pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from bfm_adapters import DevicePDCommand, PDCommand
from command_optimization.simulation.mjwarp import MjwarpWorld
from command_optimization.simulation.randomization import ScenarioBatch
from command_optimization.simulation.execution_state import ExecutionState


JOINTS = ("joint_a", "joint_b")
BODIES = ("root", "link_a", "link_b")
MJCF = """
<mujoco model="task7_probe">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="root" pos="0 0 0.5">
      <freejoint name="root_free"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link_a" pos="0 0 0.2">
        <joint name="joint_a" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1" mass="0.2"/>
        <body name="link_b" pos="0 0 0.2">
          <joint name="joint_b" type="hinge" axis="1 0 0"/>
          <geom type="capsule" size="0.02 0.1" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "probe.xml"
    path.write_text(MJCF)
    return path


def _world(
    model_path: Path,
    *,
    batch_size: int = 1,
    scenario_count: int = 1,
    motion_command_velocity_stencil: str | None = None,
) -> MjwarpWorld:
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp CUDA device is unavailable")
    options = (
        {}
        if motion_command_velocity_stencil is None
        else {"motion_command_velocity_stencil": motion_command_velocity_stencil}
    )
    return MjwarpWorld(
        model_path,
        batch_size=batch_size,
        joint_names=JOINTS,
        body_names=BODIES,
        physics_timestep=0.002,
        joint_effort_limit=np.asarray([5.0, 7.0]),
        floor_friction=(1.0, 0.005, 0.0001),
        joint_armature=np.asarray([0.01, 0.02]),
        joint_frictionloss=np.asarray([0.03, 0.04]),
        scenario_count=scenario_count,
        nconmax=16,
        njmax=32,
        **options,
    )


def _execution_state(model_path: Path, batch_size: int) -> ExecutionState:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = np.repeat(model.qpos0[None], batch_size, axis=0)
    qvel = np.zeros((batch_size, model.nv))
    return ExecutionState(qpos, qvel, 0, 0.0, 0, None, {"lane": batch_size})


def _scenarios(
    *,
    contact: tuple[float, ...],
    armature: tuple[float, ...],
    frictionloss: tuple[float, ...],
    actuation: tuple[float, ...],
) -> ScenarioBatch:
    count = len(contact)
    return ScenarioBatch(
        contact_friction_scale=np.asarray(contact),
        joint_armature_scale=np.asarray(armature),
        joint_frictionloss_scale=np.asarray(frictionloss),
        actuation_strength_scale=np.asarray(actuation),
        root_linear_velocity_offset=np.zeros((count, 3)),
        root_angular_velocity_offset=np.zeros((count, 3)),
        joint_velocity_offset=np.zeros((count, len(JOINTS))),
    )


def test_reset_composition_and_pd_force_mapping(model_path: Path) -> None:
    world = _world(model_path)
    world.reset(_execution_state(model_path, 1))
    np.testing.assert_allclose(world.state_view.joint_position.numpy(), 0.0)

    # Qualification intentionally inspects MJWarp force storage: this proves
    # controlled dofs are mapped and all other generalized forces stay zero.
    assert world._model.nu == 0
    assert mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    np.testing.assert_allclose(
        world._model.dof_armature[world._dof_addresses_np], [0.01, 0.02]
    )
    np.testing.assert_allclose(
        world._model.dof_frictionloss[world._dof_addresses_np], [0.03, 0.04]
    )

    target = np.asarray([[1.0, 0.0]], dtype=np.float32)
    zeros = np.zeros_like(target)
    command = PDCommand(
        JOINTS,
        target,
        zeros,
        zeros,
        np.asarray([[10.0, 20.0]], dtype=np.float32),
        zeros,
    )
    world.apply_pd(command)
    world.step()
    wp.synchronize()
    force = np.asarray(world._data.qfrc_applied.numpy())[0]
    np.testing.assert_allclose(force[:6], 0.0, atol=1e-7)
    assert force[world._dof_addresses_np[0]] == pytest.approx(5.0)
    assert force[world._dof_addresses_np[1]] == pytest.approx(0.0)
    _, qvel, time = world.physical_state()
    assert qvel[0, world._dof_addresses_np[0]] > 0.0
    assert time[0] == pytest.approx(0.002, abs=1e-7)

    # The first call executes eagerly while capturing the graph. Subsequent
    # calls, including after reset, must advance exactly one requested step.
    assert tuple(world._step_graphs) == (1,)
    world.step()
    assert world.physical_state()[2][0] == pytest.approx(0.004, abs=1e-7)
    world.reset(_execution_state(model_path, 1))
    world.apply_pd(command)
    world.step()
    assert world.physical_state()[2][0] == pytest.approx(0.002, abs=1e-7)


def test_post_step_bodies_match_physical_state_and_restore(model_path: Path) -> None:
    world = _world(model_path)
    initial = _execution_state(model_path, 1)
    initial.qvel[0, :6] = [1.0, 0.2, 0.0, 0.3, 0.1, 0.2]
    world.reset(initial)
    zeros = np.zeros((1, len(JOINTS)))
    command = PDCommand(JOINTS, zeros, zeros, zeros, zeros, zeros)
    oracle = mujoco.MjData(world._model)

    for _ in range(2):  # Eager warm-up and captured replay must agree.
        world.apply_pd(command)
        world.step(4)
        qpos, qvel, time = world.physical_state()
        state = world.robot_state()
        oracle.qpos[:] = qpos[0]
        oracle.qvel[:] = qvel[0]
        mujoco.mj_forward(world._model, oracle)
        bodies = world._body_ids_np
        expected = {
            "body_position_world": oracle.xpos[bodies],
            "body_orientation_world": oracle.xquat[bodies],
            "body_linear_velocity_world": oracle.cvel[bodies, 3:],
            "body_angular_velocity_world": oracle.cvel[bodies, :3],
        }
        for name, value in expected.items():
            np.testing.assert_allclose(getattr(state, name)[0], value, atol=2e-6)

        world.reset(ExecutionState(qpos, qvel, 0, float(time[0]), 0, None, {}))
        restored = world.robot_state()
        for name in expected:
            np.testing.assert_allclose(
                getattr(restored, name), getattr(state, name), atol=2e-6
            )


def test_device_command_view_avoids_host_round_trip(model_path: Path) -> None:
    world = _world(model_path, batch_size=2)
    world.reset(_execution_state(model_path, 2))
    command = world.command_view
    assert isinstance(command, DevicePDCommand)
    values = {
        "position_target": np.asarray([[0.4, -0.3], [0.2, 0.1]], dtype=np.float32),
        "velocity_target": np.zeros((2, 2), dtype=np.float32),
        "feedforward_torque": np.zeros((2, 2), dtype=np.float32),
        "proportional_gain": np.full((2, 2), 5.0, dtype=np.float32),
        "derivative_gain": np.full((2, 2), 0.2, dtype=np.float32),
    }
    for name, value in values.items():
        destination = getattr(command, name)
        wp.copy(destination, wp.array(value, dtype=wp.float32, device=destination.device))

    world.apply_pd(command)
    world.step()

    materialized = world.host_command()
    for name, expected in values.items():
        np.testing.assert_array_equal(getattr(materialized, name), expected)


def test_scenario_rows_scale_model_and_actuation_in_place(model_path: Path) -> None:
    world = _world(model_path, batch_size=4, scenario_count=2)
    scenarios = _scenarios(
        contact=(0.5, 1.5),
        armature=(1.0, 1.0),
        frictionloss=(0.5, 2.0),
        actuation=(0.5, 1.0),
    )
    world.configure_scenarios(scenarios)

    assert world.scenario_count == 2
    friction = np.asarray(world._model_wp.geom_friction.numpy())
    armature = np.asarray(world._model_wp.dof_armature.numpy())
    frictionloss = np.asarray(world._model_wp.dof_frictionloss.numpy())
    collidable = world._collidable_geom_ids_np
    np.testing.assert_allclose(
        friction[0, collidable], world._nominal_geom_friction[collidable] * 0.5
    )
    np.testing.assert_allclose(
        friction[1, collidable], world._nominal_geom_friction[collidable] * 1.5
    )
    np.testing.assert_allclose(
        armature[:, world._dof_addresses_np],
        np.repeat(
            world._nominal_dof_armature[None, world._dof_addresses_np], 2, axis=0
        ),
    )
    np.testing.assert_allclose(
        frictionloss[:, world._dof_addresses_np],
        world._nominal_dof_frictionloss[None, world._dof_addresses_np]
        * np.asarray([[0.5], [2.0]]),
    )

    pointers = (
        world._model_wp.geom_friction.ptr,
        world._model_wp.dof_armature.ptr,
        world._model_wp.dof_frictionloss.ptr,
        world._actuation_strength.ptr,
    )
    world.reset(_execution_state(model_path, 4))
    target = np.zeros((4, 2), dtype=np.float32)
    target[:, 0] = 1.0
    zeros = np.zeros_like(target)
    world.apply_pd(
        PDCommand(JOINTS, target, zeros, zeros, np.full_like(target, 10.0), zeros)
    )
    world.step()
    forces = np.asarray(world._data.qfrc_applied.numpy())[
        :, world._dof_addresses_np[0]
    ]
    np.testing.assert_allclose(forces, [2.5, 5.0, 2.5, 5.0])

    world.configure_scenarios(
        _scenarios(
            contact=(1.5, 0.5),
            armature=(1.0, 1.0),
            frictionloss=(2.0, 0.5),
            actuation=(1.0, 0.5),
        )
    )
    assert pointers == (
        world._model_wp.geom_friction.ptr,
        world._model_wp.dof_armature.ptr,
        world._model_wp.dof_frictionloss.ptr,
        world._actuation_strength.ptr,
    )
    world.reset(_execution_state(model_path, 4))
    world.apply_pd(
        PDCommand(JOINTS, target, zeros, zeros, np.full_like(target, 10.0), zeros)
    )
    world.step()
    swapped = np.asarray(world._data.qfrc_applied.numpy())[
        :, world._dof_addresses_np[0]
    ]
    np.testing.assert_allclose(swapped, [5.0, 2.5, 5.0, 2.5])
    assert tuple(world._step_graphs) == (1,)


def test_armature_refresh_preserves_graph_and_scenario_mapping(
    model_path: Path,
) -> None:
    world = _world(model_path, batch_size=4, scenario_count=2)
    first = _scenarios(
        contact=(1.0, 1.0),
        armature=(0.2, 5.0),
        frictionloss=(1.0, 1.0),
        actuation=(1.0, 1.0),
    )
    world.configure_scenarios(first)
    derived_pointers = (
        world._model_wp.dof_invweight0.ptr,
        world._model_wp.body_invweight0.ptr,
        world._model_wp.stat.meaninertia.ptr,
    )

    target = np.zeros((4, 2), dtype=np.float32)
    target[:, 0] = 1.0
    zeros = np.zeros_like(target)
    command = PDCommand(
        JOINTS, target, zeros, zeros, np.full_like(target, 10.0), zeros
    )

    def rollout() -> np.ndarray:
        world.reset(_execution_state(model_path, 4))
        world.apply_pd(command)
        for _ in range(4):
            world.step()
        return world.physical_state()[1][:, world._dof_addresses_np[0]]

    before = rollout()
    np.testing.assert_allclose(before[[0, 2]], before[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(before[[1, 3]], before[1], rtol=1e-5, atol=1e-7)
    assert abs(before[0] - before[1]) > 1e-3

    world.configure_scenarios(
        _scenarios(
            contact=(1.0, 1.0),
            armature=(5.0, 0.2),
            frictionloss=(1.0, 1.0),
            actuation=(1.0, 1.0),
        )
    )
    after = rollout()
    assert derived_pointers == (
        world._model_wp.dof_invweight0.ptr,
        world._model_wp.body_invweight0.ptr,
        world._model_wp.stat.meaninertia.ptr,
    )
    np.testing.assert_allclose(after[[0, 2]], after[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(after[[1, 3]], after[1], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(after[0], before[1], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(after[1], before[0], rtol=1e-4, atol=1e-6)
    assert tuple(world._step_graphs) == (1,)


def test_projection_uses_the_composed_simulation_model(model_path: Path) -> None:
    world = _world(model_path)
    root_position = np.asarray([[[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]])
    root_orientation = np.zeros((1, 2, 4))
    root_orientation[..., 0] = 1.0
    joint_position = np.asarray([[[0.0, 0.0], [0.2, -0.1]]])
    motion_command = world.project(
        query_offsets=(0, 1),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=root_position,
        root_orientation_world=root_orientation,
        joint_position=joint_position,
        control_period=0.02,
    )
    assert motion_command.joint_position.dtype == np.float64
    np.testing.assert_allclose(motion_command.body_position_world[:, :, 0], root_position)
    assert np.isfinite(motion_command.body_position_world).all()
    assert np.isfinite(motion_command.body_linear_velocity_world).all()
    qpos, qvel = world.physical_trajectory(motion_command, control_period=0.02)
    assert qpos.shape == (1, 2, world._model.nq)
    assert qvel.shape == (1, 2, world._model.nv)
    np.testing.assert_allclose(qpos[:, :, world._qpos_addresses_np], joint_position)
    np.testing.assert_allclose(qpos[:, :, :3], root_position)
    assert np.isfinite(qvel).all()
    expected_position = np.empty_like(motion_command.body_position_world)
    expected_orientation = np.empty_like(motion_command.body_orientation_world)
    expected_linear_velocity = np.empty_like(motion_command.body_linear_velocity_world)
    expected_angular_velocity = np.empty_like(motion_command.body_angular_velocity_world)
    data = mujoco.MjData(world._model)
    for step_index in range(qpos.shape[1]):
        data.qpos[:] = qpos[0, step_index]
        data.qvel[:] = qvel[0, step_index]
        mujoco.mj_forward(world._model, data)
        expected_position[0, step_index] = data.xpos[world._body_ids_np]
        expected_orientation[0, step_index] = data.xquat[world._body_ids_np]
        expected_angular_velocity[0, step_index] = data.cvel[world._body_ids_np, :3]
        expected_linear_velocity[0, step_index] = data.cvel[world._body_ids_np, 3:]
    np.testing.assert_allclose(
        motion_command.body_position_world, expected_position, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        motion_command.body_orientation_world,
        expected_orientation,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        motion_command.body_linear_velocity_world,
        expected_linear_velocity,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        motion_command.body_angular_velocity_world,
        expected_angular_velocity,
        rtol=1e-5,
        atol=1e-6,
    )

    first_projection = world._projection_data[2]
    world.project(
        query_offsets=(0, 1),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=np.repeat(root_position, 2, axis=0),
        root_orientation_world=np.repeat(root_orientation, 2, axis=0),
        joint_position=np.repeat(joint_position, 2, axis=0),
        control_period=0.02,
    )
    repeated = world.project(
        query_offsets=(0, 1),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=root_position,
        root_orientation_world=root_orientation,
        joint_position=joint_position,
        control_period=0.02,
    )
    assert set(world._projection_data) == {2, 4}
    assert world._projection_data[2] is first_projection
    for field in (
        "joint_position",
        "joint_velocity",
        "body_position_world",
        "body_orientation_world",
        "body_linear_velocity_world",
        "body_angular_velocity_world",
    ):
        np.testing.assert_allclose(getattr(repeated, field), getattr(motion_command, field))


def test_backward_projection_matches_cvel_convention(
    model_path: Path,
) -> None:
    world = _world(model_path, motion_command_velocity_stencil="backward")
    root_position = np.asarray(
        [[[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.08, 0.0, 0.5]]]
    )
    root_orientation = np.zeros((1, 3, 4))
    root_orientation[0, 0] = (
        np.cos(0.3),
        np.sin(0.3),
        0.0,
        0.0,
    )
    pitch_delta = np.asarray(
        (np.cos(0.2), 0.0, np.sin(0.2), 0.0), dtype=np.float64
    )
    yaw_delta = np.asarray(
        (np.cos(0.1), 0.0, 0.0, np.sin(0.1)), dtype=np.float64
    )
    mujoco.mju_mulQuat(
        root_orientation[0, 1], root_orientation[0, 0], pitch_delta
    )
    mujoco.mju_mulQuat(
        root_orientation[0, 2], root_orientation[0, 1], yaw_delta
    )
    joint_position = np.asarray([[[0.0, 0.0], [0.1, 0.2], [0.4, 0.3]]])
    motion_command = world.project(
        query_offsets=(0, 1, 2),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=root_position,
        root_orientation_world=root_orientation,
        joint_position=joint_position,
        control_period=0.02,
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    root_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_free"
    )
    root_qpos = int(model.jnt_qposadr[root_joint])
    root_dof = int(model.jnt_dofadr[root_joint])
    qpos[:, root_qpos : root_qpos + 3] = root_position[0]
    qpos[:, root_qpos + 3 : root_qpos + 7] = root_orientation[0]
    for column, name in enumerate(JOINTS):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, int(model.jnt_qposadr[joint])] = joint_position[0, :, column]

    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODIES]
    )
    expected = np.empty((3, len(BODIES), 6))
    data = mujoco.MjData(model)
    for step, (left, right) in enumerate(((0, 1), (0, 1), (1, 2))):
        velocity = np.zeros(model.nv)
        mujoco.mj_differentiatePos(
            model, velocity, 0.02, qpos[left], qpos[right]
        )
        # The backward projection rotates the local quaternion delta by
        # the preceding root orientation before MuJoCo forward kinematics.
        angular_delta = np.zeros(3)
        mujoco.mju_subQuat(
            angular_delta,
            qpos[right, root_qpos + 3 : root_qpos + 7],
            qpos[left, root_qpos + 3 : root_qpos + 7],
        )
        current_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(
            current_rotation,
            qpos[left, root_qpos + 3 : root_qpos + 7],
        )
        velocity[root_dof + 3 : root_dof + 6] = (
            current_rotation.reshape(3, 3) @ angular_delta / 0.02
        )
        data.qpos[:] = qpos[step]
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
        expected[step] = data.cvel[body_ids]

    np.testing.assert_allclose(
        motion_command.body_angular_velocity_world[0],
        expected[:, :, :3],
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        motion_command.body_linear_velocity_world[0],
        expected[:, :, 3:],
        rtol=1.0e-5,
        atol=1.0e-6,
    )
