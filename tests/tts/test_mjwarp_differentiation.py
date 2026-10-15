from __future__ import annotations

import mujoco
import numpy as np

from command_optimization.simulation.mjwarp import MjwarpWorld


def test_vectorized_g1_qpos_differentiation_matches_mujoco() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <freejoint name="root_joint"/>
              <geom type="sphere" size="0.1" mass="1"/>
              <body name="link" pos="0 0 0.2">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.03 0.1" mass="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    world = object.__new__(MjwarpWorld)
    world._model = model
    world._root_qpos_address = int(model.jnt_qposadr[root_id])
    world._root_dof_address = int(model.jnt_dofadr[root_id])
    world._qpos_addresses_np = np.asarray([model.jnt_qposadr[hinge_id]])
    world._dof_addresses_np = np.asarray([model.jnt_dofadr[hinge_id]])
    world._motion_command_velocity_stencil = "centered"

    rng = np.random.default_rng(47)
    qpos = np.broadcast_to(model.qpos0, (4, 7, model.nq)).copy()
    qpos[:, :, :3] = rng.normal(size=(4, 7, 3))
    quaternions = rng.normal(size=(4, 7, 4))
    qpos[:, :, 3:7] = quaternions / np.linalg.norm(
        quaternions, axis=-1, keepdims=True
    )
    qpos[:, :, int(model.jnt_qposadr[hinge_id])] = rng.normal(size=(4, 7))

    actual = world._differentiate(qpos, 0.02)
    expected = np.zeros_like(actual)
    for lane in range(qpos.shape[0]):
        for step_index in range(qpos.shape[1]):
            left = max(step_index - 1, 0)
            right = min(step_index + 1, qpos.shape[1] - 1)
            mujoco.mj_differentiatePos(
                model,
                expected[lane, step_index],
                (right - left) * 0.02,
                qpos[lane, left],
                qpos[lane, right],
            )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
