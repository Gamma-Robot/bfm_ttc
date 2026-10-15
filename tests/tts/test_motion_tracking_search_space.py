from dataclasses import replace

import numpy as np
import pytest

from bfm_adapters.contracts import MotionCommand
from command_optimization.motion_tracking.search_space import (
    KnotScales,
    QposKnotsSpace,
    _quat_multiply,
    _rotvec_to_quat,
)

JOINTS = ("hip", "knee", "ankle")
BODIES = ("torso_link", "pelvis")
PELVIS = BODIES.index("pelvis")
H = 9
K = 3
D = 6 + len(JOINTS)
KNOT_STEPS = (0, 4, 8)
SCALES = KnotScales(
    root_xy_m=0.1, root_z_m=0.2, root_rotation_rad=0.3, joint_position_rad=0.4
)
ARRAY_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)


def _stable_unit_quats(rng, count):
    """Unit quaternions with w > 0 that are bitwise fixpoints of normalization."""
    quats = []
    while len(quats) < count:
        quat = rng.standard_normal(4)
        quat[0] = abs(quat[0]) + 0.5
        quat /= np.linalg.norm(quat)
        if np.linalg.norm(quat) == 1.0:
            quats.append(quat)
    return np.array(quats)


def _quat_rotate(quaternion, vector):
    zeros = np.zeros((*vector.shape[:-1], 1), dtype=vector.dtype)
    pure = np.concatenate((zeros, vector), axis=-1)
    conjugate = quaternion.copy()
    conjugate[..., 1:] *= -1
    return _quat_multiply(_quat_multiply(quaternion, pure), conjugate)[..., 1:]


class _SyntheticProjector:
    """Small nonlinear kinematic oracle; torso pose depends on every joint."""

    def __init__(self):
        self.calls = 0

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
    ):
        self.calls += 1
        batch, horizon, joints = joint_position.shape
        assert joints == len(joint_names)
        body_position = np.empty((batch, horizon, len(body_names), 3))
        body_orientation = np.empty((batch, horizon, len(body_names), 4))
        pelvis = body_names.index("pelvis")
        torso = body_names.index("torso_link")
        body_position[:, :, pelvis] = root_position_world
        body_orientation[:, :, pelvis] = root_orientation_world
        torso_offset = np.stack(
            (
                0.2 + 0.1 * joint_position[:, :, 0],
                0.1 * joint_position[:, :, 1],
                0.5 + 0.1 * joint_position[:, :, 2],
            ),
            axis=-1,
        )
        body_position[:, :, torso] = root_position_world + _quat_rotate(
            root_orientation_world, torso_offset
        )
        torso_delta = _rotvec_to_quat(0.2 * joint_position)
        body_orientation[:, :, torso] = _quat_multiply(
            root_orientation_world, torso_delta
        )
        def derivative(value):
            if horizon == 1:
                return np.zeros_like(value)
            return np.gradient(value, control_period, axis=1)
        return MotionCommand(
            query_offsets=query_offsets,
            joint_names=joint_names,
            body_names=body_names,
            joint_position=joint_position.copy(),
            joint_velocity=derivative(joint_position),
            body_position_world=body_position,
            body_orientation_world=body_orientation,
            body_linear_velocity_world=derivative(body_position),
            body_angular_velocity_world=derivative(body_orientation[..., 1:]),
        )


def _reference(horizon=H, batch=1, seed=0, control_period=0.02):
    rng = np.random.default_rng(seed)
    root_position = rng.standard_normal((batch, horizon, 3))
    root_orientation = _stable_unit_quats(rng, batch * horizon).reshape(
        batch, horizon, 4
    )
    joint_position = rng.uniform(-0.5, 0.5, (batch, horizon, len(JOINTS)))
    return _SyntheticProjector().project(
        query_offsets=tuple(range(horizon)),
        joint_names=JOINTS,
        body_names=BODIES,
        root_position_world=root_position,
        root_orientation_world=root_orientation,
        joint_position=joint_position,
        control_period=control_period,
    )


def _space(reference=None, **kwargs):
    defaults = dict(
        num_knots=K,
        control_period=0.02,
        scales=SCALES,
        projector=_SyntheticProjector(),
    )
    defaults.update(kwargs)
    return QposKnotsSpace(
        reference=(
            reference
            if reference is not None
            else _reference(control_period=defaults["control_period"])
        ),
        **defaults,
    )


def test_latent_shape_zero_and_decode_shape_validation() -> None:
    space = _space()
    assert space.latent_shape == (K, D)
    zero = space.zero()
    assert zero.shape == (K, D)
    assert zero.dtype == np.float64
    assert not zero.any()
    with pytest.raises(ValueError, match="shape"):
        space.decode(np.zeros((K, D)))
    with pytest.raises(ValueError, match="shape"):
        space.decode(np.zeros((2, K + 1, D)))


def test_null_latent_reproduces_reference_across_batch() -> None:
    reference = _reference()
    space = _space(reference)
    decoded = space.decode(np.tile(space.zero(), (3, 1, 1)))
    assert decoded.query_offsets == reference.query_offsets
    for name in ARRAY_FIELDS:
        out, expected = getattr(decoded, name), getattr(reference, name)
        assert out.dtype == np.float64
        assert out.shape[0] == 3
        for lane in range(3):
            np.testing.assert_array_equal(out[lane], expected[0])


def test_reference_is_defensively_copied() -> None:
    reference = _reference()
    space = _space(reference)
    expected = space.decode(space.zero()[None])
    for name in ARRAY_FIELDS:
        getattr(reference, name)[:] = 99.0
    decoded = space.decode(space.zero()[None])
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(decoded, name), getattr(expected, name))


def test_constant_knot_deltas_yield_constant_dense_deltas() -> None:
    reference = _reference()
    space = _space(reference)
    decoded = space.decode(np.full((1, K, D), 0.5))
    np.testing.assert_allclose(
        decoded.joint_position - reference.joint_position, 0.5 * 0.4, atol=1e-12
    )
    np.testing.assert_allclose(
        decoded.body_position_world[:, :, PELVIS]
        - reference.body_position_world[:, :, PELVIS],
        np.broadcast_to(0.5 * np.array([0.1, 0.1, 0.2]), (1, H, 3)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decoded.joint_velocity, reference.joint_velocity, atol=1e-9
    )
    np.testing.assert_allclose(
        decoded.body_linear_velocity_world[:, :, PELVIS],
        reference.body_linear_velocity_world[:, :, PELVIS],
        atol=1e-9,
    )


def test_interpolation_at_knot_steps_reproduces_knot_values() -> None:
    reference = _reference()
    space = _space(reference)
    latents = np.random.default_rng(7).uniform(-1.0, 1.0, (2, K, D))
    decoded = space.decode(latents)
    scale = SCALES.scale_vector(len(JOINTS))
    for knot, step_index in enumerate(KNOT_STEPS):
        delta = latents[:, knot] * scale
        np.testing.assert_array_equal(
            decoded.joint_position[:, step_index],
            reference.joint_position[0, step_index] + delta[:, 6:],
        )
        np.testing.assert_array_equal(
            decoded.body_position_world[:, step_index, PELVIS],
            reference.body_position_world[0, step_index, PELVIS] + delta[:, :3],
        )


def test_linear_interpolation_with_fractional_knot_times() -> None:
    horizon, knots = 40, 5
    reference = _reference(horizon=horizon)
    space = _space(reference, num_knots=knots)
    latents = np.random.default_rng(47).uniform(-1.0, 1.0, (2, knots, D))
    positions = np.linspace(0.0, horizon - 1.0, knots)
    physical = latents * SCALES.scale_vector(len(JOINTS))
    dense = np.stack(
        [
            np.stack(
                [np.interp(np.arange(horizon), positions, lane[:, d]) for d in range(D)],
                axis=-1,
            )
            for lane in physical
        ]
    )
    decoded = space.decode(latents)
    np.testing.assert_allclose(
        decoded.joint_position, reference.joint_position + dense[..., 6:],
        rtol=0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        decoded.body_position_world[:, :, PELVIS],
        reference.body_position_world[:, :, PELVIS] + dense[..., :3],
        rtol=0.0, atol=1e-15,
    )


def test_out_of_range_latents_are_clipped() -> None:
    space = _space()
    big = np.full((1, K, D), 5.0)
    big[0, 0, 0] = -7.0
    clipped = space.decode(big)
    unit = space.decode(np.clip(big, -1.0, 1.0))
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(clipped, name), getattr(unit, name))


def test_scale_vector_layout_and_per_group_application() -> None:
    np.testing.assert_array_equal(
        SCALES.scale_vector(2), [0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4]
    )
    reference = _reference()
    decoded = _space(reference).decode(np.ones((1, K, D)))
    np.testing.assert_array_equal(
        decoded.body_position_world[0, 0, PELVIS],
        reference.body_position_world[0, 0, PELVIS] + [0.1, 0.1, 0.2],
    )
    np.testing.assert_array_equal(
        decoded.joint_position[0, 0], reference.joint_position[0, 0] + 0.4
    )


def test_joint_limits_clamp_decoded_targets() -> None:
    reference = _reference()
    lower, upper = np.full(3, -0.6), np.full(3, 0.6)
    space = _space(reference, joint_lower=lower, joint_upper=upper)
    decoded = space.decode(np.ones((1, K, D)))
    assert (decoded.joint_position <= 0.6).all()
    assert (decoded.joint_position == 0.6).any()
    for step_index in KNOT_STEPS:
        np.testing.assert_array_equal(
            decoded.joint_position[0, step_index],
            np.clip(reference.joint_position[0, step_index] + 0.4, lower, upper),
        )


def test_joint_velocity_uses_post_clamp_trajectory() -> None:
    # Non-constant pre-clamp deltas that partially saturate the upper limit:
    # post-clamp finite differences differ from the pre-clamp delta gradient.
    horizon = 3
    reference = replace(
        _reference(horizon=horizon, seed=2, control_period=0.1),
        joint_position=np.full((1, horizon, len(JOINTS)), 0.4),
        joint_velocity=np.zeros((1, horizon, len(JOINTS))),
    )
    space = _space(
        reference,
        num_knots=3,
        control_period=0.1,
        scales=KnotScales(1.0, 1.0, 1.0, 1.0),
        joint_lower=np.full(len(JOINTS), -1.0),
        joint_upper=np.full(len(JOINTS), 0.5),
    )
    latents = np.zeros((1, 3, D))
    latents[0, :, 6] = [0.05, 0.2, 0.3]
    decoded = space.decode(latents)
    np.testing.assert_allclose(
        decoded.joint_position[0, :, 0], [0.45, 0.5, 0.5], atol=1e-12
    )
    # Effective post-clamp deltas from the reference: [0.05, 0.1, 0.1].
    np.testing.assert_allclose(
        decoded.joint_velocity[0, :, 0],
        [(0.1 - 0.05) / 0.1, (0.1 - 0.05) / 0.2, (0.1 - 0.1) / 0.1],
        atol=1e-12,
    )
    # Pre-clamp gradient of [0.05, 0.2, 0.3] would be [1.5, 1.25, 1.0].
    assert not np.allclose(decoded.joint_velocity[0, :, 0], [1.5, 1.25, 1.0])


def test_pelvis_orientation_composes_rotvec_with_reference() -> None:
    reference = _reference()
    space = _space(reference)
    latents = np.random.default_rng(3).uniform(-1.0, 1.0, (2, K, D))
    decoded = space.decode(latents)
    positions = np.linspace(0.0, H - 1.0, K)
    knot_rotvec = latents[:, :, 3:6] * 0.3
    for lane in range(2):
        for step_index in range(H):
            rotvec = np.array(
                [
                    np.interp(step_index, positions, knot_rotvec[lane, :, axis])
                    for axis in range(3)
                ]
            )
            angle = np.linalg.norm(rotvec)
            w1, (x1, y1, z1) = np.cos(angle / 2), rotvec * np.sin(angle / 2) / angle
            w2, x2, y2, z2 = reference.body_orientation_world[0, step_index, PELVIS]
            quat = np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ]
            )
            quat /= np.linalg.norm(quat)
            if quat[0] < 0:
                quat = -quat
            np.testing.assert_allclose(
                decoded.body_orientation_world[lane, step_index, PELVIS], quat, atol=1e-12
            )


def test_velocity_matches_hand_computed_finite_difference() -> None:
    # H=3 with K=3 puts one knot on every step_index, so the dense deltas are the
    # knot values themselves; np.gradient over dt=0.1 gives forward/backward
    # differences at the edges and a central difference in the middle.
    reference = _reference(horizon=3, seed=1, control_period=0.1)
    space = _space(
        reference, num_knots=3, control_period=0.1, scales=KnotScales(1.0, 1.0, 1.0, 1.0)
    )
    latents = np.zeros((1, 3, D))
    latents[0, :, 0] = [0.0, 0.4, -0.2]
    latents[0, :, 6] = [0.0, 0.4, -0.2]
    decoded = space.decode(latents)
    expected = [(0.4 - 0.0) / 0.1, (-0.2 - 0.0) / 0.2, (-0.2 - 0.4) / 0.1]
    np.testing.assert_allclose(
        decoded.joint_velocity[0, :, 0] - reference.joint_velocity[0, :, 0],
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decoded.body_linear_velocity_world[0, :, PELVIS, 0]
        - reference.body_linear_velocity_world[0, :, PELVIS, 0],
        expected,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        decoded.joint_velocity[0, :, 1:], reference.joint_velocity[0, :, 1:]
    )


def test_projector_recomputes_joint_induced_body_motion_and_velocities() -> None:
    reference = _reference()
    projector = _SyntheticProjector()
    space = _space(reference, projector=projector)
    latents = np.zeros((1, K, D))
    latents[0, :, 6:] = np.asarray(
        [[-0.4, 0.1, 0.2], [0.3, -0.2, 0.1], [0.1, 0.4, -0.3]]
    )
    decoded = space.decode(latents)
    torso = BODIES.index("torso_link")
    assert projector.calls == 1
    assert not np.array_equal(
        decoded.body_position_world[:, :, torso],
        reference.body_position_world[:, :, torso],
    )
    assert not np.array_equal(
        decoded.body_orientation_world[:, :, torso],
        reference.body_orientation_world[:, :, torso],
    )
    assert not np.array_equal(
        decoded.body_linear_velocity_world[:, :, torso],
        reference.body_linear_velocity_world[:, :, torso],
    )
    assert not np.array_equal(
        decoded.body_angular_velocity_world[:, :, torso],
        reference.body_angular_velocity_world[:, :, torso],
    )


def test_decode_rejects_invalid_projector_output() -> None:
    class BadProjector(_SyntheticProjector):
        def project(self, **kwargs):
            motion_command = super().project(**kwargs)
            return replace(
                motion_command,
                query_offsets=tuple(offset + 1 for offset in motion_command.query_offsets),
            )

    with pytest.raises(ValueError, match="incompatible metadata"):
        _space(projector=BadProjector()).decode(np.zeros((1, K, D)))


def test_decode_is_deterministic() -> None:
    space = _space()
    latents = np.random.default_rng(5).uniform(-1.0, 1.0, (4, K, D))
    first, second = space.decode(latents), space.decode(latents)
    for name in ARRAY_FIELDS:
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes()


def test_shift_advances_deformation_and_returns_to_reference_at_the_tail() -> None:
    space = _space()
    latent = np.zeros(space.latent_shape)
    latent[:, 0] = [0.0, 0.5, 1.0]

    shifted = space.shift(latent, steps=2)

    positions = np.linspace(0.0, H - 1.0, K)
    expected = np.interp(positions + 2, positions, latent[:, 0], right=0.0)
    np.testing.assert_allclose(shifted[:, 0], expected)
    np.testing.assert_array_equal(shifted[:, 1:], 0.0)
    assert not np.shares_memory(shifted, latent)
    np.testing.assert_array_equal(space.shift(latent, H), 0.0)


def test_projection_is_canonical_for_decode_and_warm_start() -> None:
    space = _space()
    latent = np.random.default_rng(9).uniform(-3.0, 3.0, space.latent_shape)

    projected = space.project(latent)

    assert np.max(np.abs(projected)) <= 1.0
    assert not np.shares_memory(projected, latent)
    raw_motion_command = space.decode(latent[None])
    projected_motion_command = space.decode(projected[None])
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(
            getattr(raw_motion_command, name), getattr(projected_motion_command, name)
        )
    np.testing.assert_array_equal(
        space.shift(latent, steps=2), space.shift(projected, steps=2)
    )


def test_shift_rejects_invalid_latent_or_steps() -> None:
    space = _space()
    with pytest.raises(ValueError, match="latent"):
        space.shift(np.zeros((1, 1)), 1)
    with pytest.raises(ValueError, match="steps"):
        space.shift(space.zero(), -1)


def test_construction_rejects_invalid_inputs() -> None:
    reference = _reference()
    for kwargs in (
        dict(reference=replace(reference, query_offsets=tuple(range(0, 2 * H, 2)))),
        dict(reference=_reference(batch=2)),
        dict(reference=_reference(horizon=1)),
        dict(num_knots=1),
        dict(control_period=0.0),
        dict(control_period=np.inf),
        dict(pelvis_body="missing_body"),
        dict(joint_lower=np.zeros(3)),
        dict(joint_upper=np.zeros(3)),
        dict(joint_lower=np.zeros(3), joint_upper=np.zeros(3)),
        dict(joint_lower=np.zeros(2), joint_upper=np.ones(2)),
        dict(joint_lower=np.full(3, -0.1), joint_upper=np.full(3, 0.1)),
    ):
        with pytest.raises(ValueError):
            _space(**{"reference": reference, **kwargs})


def test_knot_scales_reject_negative_or_nonfinite_values() -> None:
    good = dict(
        root_xy_m=0.1, root_z_m=0.1, root_rotation_rad=0.1, joint_position_rad=0.1
    )
    for kwargs in (
        dict(root_xy_m=-0.1),
        dict(root_z_m=np.nan),
        dict(root_rotation_rad=np.inf),
        dict(joint_position_rad=-1.0),
    ):
        with pytest.raises(ValueError):
            KnotScales(**{**good, **kwargs})
