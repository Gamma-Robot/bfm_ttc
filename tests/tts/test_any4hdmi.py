from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bfm_adapters.contracts import MotionCommand
from command_optimization.motion_tracking.any4hdmi import (
    load_reference_motion,
    resolve_any4hdmi_motion,
)
from command_optimization.motion_tracking.export import OutputConfig, export_motion

ROOT_NAMES = (
    "root_tx",
    "root_ty",
    "root_tz",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
)
JOINTS = ("hip", "knee")
SOURCE_JOINTS = ("knee", "hip")
BODIES = ("pelvis", "torso")
ARRAY_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)


class _Projector:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.nonfinite = nonfinite

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
        self.calls.append(
            {
                "root_position_world": root_position_world.copy(),
                "root_orientation_world": root_orientation_world.copy(),
                "joint_position": joint_position.copy(),
                "control_period": control_period,
            }
        )
        batch, horizon = joint_position.shape[:2]

        def derivative(value: np.ndarray) -> np.ndarray:
            if horizon == 1:
                return np.zeros_like(value)
            return np.gradient(value, control_period, axis=1)

        body_position = np.repeat(
            root_position_world[:, :, None, :], len(body_names), axis=2
        )
        body_position[:, :, 1, 2] += 0.5
        body_orientation = np.repeat(
            root_orientation_world[:, :, None, :], len(body_names), axis=2
        )
        if self.nonfinite:
            body_position[0, 0, 0, 0] = np.nan
        return MotionCommand(
            query_offsets=query_offsets,
            joint_names=joint_names,
            body_names=body_names,
            joint_position=joint_position,
            joint_velocity=derivative(joint_position),
            body_position_world=body_position,
            body_orientation_world=body_orientation,
            body_linear_velocity_world=derivative(body_position),
            body_angular_velocity_world=np.zeros((batch, horizon, len(body_names), 3)),
        )


def _qpos(frames: int = 3) -> np.ndarray:
    qpos = np.zeros((frames, 7 + len(JOINTS)), dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.arange(frames) + 10.0  # source knee order
    qpos[:, 8] = np.arange(frames) + 20.0  # source hip order
    return qpos


def _write_dataset(
    root: Path,
    qpos: np.ndarray,
    *,
    timestep: float = 0.1,
    manifest_updates: dict[str, object] | None = None,
    extra_motion_field: bool = False,
    nested: bool = False,
) -> Path:
    manifest: dict[str, object] = {
        "format_version": 2,
        "dataset_name": "synthetic-g1",
        "mjcf": "g1.xml",
        "motions_subdir": "motions",
        "timestep": timestep,
        "qpos_dim": qpos.shape[1],
        "qpos_names": [*ROOT_NAMES, *SOURCE_JOINTS],
        "num_motions": 1,
    }
    manifest.update(manifest_updates or {})
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    motion_dir = root / "motions"
    motion_dir.mkdir()
    motion_path = motion_dir / "demo.npz"
    if nested:
        motion_path = motion_dir / "demo" / "motion.npz"
        motion_path.parent.mkdir()
    if extra_motion_field:
        np.savez(motion_path, qpos=qpos, extra=np.zeros(1))
    else:
        np.savez(motion_path, qpos=qpos)
    return root


def _load(root: Path, projector: _Projector | None = None, **kwargs) -> MotionCommand:
    return load_reference_motion(
        root,
        "demo",
        joint_names=JOINTS,
        body_names=BODIES,
        control_period=kwargs.pop("control_period", 0.1),
        projector=projector or _Projector(),
        **kwargs,
    )


def test_loads_canonical_batch_one_target_and_freezes_arrays(tmp_path: Path) -> None:
    qpos = _qpos()
    projector = _Projector()
    motion_command = _load(_write_dataset(tmp_path / "dataset", qpos), projector)

    assert motion_command.query_offsets == (0, 1, 2)
    assert motion_command.joint_names == JOINTS
    assert motion_command.body_names == BODIES
    assert len(projector.calls) == 1
    call = projector.calls[0]
    np.testing.assert_array_equal(call["joint_position"][0], qpos[:, [8, 7]])
    np.testing.assert_array_equal(call["root_position_world"][0], qpos[:, :3])
    assert call["control_period"] == 0.1
    for field in ARRAY_FIELDS:
        value = getattr(motion_command, field)
        assert value.dtype == np.float64
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 1.0


@pytest.mark.parametrize("trajectory", ("execution", "command"))
def test_exports_npz_and_any4hdmi_round_trip(tmp_path: Path, trajectory: str) -> None:
    qpos = _qpos()
    before = qpos.copy()
    export_motion(
        tmp_path, trajectory=trajectory, qpos=qpos,
        valid=np.ones(len(qpos), dtype=bool), joint_names=SOURCE_JOINTS,
        control_period=0.1, mjcf="g1.xml", formats=("npz", "any4hdmi"),
    )
    with np.load(tmp_path / f"{trajectory}.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["qpos"], qpos)
        np.testing.assert_allclose(payload["timeline"], [0.0, 0.1, 0.2])
        assert tuple(payload["qpos_names"]) == (*ROOT_NAMES, *SOURCE_JOINTS)
        assert payload["trajectory"].item() == trajectory
    projector = _Projector()
    target = load_reference_motion(
        tmp_path / f"{trajectory}.any4hdmi", trajectory,
        joint_names=JOINTS, body_names=BODIES, control_period=0.1,
        projector=projector,
    )
    assert target.query_offsets == (0, 1, 2)
    np.testing.assert_array_equal(target.joint_position[0], qpos[:, [8, 7]])
    np.testing.assert_array_equal(projector.calls[0]["root_position_world"][0], qpos[:, :3])
    np.testing.assert_array_equal(qpos, before)


def test_export_rejects_incomplete_motion_and_invalid_options(tmp_path: Path) -> None:
    assert OutputConfig().formats == ()
    for values in ({"formats": "npz"}, {"formats": ["unknown"]},
                   {"trajectories": ["execution", "execution"]}):
        with pytest.raises(ValueError):
            OutputConfig(**values)
    with pytest.raises(ValueError, match="incomplete"):
        export_motion(
            tmp_path, trajectory="execution", qpos=_qpos(),
            valid=np.array([True, True, False]), joint_names=SOURCE_JOINTS,
            control_period=0.1, mjcf="g1.xml", formats=("any4hdmi",),
        )
    assert not list(tmp_path.iterdir())


def test_resamples_translation_joints_and_root_rotation_at_control_steps(
    tmp_path: Path,
) -> None:
    qpos = _qpos(frames=2)
    qpos[:, 0] = [0.0, 2.0]
    qpos[1, 3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    projector = _Projector()
    motion_command = _load(
        _write_dataset(tmp_path / "dataset", qpos, timestep=1.0),
        projector,
        control_period=0.5,
    )

    assert motion_command.query_offsets == (0, 1, 2)
    call = projector.calls[0]
    np.testing.assert_allclose(
        call["root_position_world"][0, :, 0], [0.0, 1.0, 2.0]
    )
    np.testing.assert_allclose(call["joint_position"][0, :, 0], [20.0, 20.5, 21.0])
    half_angle = np.pi / 8.0
    np.testing.assert_allclose(
        call["root_orientation_world"][0, 1],
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
        atol=1e-7,
    )


def test_duration_selects_a_prefix_before_projection(tmp_path: Path) -> None:
    projector = _Projector()
    motion_command = _load(
        _write_dataset(tmp_path / "dataset", _qpos(frames=5), timestep=0.1),
        projector,
        control_period=0.05,
        clip_duration_seconds=0.2,
    )

    assert motion_command.query_offsets == tuple(range(5))
    assert projector.calls[0]["joint_position"].shape == (1, 5, 2)


def test_explicit_duration_must_align_with_control_grid(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos(frames=5), timestep=0.1)

    with pytest.raises(ValueError, match="control_period grid"):
        _load(root, control_period=0.1, clip_duration_seconds=0.25)


def test_resolves_flat_and_nested_motion_layouts(tmp_path: Path) -> None:
    flat = _write_dataset(tmp_path / "flat", _qpos())
    nested = _write_dataset(tmp_path / "nested", _qpos(), nested=True)

    assert resolve_any4hdmi_motion(flat, "demo") == (
        flat / "motions" / "demo.npz"
    ).resolve()
    assert resolve_any4hdmi_motion(nested, "demo") == (
        nested / "motions" / "demo" / "motion.npz"
    ).resolve()
    assert _load(nested).query_offsets == (0, 1, 2)


@pytest.mark.parametrize("motion_id", (
    "ACCAD/Female1Walking_c3d/B22 - side step left_poses/motion_seg1",
    "BioMotionLab_NTroje/rub067/0019_lifting_heavy1_poses/motion_seg1",
    "ACCAD/Female1Running_c3d/C10 -  run backwards stop run forward_poses/motion_seg1",
    "ACCAD/Female1Running_c3d/C11 -  run turn left (90)_poses/motion_seg1",
))
def test_loads_exact_full_amass_motion_id(tmp_path: Path, motion_id: str) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos())
    motion = root / "motions" / motion_id / "motion.npz"
    motion.parent.mkdir(parents=True)
    (root / "motions" / "demo.npz").rename(motion)

    assert resolve_any4hdmi_motion(root, motion_id) == motion
    result = load_reference_motion(
        root, motion_id, joint_names=JOINTS, body_names=BODIES,
        control_period=0.1, projector=_Projector(),
    )
    np.testing.assert_array_equal(result.joint_position[0], _qpos()[:, [8, 7]])


@pytest.mark.parametrize("basename", ("motion_seg1", "0019_lifting_heavy1_poses"))
def test_full_ids_distinguish_repeated_basenames(tmp_path: Path, basename: str) -> None:
    root = tmp_path / "dataset"
    flat_id = f"subject one/{basename}"
    nested_id = f"subject two/{basename}"
    flat = root / "motions" / f"{flat_id}.npz"
    nested = root / "motions" / nested_id / "motion.npz"
    for path in (flat, nested):
        path.parent.mkdir(parents=True)
        np.savez(path, qpos=_qpos())

    assert resolve_any4hdmi_motion(root, flat_id) == flat
    assert resolve_any4hdmi_motion(root, nested_id) == nested
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_any4hdmi_motion(root, basename)


@pytest.mark.parametrize("motion_id", ("demo", "subject one/clip (1)"))
def test_exact_id_rejects_ambiguous_layouts(tmp_path: Path, motion_id: str) -> None:
    root = tmp_path / "dataset"
    flat = root / "motions" / f"{motion_id}.npz"
    nested = root / "motions" / motion_id / "motion.npz"
    nested.parent.mkdir(parents=True)
    np.savez(flat, qpos=_qpos())
    np.savez(nested, qpos=_qpos())

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_any4hdmi_motion(root, motion_id)


@pytest.mark.parametrize("motion_id", (
    "", "/demo", "../demo", "subject/../demo", "./demo", "subject/./demo",
    "subject//demo", "subject/demo/", "subject/..", "subject/.",
    "C:/demo", "subject\\demo", "demo\x00", None,
))
def test_motion_resolution_rejects_unsafe_components(
    tmp_path: Path, motion_id: object,
) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos())
    with pytest.raises(ValueError, match="invalid motion_id"):
        resolve_any4hdmi_motion(root, motion_id)


def test_motion_resolution_rejects_missing_ambiguous_and_invalid_ids(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos())
    nested = root / "motions" / "demo" / "motion.npz"
    nested.parent.mkdir()
    np.savez(nested, qpos=_qpos())

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_any4hdmi_motion(root, "demo")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_any4hdmi_motion(root, "missing")
    with pytest.raises(ValueError, match="invalid motion_id"):
        resolve_any4hdmi_motion(root, "../demo")


@pytest.mark.parametrize("motion_id", ("demo", "subject one/clip (1)"))
def test_motion_resolution_rejects_symlink_escape(
    tmp_path: Path, motion_id: str,
) -> None:
    root = tmp_path / "dataset"
    (root / "motions").mkdir(parents=True)
    outside = tmp_path / "outside.npz"
    np.savez(outside, qpos=_qpos())
    candidate = root / "motions" / f"{motion_id}.npz"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes dataset root"):
        resolve_any4hdmi_motion(root, motion_id)


def test_rejects_invalid_or_excessive_duration(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos(), timestep=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        _load(root, clip_duration_seconds=0.0)
    with pytest.raises(ValueError, match="exceeds"):
        _load(root, clip_duration_seconds=0.3)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"format_version": 1}, "format_version"),
        ({"motions_subdir": "clips"}, "motions_subdir"),
        ({"timestep": 0.0}, "timestep"),
        ({"qpos_dim": 99}, "qpos_dim"),
        ({"qpos_names": ["bad", *ROOT_NAMES[1:], *SOURCE_JOINTS]}, "free-root"),
    ],
)
def test_rejects_invalid_manifest(
    tmp_path: Path, updates: dict[str, object], match: str
) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos(), manifest_updates=updates)
    with pytest.raises(ValueError, match=match):
        _load(root)


def test_rejects_motion_schema_and_values(tmp_path: Path) -> None:
    extra = _write_dataset(tmp_path / "extra", _qpos(), extra_motion_field=True)
    with pytest.raises(ValueError, match="only 'qpos'"):
        _load(extra)

    nonfinite_qpos = _qpos()
    nonfinite_qpos[1, 0] = np.nan
    nonfinite = _write_dataset(tmp_path / "nonfinite", nonfinite_qpos)
    with pytest.raises(ValueError, match="finite"):
        _load(nonfinite)

    bad_quaternion = _qpos()
    bad_quaternion[:, 3] = 2.0
    bad_quat = _write_dataset(tmp_path / "bad_quat", bad_quaternion)
    with pytest.raises(ValueError, match="normalized"):
        _load(bad_quat)


def test_rejects_joint_mismatch_unsafe_id_and_nonfinite_projection(
    tmp_path: Path,
) -> None:
    root = _write_dataset(tmp_path / "dataset", _qpos())
    with pytest.raises(ValueError, match="canonical joint_names"):
        load_reference_motion(
            root,
            "demo",
            joint_names=("hip", "ankle"),
            body_names=BODIES,
            control_period=0.1,
            projector=_Projector(),
        )
    with pytest.raises(ValueError, match="motion_id"):
        load_reference_motion(
            root,
            "../demo",
            joint_names=JOINTS,
            body_names=BODIES,
            control_period=0.1,
            projector=_Projector(),
        )
    with pytest.raises(ValueError, match="non-finite body_position_world"):
        _load(root, _Projector(nonfinite=True))
