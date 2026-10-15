from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper

import command_optimization.run as run_module
from bfm_adapters import MotionCommand, instance_config_from_mapping
from command_optimization.artifacts import ArtifactEntry
from command_optimization.config import ConfigError, dataclass_from_dict, resolve_config
from command_optimization.motion_tracking.search_space import KnotScales
from command_optimization.records import PlanningDiagnostics
from command_optimization.run import (
    ExperimentConfig,
    PlanningConfig,
    ReferenceConfig,
    build_optimizer,
    load_experiment,
)
from command_optimization.simulation.randomization import (
    RandomizationRanges,
    RobustnessConfig,
    sample_scenarios,
)
from command_optimization.visualization import (
    ComparisonView,
    RenderConfig,
    RenderBindings,
    RenderProvenance,
)


REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
EXPERIMENT = REPO / "tests" / "fixtures" / "experiments" / "heft_smoke.yaml"
RUNTIME_PROFILE_EXPERIMENT = REPO / "tests" / "fixtures" / "experiments" / "heft_runtime_profile.yaml"


def _motion_command(values: list[float]) -> MotionCommand:
    joint = np.asarray(values, dtype=np.float64)[None, :, None]
    num_steps = joint.shape[1]
    body_position = np.zeros((1, num_steps, 1, 3))
    body_orientation = np.zeros((1, num_steps, 1, 4))
    body_orientation[..., 0] = 1.0
    vectors = np.zeros_like(body_position)
    return MotionCommand(
        query_offsets=tuple(range(num_steps)),
        joint_names=("joint",),
        body_names=("pelvis",),
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body_position,
        body_orientation_world=body_orientation,
        body_linear_velocity_world=vectors,
        body_angular_velocity_world=vectors.copy(),
    )


def test_heft_smoke_resolves_refs_and_binds_typed_config() -> None:
    resolved, config = load_experiment(EXPERIMENT)

    assert not _contains_ref(resolved)
    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.reference, ReferenceConfig)
    assert isinstance(config.planning, PlanningConfig)
    assert isinstance(config.planning.scales, KnotScales)
    assert isinstance(config.visualization, RenderConfig)
    assert config.robustness == RobustnessConfig()
    assert config.bfm["family"] == "heft"
    assert config.robot["instance"] == "g1_mjwarp"
    assert config.optimizer["algorithm"] == "weighted_elites"
    assert config.reference.motion_id == "0019"
    assert config.schema_version == 3


@pytest.mark.parametrize("schema_version", (None, 1, 2, True, 3.0))
def test_experiment_config_requires_exact_schema_v3(schema_version: object) -> None:
    resolved = resolve_config(EXPERIMENT, configs_root=CONFIGS)
    if schema_version is None:
        resolved.pop("schema_version")
    else:
        resolved["schema_version"] = schema_version

    with pytest.raises((ConfigError, ValueError), match="schema_version"):
        dataclass_from_dict(ExperimentConfig, resolved)


@pytest.mark.parametrize("family", ("heft", "sonic", "twist2"))
def test_world_collects_task_and_policy_bodies(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = load_experiment(REPO / "tests" / "fixtures" / "experiments" / f"{family}_smoke.yaml")
    bfm = instance_config_from_mapping(config.bfm)
    task_bodies = tuple(config.motion_tracking["body_names"])
    captured: dict[str, object] = {}

    def world(model_path: Path, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(run_module, "MjwarpWorld", world)
    run_module._build_world(
        Path("unused.xml"), config.robot, bfm,
        body_names=task_bodies, batch=1, device_index=0,
    )
    assert captured["body_names"] == tuple(
        dict.fromkeys((*task_bodies, *bfm.motion_command_body_names))
    )
    if family in ("heft", "sonic"):
        assert task_bodies == bfm.motion_command_body_names
    else:
        assert task_bodies != bfm.motion_command_body_names


def test_experiment_binds_nested_robustness_config() -> None:
    resolved = resolve_config(EXPERIMENT, configs_root=CONFIGS)
    resolved["robustness"] = {
        "enabled": True,
        "num_scenarios": 4,
        "scenario_seed": 19,
        "risk": "worst",
        "ranges": {"contact_friction_scale": [0.8, 1.2]},
    }

    config = dataclass_from_dict(ExperimentConfig, resolved)

    assert config.robustness.num_scenarios == 4
    assert config.robustness.ranges.contact_friction_scale == (0.8, 1.2)


def test_repository_config_surface_is_curated_and_resolvable() -> None:
    from command_optimization.batch import load_experiment_batch

    assert {path.name for path in CONFIGS.iterdir() if path.is_dir()} == {
        "experiments",
        "motion_tracking",
        "optimizers",
        "robots",
        "bfms",
    }
    tracked = tuple(CONFIGS.rglob("*.yaml"))
    assert all("stage" not in path.stem.lower() for path in tracked)

    experiments = CONFIGS / "experiments"
    for path in experiments.glob("*.yaml"):
        if path.name.endswith(".base.yaml"):
            resolve_config(path, configs_root=CONFIGS)
        elif path.name.endswith(".batch.yaml"):
            load_experiment_batch(path)
        else:
            load_experiment(path)


def _artifact_entry(
    artifact_id: str,
    *,
    kind: str,
    digest: str,
    tree_digest: str | None = None,
) -> ArtifactEntry:
    return ArtifactEntry(
        id=artifact_id,
        kind=kind,
        sha256=digest,
        bytes=1,
        tree_sha256=tree_digest,
    )


def test_target_assets_accept_file_pair_and_preserve_motion_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    (root / "motions").mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text("{}")
    motion = root / "motions" / "demo.npz"
    motion.write_bytes(b"motion")
    entries = {
        "manifest": _artifact_entry("manifest", kind="file", digest="1" * 64),
        "motion": _artifact_entry("motion", kind="file", digest="2" * 64),
    }
    paths = {"manifest": manifest, "motion": motion}
    monkeypatch.setattr(
        run_module,
        "ensure_artifact",
        lambda artifact_id, **_kwargs: paths[artifact_id],
    )

    assets = run_module._resolve_reference_assets(
        ReferenceConfig("manifest", "motion", "demo", 0.2), entries, tmp_path
    )

    assert assets.dataset_root == root.resolve()
    assert assets.reference_id == "motion"
    assert assets.reference_digest == "2" * 64


def test_local_target_resolves_relative_to_user_yaml_and_hashes_both_files(
    tmp_path: Path,
) -> None:
    resolved, _ = load_experiment(EXPERIMENT)
    resolved["reference"] = {"dataset_root": "data", "motion_id": "demo"}
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(resolved))
    root = tmp_path / "data"
    (root / "motions").mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text("{}")
    motion = root / "motions" / "demo.npz"
    motion.write_bytes(b"motion")
    _, config = load_experiment(config_path)
    assert config.reference.dataset_root == str(root)
    assert config.reference.clip_duration_seconds is None
    first = run_module._resolve_reference_assets(config.reference, {}, REPO)
    manifest.write_text('{"timestep": 0.02}')
    second = run_module._resolve_reference_assets(config.reference, {}, REPO)
    motion.write_bytes(b"changed")
    third = run_module._resolve_reference_assets(config.reference, {}, REPO)
    assert first.dataset_root == root
    assert first.reference_id == "local:demo"
    assert len({first.reference_digest, second.reference_digest, third.reference_digest}) == 3
    with pytest.raises(ValueError, match="not both"):
        dataclasses.replace(config.reference, motion_artifact="locked")
    assert run_module.run_experiment(config_path, dry_run=True) is None


@pytest.mark.gpu
@pytest.mark.conformance
@pytest.mark.parametrize("family", ("heft", "sonic", "twist2"))
def test_local_motion_pipeline_with_cem_and_selected_exports(
    tmp_path: Path, family: str,
) -> None:
    manifest_path = REPO / "data" / "manifest.json"
    source_path = REPO / "data" / "motions" / "0019.npz"
    manifest = json.loads(manifest_path.read_text())
    root = tmp_path / "data"
    (root / "motions").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with np.load(source_path, allow_pickle=False) as source:
        np.savez_compressed(root / "motions" / "local.npz", qpos=source["qpos"][:4])
    resolved, _ = load_experiment(CONFIGS / "experiments" / "single_motion.yaml")
    _, short = load_experiment(REPO / "tests" / "fixtures" / "experiments" / f"{family}_smoke.yaml")
    resolved["bfm"] = short.bfm
    resolved["reference"] = {"dataset_root": "data", "motion_id": "local"}
    resolved["visualization"].update(enabled=family == "twist2", width=160, height=128)
    task = tmp_path / "local.yaml"
    task.write_text(yaml.safe_dump(resolved))
    output = tmp_path / "output"
    result = run_module.run_experiment(
        task, device_index=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")), output_dir=output,
    )
    assert result is not None and result.pair.optimized.trace.success
    trace = result.pair.optimized.trace
    assert trace.num_steps == int(np.floor(3 * manifest["timestep"] / short.bfm["control_period"] + 1e-9))
    assert all(update.logical_rollouts == 80 * 4 for update in result.pair.optimized.planning)
    assert (output / run_module.TRACE_FILE).is_file()
    for trajectory in ("execution", "command"):
        with np.load(output / "exports" / f"{trajectory}.npz", allow_pickle=False) as data:
            assert data["qpos"].shape == (trace.num_steps + 1, 36)
            assert data["timeline"][0] == 0.0
            np.testing.assert_array_equal(data["timeline"][1:], trace.timeline)
            qpos = data["qpos"].copy()
        exported_root = output / "exports" / f"{trajectory}.any4hdmi"
        exported_manifest = json.loads((exported_root / "manifest.json").read_text())
        assert exported_manifest["mjcf"] == manifest["mjcf"]
        assert exported_manifest["source"]["trajectory"] == trajectory
        with np.load(exported_root / "motions" / f"{trajectory}.npz") as data:
            np.testing.assert_array_equal(data["qpos"], qpos.astype(np.float32))
    video = output / run_module.RENDER_FILE
    assert video.exists() == (family == "twist2")
    if video.exists():
        frame = run_module.subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True,
        ).stdout
        pixels = np.frombuffer(frame, dtype=np.uint8)
        assert pixels.size == 160 * 128 * 3 and pixels.std() > 1.0


def test_target_assets_reject_misnamed_manifest_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    (root / "motions").mkdir(parents=True)
    manifest = root / "metadata.json"
    manifest.write_text("{}")
    motion = root / "motions" / "demo.npz"
    motion.write_bytes(b"motion")
    entries = {
        "manifest": _artifact_entry("manifest", kind="file", digest="1" * 64),
        "motion": _artifact_entry("motion", kind="file", digest="2" * 64),
    }
    paths = {"manifest": manifest, "motion": motion}
    monkeypatch.setattr(
        run_module,
        "ensure_artifact",
        lambda artifact_id, **_kwargs: paths[artifact_id],
    )

    with pytest.raises(ValueError, match="must be manifest.json"):
        run_module._resolve_reference_assets(
            ReferenceConfig("manifest", "motion", "demo", 0.2), entries, tmp_path
        )


def test_target_assets_accept_one_shared_tree_and_bind_motion_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    motion = root / "motions" / "jump" / "motion.npz"
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"motion")
    entry = _artifact_entry(
        "dataset",
        kind="tar.gz",
        digest="3" * 64,
        tree_digest="4" * 64,
    )
    monkeypatch.setattr(run_module, "ensure_artifact", lambda *_args, **_kwargs: root)

    assets = run_module._resolve_reference_assets(
        ReferenceConfig("dataset", "dataset", "jump", 0.2),
        {"dataset": entry},
        tmp_path,
    )

    assert assets.dataset_root == root.resolve()
    assert assets.reference_id == "dataset:jump"
    assert assets.reference_digest == "4" * 64


def test_target_assets_reject_mixed_or_distinct_tree_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        motion = root / "motions" / "demo.npz"
        motion.parent.mkdir(parents=True)
        motion.write_bytes(b"motion")
    tree_entries = {
        "manifest": _artifact_entry(
            "manifest", kind="tar.gz", digest="1" * 64, tree_digest="2" * 64
        ),
        "motion": _artifact_entry(
            "motion", kind="tar.gz", digest="3" * 64, tree_digest="4" * 64
        ),
    }
    paths = {"manifest": first, "motion": second}
    monkeypatch.setattr(
        run_module,
        "ensure_artifact",
        lambda artifact_id, **_kwargs: paths[artifact_id],
    )
    config = ReferenceConfig("manifest", "motion", "demo", 0.2)

    with pytest.raises(ValueError, match="same dataset root"):
        run_module._resolve_reference_assets(config, tree_entries, tmp_path)

    mixed = {
        **tree_entries,
        "motion": _artifact_entry("motion", kind="file", digest="3" * 64),
    }
    with pytest.raises(ValueError, match="both be files or trees"):
        run_module._resolve_reference_assets(config, mixed, tmp_path)


def test_cross_config_rejects_duration_off_the_bfm_grid() -> None:
    _, config = load_experiment(EXPERIMENT)
    bfm = instance_config_from_mapping(config.bfm)
    off_grid = dataclasses.replace(
        config,
        reference=dataclasses.replace(config.reference, clip_duration_seconds=0.13),
    )

    with pytest.raises(ValueError, match="align with bfm control_period"):
        run_module._validate_cross_config(off_grid, bfm)


def test_cross_config_requires_one_complete_robust_scenario_group() -> None:
    _, config = load_experiment(EXPERIMENT)
    bfm = instance_config_from_mapping(config.bfm)
    robust = dataclasses.replace(
        config,
        robustness=RobustnessConfig(enabled=True, num_scenarios=6),
    )

    with pytest.raises(ValueError, match="complete scenario group"):
        run_module._validate_cross_config(robust, bfm)


def test_episode_context_edge_holds_reset_and_reference_boundaries() -> None:
    reference = _motion_command([10.0, 11.0, 12.0])

    history = run_module._initial_motion_command_history(reference, 2)
    right = run_module.reference_right_context(reference, 2, 3)

    assert history is not None and history.query_offsets == (-2, -1)
    np.testing.assert_array_equal(history.joint_position[0, :, 0], [10.0, 10.0])
    assert right is not None and right.query_offsets == (0, 1, 2)
    np.testing.assert_array_equal(right.joint_position[0, :, 0], [12.0, 12.0, 12.0])
    assert not right.joint_position.flags.writeable


def test_runtime_profile_resolves_one_canonical_single_wave_update() -> None:
    resolved, profile = load_experiment(RUNTIME_PROFILE_EXPERIMENT)
    assert not _contains_ref(resolved)
    assert profile.name == "heft_runtime_profile"
    assert profile.optimizer["num_candidates"] == 513
    assert profile.planning.max_worlds == 513
    assert profile.reference.clip_duration_seconds == 0.4
    assert float(profile.robot["physics_timestep"]) == 0.005

    # One executed prefix: future Control Steps equal replanning_interval_steps.
    control_period = float(profile.bfm["control_period"])
    future_steps = round(profile.reference.clip_duration_seconds / control_period)
    assert control_period == 0.02
    assert future_steps == profile.planning.replanning_interval_steps


def test_rollout_runtime_keeps_host_reference_as_new_family_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, object, int]] = []

    class HostPolicy:
        def __init__(self, path: Path, config: object, *, device_id: int) -> None:
            calls.append((path, config, device_id))

    adapter = object()
    bfm_config = object()
    policy_path = tmp_path / "policy.onnx"
    monkeypatch.setattr(run_module, "BatchedOrtCudaPolicy", HostPolicy)

    runtime = run_module._RolloutRuntimePool(
        world_factory=lambda _batch: None,
        adapter=adapter,
        bfm_config=bfm_config,
        policy_path=policy_path,
        device_index=3,
        objective=object(),
        control_period=0.02,
        motion_command_query_offsets=(0,),
    )
    try:
        assert runtime.implementation == "host-reference"
        assert runtime.reducer_factory is None
        assert calls == [(policy_path, bfm_config, 3)]
    finally:
        runtime.close()


def test_planning_json_serializes_the_measured_update_wall_time() -> None:
    diagnostics = PlanningDiagnostics(
        update_step=0,
        selected_source="best_observed",
        selected_candidate_id="best-sha",
        selected_cost=1.0,
        best_observed_candidate_id="best-sha",
        best_observed_cost=1.0,
        proposal_candidate_id="proposal-sha",
        proposal_cost=2.0,
        cost_quantiles=np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        candidate_deformation={"joint_rmse_rad": 0.2},
        candidate_evaluations=8,
        scenario_count=1,
        scenario_digest="sha256:scenario",
        logical_rollouts=8,
        physical_rollouts=16,
        penalized_candidate_evaluations=2,
        invalid_scenario_rollouts=3,
        update_wall_seconds=0.25,
    )

    payload = run_module._planning_json(diagnostics)

    assert payload["update_wall_seconds"] == 0.25
    assert payload["penalized_candidate_evaluations"] == 2
    assert payload["invalid_scenario_rollouts"] == 3
    assert payload["scenario_digest"] == "sha256:scenario"
    json.loads(json.dumps(payload))


def test_randomization_json_recomputes_and_binds_update_scenarios() -> None:
    ranges = RandomizationRanges(contact_friction_scale=(0.8, 1.2))
    config = RobustnessConfig(
        enabled=True,
        num_scenarios=2,
        scenario_seed=11,
        risk="worst",
        ranges=ranges,
    )
    expected = sample_scenarios(
        ranges, count=2, seed=11, num_joints=1
    ).digest
    diagnostics = PlanningDiagnostics(
        update_step=0,
        selected_source="best_observed",
        selected_candidate_id="best-sha",
        selected_cost=1.0,
        best_observed_candidate_id="best-sha",
        best_observed_cost=1.0,
        proposal_candidate_id="proposal-sha",
        proposal_cost=2.0,
        cost_quantiles=np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        candidate_deformation={"joint_rmse_rad": 0.2},
        candidate_evaluations=8,
        scenario_count=2,
        scenario_digest=expected,
        logical_rollouts=16,
        physical_rollouts=16,
        penalized_candidate_evaluations=0,
        invalid_scenario_rollouts=0,
        update_wall_seconds=0.25,
    )

    payload = run_module._randomization_json(
        config, (diagnostics,), {"joint_names": ["joint"]}
    )

    assert payload["updates"] == [
        {"update_step": 0, "seed": 11, "scenario_digest": expected}
    ]
    assert payload["ranges"]["contact_friction_scale"] == (0.8, 1.2)


@pytest.mark.parametrize("optimizer_path", sorted((CONFIGS / "optimizers").glob("*.yaml")))
def test_all_optimizer_yamls_use_the_generic_iteration_contract(
    optimizer_path: Path,
) -> None:
    data = yaml.safe_load(optimizer_path.read_text())
    assert not isinstance(data["num_iterations"], bool)
    assert isinstance(data["num_iterations"], int)
    assert data["num_iterations"] > 0

    try:
        optimizer, iterations = build_optimizer(data, num_knots=3, dim=2)
    except ModuleNotFoundError as exc:
        expected = f"optimizer {data['algorithm']!r} requires optional dependency"
        if expected not in str(exc):
            raise
        return

    assert iterations == data["num_iterations"]
    params = optimizer.init_params(seed=11)
    params, report = optimizer.optimize(
        params,
        lambda candidates: np.square(candidates - 0.25).sum(axis=(1, 2)),
        iterations=iterations,
    )
    assert len(report.iterations) == iterations
    assert params.iteration == iterations


def test_optimizer_config_rejects_unknown_and_search_space_owned_fields() -> None:
    valid = {
        "algorithm": "predictive_sampling",
        "num_candidates": 5,
        "num_iterations": 1,
        "noise_level": 0.2,
    }
    with pytest.raises(ValueError, match="unknown parameters.*typo"):
        build_optimizer({**valid, "typo": 1}, num_knots=3, dim=2)
    with pytest.raises(ValueError, match="search-space fields"):
        build_optimizer({**valid, "num_knots": 9}, num_knots=3, dim=2)


def test_hardware_identity_selects_the_logical_cuda_device(monkeypatch) -> None:
    seen: list[str] = []

    def fake_run(command, **_kwargs):
        seen.extend(command)
        return run_module.subprocess.CompletedProcess(
            command,
            0,
            stdout="2, GPU-example, NVIDIA GeForce RTX 4090, 8.9, 570.195.03\n",
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2")
    monkeypatch.setattr(run_module.subprocess, "run", fake_run)

    identity = run_module._hardware_identity(1)

    assert "--id=2" in seen
    assert identity == {
        "available": True,
        "logical_device_index": 1,
        "cuda_visible_devices": "5,2",
        "selector": "2",
        "physical_device_index": 2,
        "gpu_uuid": "GPU-example",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "compute_capability": "8.9",
        "driver_version": "570.195.03",
    }


def test_cli_keeps_behavior_in_yaml_and_render_has_no_overrides() -> None:
    parser = run_module._parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices

    assert _options(commands["run"]) == {
        "--device-index",
        "--output-dir",
        "--dry-run",
    }
    assert _options(commands["batch"]) == {
        "--device-index",
        "--output-dir",
        "--dry-run",
    }
    assert _options(commands["render"]) == set()
    assert _options(commands["prepare-bfm"]) == set()

    args = parser.parse_args(
        [
            "run",
            str(EXPERIMENT),
            "--device-index",
            "3",
            "--output-dir",
            "output",
            "--dry-run",
        ]
    )
    assert args.device_index == 3
    assert args.output_dir == Path("output")
    assert args.dry_run is True
    assert parser.parse_args(
        ["prepare-bfm", "heft-pmg-policy.batched"]
    ).artifact_id == "heft-pmg-policy.batched"


def test_default_output_is_grouped_by_experiment_and_run_identity(
    tmp_path: Path,
) -> None:
    output = run_module._default_output_dir(tmp_path, "experiment", "deadbeef1234")

    assert output.parent == tmp_path / "outputs" / "experiment"
    assert re.fullmatch(
        r"\d{8}T\d{6}\.\d{6}Z-deadbeef", output.name
    )


def _write_v2_render_inputs(
    experiment_dir: Path, resolved: dict[str, object]
) -> Path:
    (experiment_dir / "resolved_experiment.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True)
    )
    model = experiment_dir / "g1.xml"
    model.write_text("<mujoco/>")
    for name in (
        run_module.REPORT_FILE,
        run_module.PLANNING_FILE,
        run_module.RANDOMIZATION_FILE,
    ):
        (experiment_dir / name).write_text(json.dumps({"schema_version": 3}))
    provenance = {
        "schema_version": 3,
        "release_version": "0.1.0",
        "source_sha256": "a" * 64,
        "reference_sha256": "1" * 64,
    }
    (experiment_dir / run_module.PROVENANCE_FILE).write_text(json.dumps(provenance))
    np.savez_compressed(
        experiment_dir / run_module.TRACE_FILE,
        schema_version=np.int64(3),
        baseline_timeline=np.array([0.02], dtype=np.float64),
        reference_qpos=np.zeros((1, 7), dtype=np.float64),
        baseline_qpos=np.zeros((1, 7), dtype=np.float64),
        optimized_qpos=np.zeros((1, 7), dtype=np.float64),
    )
    return model


@pytest.mark.parametrize("automatic_render", (False, True))
def test_render_reads_presentation_values_from_persisted_resolved_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic_render: bool
) -> None:
    resolved, _ = load_experiment(EXPERIMENT)
    resolved["visualization"] = {
        "enabled": automatic_render,
        "fps": 17,
        "width": 96,
        "height": 64,
        "distance": 3.5,
        "azimuth": 75.0,
        "elevation": -12.0,
        "reference_lookat_height": 1.1,
        "camera_mode": "reference-follow",
        "show_root_error": False,
        "show_time": True,
    }
    model = _write_v2_render_inputs(tmp_path, resolved)
    captured: dict[str, object] = {}

    def fake_render(**kwargs: object) -> RenderProvenance:
        config = kwargs["config"]
        bindings = kwargs["bindings"]
        output_path = Path(kwargs["output_path"])
        assert isinstance(config, RenderConfig)
        assert isinstance(bindings, RenderBindings)
        view = kwargs["view"]
        assert isinstance(view, ComparisonView)
        assert view.left.label == "baseline"
        assert view.right.label == "optimized"
        captured["config"] = config
        output_path.write_bytes(b"paired-mp4")
        return RenderProvenance(
            renderer_contract_version="test-v1",
            config=config,
            view="comparison",
            track_labels=("baseline", "optimized"),
            release_version=bindings.release_version,
            source_sha256=bindings.source_sha256,
            reference_sha256=bindings.reference_sha256,
            trace_sha256=bindings.trace_sha256,
            report_sha256=bindings.report_sha256,
            mjcf_sha256=bindings.mjcf_sha256,
            frame_indices=(0,),
            output_sha256=hashlib.sha256(b"paired-mp4").hexdigest(),
        )

    monkeypatch.setattr(run_module, "render_mp4", fake_render)
    output = run_module._render_experiment(tmp_path, model_path=model)

    assert output == tmp_path / run_module.RENDER_FILE
    expected = {**resolved["visualization"], "enabled": True}
    assert captured["config"] == RenderConfig(**expected)
    render_provenance = json.loads(
        (tmp_path / run_module.RENDER_PROVENANCE_FILE).read_text()
    )
    assert render_provenance["config"] == expected
    assert yaml.safe_load(
        (tmp_path / "resolved_experiment.yaml").read_text()
    ) == resolved
    assert render_provenance["output_sha256"] == hashlib.sha256(
        b"paired-mp4"
    ).hexdigest()


@pytest.mark.parametrize(
    "artifact_name",
    (
        run_module.REPORT_FILE,
        run_module.PLANNING_FILE,
        run_module.RANDOMIZATION_FILE,
        run_module.PROVENANCE_FILE,
        run_module.TRACE_FILE,
    ),
)
def test_render_rejects_missing_live_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    resolved, _ = load_experiment(EXPERIMENT)
    model = _write_v2_render_inputs(tmp_path, resolved)
    (tmp_path / artifact_name).unlink()

    with pytest.raises(FileNotFoundError):
        run_module._render_experiment(tmp_path, model_path=model)


@pytest.mark.parametrize(
    "artifact_name",
    (
        run_module.REPORT_FILE,
        run_module.PLANNING_FILE,
        run_module.RANDOMIZATION_FILE,
        run_module.PROVENANCE_FILE,
        run_module.TRACE_FILE,
    ),
)
def test_render_rejects_non_v2_live_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    resolved, _ = load_experiment(EXPERIMENT)
    model = _write_v2_render_inputs(tmp_path, resolved)
    path = tmp_path / artifact_name
    if artifact_name == run_module.TRACE_FILE:
        np.savez_compressed(path, schema_version=np.int64(1))
    else:
        path.write_text(json.dumps({"schema_version": 1}))

    with pytest.raises(ValueError, match="schema_version"):
        run_module._render_experiment(tmp_path, model_path=model)


def test_prepare_bfm_rebuilds_and_verifies_locked_recipe(tmp_path: Path) -> None:
    source = tmp_path / "artifacts" / "source.onnx"
    prepared = tmp_path / "artifacts" / "prepared.onnx"
    source.parent.mkdir()
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 2])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 2])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["value"], ["result"])],
            "tiny",
            [value],
            [result],
        )
    )
    onnx.save(model, source)
    expected_model = onnx.load(source)
    for tensor in (*expected_model.graph.input, *expected_model.graph.output):
        leading = tensor.type.tensor_type.shape.dim[0]
        leading.ClearField("dim_value")
        leading.dim_param = "batch"
    expected = tmp_path / "expected.onnx"
    onnx.save(expected_model, expected)
    lock = tmp_path / "lock.json"
    payload = {
        "schema_version": 1,
        "artifacts": {
            "source": {
                "kind": "file",
                "path": "artifacts/source.onnx",
                "sha256": run_module.sha256_file(source),
                "bytes": source.stat().st_size,
            },
            "prepared": {
                "kind": "file",
                "path": "artifacts/prepared.onnx",
                "sha256": run_module.sha256_file(expected),
                "bytes": expected.stat().st_size,
                "source_id": "source",
                "transform": "relax_leading",
                "transform_version": 1,
            },
        },
    }
    lock.write_text(json.dumps(payload))

    result = run_module.prepare_bfm("prepared", root=tmp_path, lock_path=lock)

    assert result == prepared
    transformed = onnx.load(prepared)
    for tensor in (*transformed.graph.input, *transformed.graph.output):
        leading = tensor.type.tensor_type.shape.dim[0]
        assert not leading.HasField("dim_value")
        assert leading.dim_param == "batch"
    assert run_module.sha256_file(prepared) == run_module.sha256_file(expected)
    assert json.loads(lock.read_text()) == payload


def _contains_ref(value: object) -> bool:
    if isinstance(value, dict):
        return "ref" in value or any(_contains_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }


def test_cem_convergence_mode_requires_explicit_criteria() -> None:
    from sampling_optimizers import ConvergenceCriteria
    values = yaml.safe_load((CONFIGS / 'optimizers/cem.yaml').read_text())
    values.update(num_iterations='inf', convergence={'patience': 5, 'min_cost_improvement': 0.0001})
    optimizer, limit = build_optimizer(values, num_knots=3, dim=2)
    assert limit == ConvergenceCriteria(patience=5, min_cost_improvement=0.0001)
    assert optimizer.num_candidates == values['num_candidates']
    params, report = optimizer.optimize(
        optimizer.init_params(seed=47),
        lambda candidates: np.zeros(len(candidates)),
        iterations=limit,
    )
    assert params.iteration == len(report.iterations) == 6


@pytest.mark.parametrize('updates,match', [
    ({'num_iterations': 'inf'}, 'convergence'),
    ({'num_iterations': 'inf', 'convergence': {}}, 'patience'),
    ({'num_iterations': 'inf', 'convergence': {'patience': 5, 'min_cost_improvement': 0.01, 'max_iterations': 10}}, 'unknown'),
    ({'num_iterations': 3, 'convergence': {'patience': 5, 'min_cost_improvement': 0.01}}, 'only.*inf'),
    ({'algorithm': 'weighted_elites', 'num_iterations': 'inf', 'convergence': {'patience': 5, 'min_cost_improvement': 0.01}}, 'only.*cem'),
    ({'num_iterations': float('inf')}, 'positive integer'),
])
def test_optimizer_rejects_ambiguous_convergence(updates, match) -> None:
    values = yaml.safe_load((CONFIGS / 'optimizers/cem.yaml').read_text())
    values.update(updates)
    with pytest.raises((ConfigError, ValueError), match=match):
        build_optimizer(values, num_knots=3, dim=2)


def test_prepare_bfm_requires_explicit_artifact() -> None:
    with pytest.raises(TypeError, match='artifact_id'):
        run_module.prepare_bfm()
    with pytest.raises(SystemExit) as exc:
        run_module.main(['prepare-bfm'])
    assert exc.value.code == 2
