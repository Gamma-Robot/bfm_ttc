from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import command_optimization.run as run_module
import command_optimization.batch as batch_run_module
from command_optimization.artifacts import sha256_file
from command_optimization.config import (
    RESOLVED_EXPERIMENT_FILE,
    ConfigError,
    dataclass_from_dict,
)
from command_optimization.simulation.randomization import RobustnessConfig
from command_optimization.batch import (
    BudgetPoint,
    MotionReference,
    OptimizerVariant,
    EXPERIMENT_BATCH_SUMMARY_FILE,
    ExperimentBatchResult,
    ExperimentBatchConfig,
    _expand_cells,
    _safe_experiment_ref,
    _safe_optimizer_ref,
    load_experiment_batch,
    run_experiment_batch,
)


REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
PROFILE = REPO / "tests" / "fixtures" / "experiments" / "candidate_count_profile.batch.yaml"


def test_batch_expands_in_stable_bfm_variant_budget_seed_order(tmp_path: Path) -> None:
    path = _write_batch(tmp_path, seeds=[1, 2])
    value = yaml.safe_load(path.read_text())
    value["budgets"] = [
        {"id": "n5-i1", "num_candidates": 5, "num_iterations": 1},
        {"id": "n9-i2", "num_candidates": 9, "num_iterations": 2},
    ]
    path.write_text(yaml.safe_dump(value))
    _, config, cells = load_experiment_batch(path)

    assert [cell.cell_id for cell in cells] == [
        f"{cells[0].motion_id}-{bfm}-weighted_elites-{budget.id}-s{seed}"
        for bfm in ("twist2_1017_20k", "heft_pmg", "sonic_release_g1")
        for budget in config.budgets
        for seed in config.seeds
    ]
    assert all(cell.rollout_batch_size == 5 for cell in cells)
    assert cells[0].candidate_latent_bytes_per_iteration == 5 * 8 * 35 * 8
    for cell in cells:
        assert cell.resolved["name"].endswith(cell.cell_id)
        assert cell.resolved["optimizer"]["num_candidates"] == cell.candidate_count
        assert cell.resolved["optimizer"]["num_iterations"] == cell.iteration_count
        assert cell.resolved["reference"]["clip_duration_seconds"] == 0.4
        assert cell.resolved["planning"]["seed"] == cell.seed
        assert cell.resolved["planning"]["max_worlds"] == min(
            cell.candidate_count, config.rollout_batch_ceiling
        )


def test_rollout_batch_ceiling_does_not_change_candidate_bytes() -> None:
    batch_config = ExperimentBatchConfig(
        schema_version=3,
        name="cap",
        experiment_refs=("../tests/fixtures/experiments/twist2_batch.yaml",),
        clip_duration_seconds=0.4,
        budgets=(BudgetPoint("n257-i1", 257, 1),),
        optimizer_variants=(
            OptimizerVariant(
                "weighted_elites", "optimizers/weighted_elites.yaml", {}
            ),
        ),
        seeds=(47,),
        rollout_batch_ceiling=65,
    )

    (cell,) = _expand_cells(batch_config, configs_root=CONFIGS.resolve())

    assert cell.rollout_batch_size == 65
    assert cell.candidate_latent_bytes_per_iteration == 257 * 8 * 35 * 8


def test_robust_experiment_batch_uses_only_complete_scenario_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = run_module.load_experiment

    def robust(path, **kwargs):
        resolved, config = original(path, **kwargs)
        resolved = copy.deepcopy(resolved)
        resolved["robustness"] = {
            "enabled": True,
            "num_scenarios": 4,
            "scenario_seed": 13,
            "risk": "mean",
            "ranges": {},
        }
        return resolved, dataclasses.replace(
            config,
            robustness=RobustnessConfig(
                enabled=True, num_scenarios=4, scenario_seed=13
            ),
        )

    monkeypatch.setattr(run_module, "load_experiment", robust)
    batch_config = ExperimentBatchConfig(
        schema_version=3,
        name="robust-cap",
        experiment_refs=("../tests/fixtures/experiments/twist2_batch.yaml",),
        clip_duration_seconds=0.4,
        budgets=(BudgetPoint("n5-i1", 5, 1),),
        optimizer_variants=(
            OptimizerVariant("weighted_elites", "optimizers/weighted_elites.yaml", {}),
        ),
        seeds=(47,),
        rollout_batch_ceiling=10,
    )

    (cell,) = _expand_cells(batch_config, configs_root=CONFIGS.resolve())

    assert cell.rollout_batch_size == 8
    assert cell.resolved["planning"]["max_worlds"] == 8
    with pytest.raises(ConfigError, match="complete scenario group"):
        _expand_cells(
            dataclasses.replace(batch_config, rollout_batch_ceiling=3),
            configs_root=CONFIGS.resolve(),
        )


def test_experiments_must_match_outside_experiment_batch_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = run_module.load_experiment

    def mismatched(path, **kwargs):
        resolved, config = original(path, **kwargs)
        if Path(path).name.startswith("heft"):
            resolved = copy.deepcopy(resolved)
            resolved["motion_tracking"]["joint_position_weight"] = 2.0
        return resolved, config

    monkeypatch.setattr(run_module, "load_experiment", mismatched)
    batch_config = ExperimentBatchConfig(
        schema_version=3,
        name="mismatch",
        experiment_refs=(
            "../tests/fixtures/experiments/twist2_batch.yaml",
            "../tests/fixtures/experiments/heft_jump_qualification.yaml",
        ),
        clip_duration_seconds=0.4,
        budgets=(BudgetPoint("n5-i1", 5, 1),),
        optimizer_variants=(
            OptimizerVariant(
                "weighted_elites", "optimizers/weighted_elites.yaml", {}
            ),
        ),
        seeds=(47,),
        rollout_batch_ceiling=5,
    )

    with pytest.raises(ConfigError, match="differs outside"):
        _expand_cells(batch_config, configs_root=CONFIGS.resolve())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"extra": True}, "unknown fields"),
        ({"mode": "formal"}, "unknown.*mode"),
        ({"motions": "jump"}, "motions must be a sequence"),
        ({"schema_version": 1}, "schema_version must be exactly 3"),
        ({"schema_version": True}, "schema_version must be exactly 3"),
        ({"schema_version": 3.0}, "schema_version must be exactly 3"),
        ({"experiment_refs": "experiments/x.yaml"}, "must be a sequence"),
        ({"experiment_refs": []}, "experiment_refs"),
        ({"budgets": []}, "budgets must not be empty"),
        ({"budgets": [{"id": "bad id", "num_candidates": 5, "num_iterations": 1}]}, "must match"),
        ({"budgets": [{"id": "n", "num_candidates": True, "num_iterations": 1}]}, ">= 1"),
        ({"optimizer_variants": []}, "optimizer_variants must not be empty"),
        (
            {
                "optimizer_variants": [
                    {
                        "id": "weighted_elites",
                        "ref": "optimizers/weighted_elites.yaml",
                        "overrides": {"algorithm": "cem"},
                    }
                ]
            },
            "cannot set fields",
        ),
        ({"clip_duration_seconds": 0.0}, "finite and > 0"),
        ({"clip_duration_seconds": True}, "finite and > 0"),
        ({"seeds": [1, 1]}, "unique"),
        ({"seeds": [-1]}, ">= 0"),
        ({"rollout_batch_ceiling": True}, "integers"),
    ],
)
def test_experiment_batch_config_rejects_unknown_or_invalid_axes(
    override: dict, match: str
) -> None:
    data = {
        "schema_version": 3,
        "name": "experiment_batch",
        "experiment_refs": ["experiments/x.yaml"],
        "clip_duration_seconds": 0.4,
        "budgets": [{"id": "n5-i1", "num_candidates": 5, "num_iterations": 1}],
        "optimizer_variants": [
            {"id": "weighted_elites", "ref": "optimizers/weighted_elites.yaml", "overrides": {}}
        ],
        "seeds": [1],
        "rollout_batch_ceiling": 5,
        **override,
    }

    with pytest.raises(ConfigError, match=match):
        dataclass_from_dict(ExperimentBatchConfig, data)


def test_experiment_batch_config_requires_schema_version() -> None:
    data = yaml.safe_load(PROFILE.read_text())
    data.pop("schema_version")

    with pytest.raises(ConfigError, match="schema_version"):
        dataclass_from_dict(ExperimentBatchConfig, data)


def test_experiment_batch_does_not_support_ref_inheritance(tmp_path: Path) -> None:
    path = _write_batch(tmp_path, seeds=[1])
    value = yaml.safe_load(path.read_text())
    value["ref"] = "another_batch.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=True))

    with pytest.raises(ConfigError, match="unknown fields.*ref"):
        load_experiment_batch(path)


@pytest.mark.parametrize("ref", ("../outside.yaml", "/tmp/absolute.yaml", "x.txt"))
def test_experiment_refs_cannot_escape_configs_root(ref: str) -> None:
    with pytest.raises(ConfigError, match="unsafe"):
        _safe_experiment_ref(ref, CONFIGS.resolve())


@pytest.mark.parametrize(
    "ref", ("../outside.yaml", "/tmp/absolute.yaml", "experiments/missing.yaml")
)
def test_optimizer_refs_are_confined_to_optimizers(ref: str) -> None:
    with pytest.raises(ConfigError, match="unsafe"):
        _safe_optimizer_ref(ref, CONFIGS.resolve())


def test_dry_run_preflights_every_cell_without_creating_output(tmp_path: Path) -> None:
    calls: list[tuple[Path, bool]] = []

    def fake_executor(path, *, dry_run, **_kwargs):
        calls.append((Path(path), dry_run))

    result = run_experiment_batch(
        PROFILE,
        execute_experiment=fake_executor,
        output_dir=tmp_path / "unused",
        dry_run=True,
    )

    assert result is None
    assert len(calls) == 9
    assert all(dry for _, dry in calls)
    assert not (tmp_path / "unused").exists()


def test_experiment_batch_recomputes_aggregate_from_persisted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1, 2])
    _clean_source(monkeypatch)
    executor = _FakeExecutor(source_sha256="a" * 64)

    result = run_experiment_batch(
        batch_path,
        execute_experiment=executor,
        output_dir=tmp_path / "output",
    )

    assert result is not None and result.success and result.cells == 6
    assert executor.preflights == 6 and executor.executions == 6
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert summary["schema_version"] == 3
    assert summary["experiment_batch"]["success"] is True
    assert summary["experiment_batch"]["initial_physical_state_digest"] == "a" * 64
    assert "source_qualification" not in summary["experiment_batch"]
    assert len(summary["cells"]) == 6
    assert len(summary["groups"]) == 3
    for row in summary["cells"]:
        assert row["status"] == "success"
        assert row["candidate_evaluations"] == 5
        assert row["logical_rollouts"] == 5
        assert row["physical_rollouts"] == 5
        assert row["penalized_candidate_evaluations"] == 0
        assert row["invalid_scenario_rollouts"] == 0
        assert row["logical_rollout_steps"] == 200
        assert row["physical_rollout_steps"] == 200
        assert row["all_penalty_updates"] == 0
        assert row["candidate_latent_bytes_per_iteration"] == 5 * 8 * 35 * 8
        assert row["hardware"]["gpu_uuid"] == "GPU-test"
        assert row["bfm_runtime"] == "device"
    for group in summary["groups"]:
        assert group["n"] == group["successes"] == 2
        assert group["budget_id"] == "n5-i1"
        assert group["optimizer_variant_id"] == "weighted_elites"
        assert group["statistics"]["compute.candidate_evaluations"]["mean"] == 5.0
        assert group["statistics"]["compute.logical_rollouts"]["mean"] == 5.0
        assert group["statistics"]["optimized.body_position_mean_error_m"][
            "sample_std"
        ] > 0.0


def test_experiment_batch_accepts_observed_nonbest_proposal_without_extra_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)
    executor = _FakeExecutor(
        source_sha256="a" * 64, observed_nonbest_proposal=True
    )

    result = run_experiment_batch(
        batch_path,
        execute_experiment=executor,
        output_dir=tmp_path / "output",
    )

    assert result is not None and result.success
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert all(row["logical_rollouts"] == 5 for row in summary["cells"])


def test_experiment_batch_separates_candidate_and_scenario_rollout_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)

    result = run_experiment_batch(
        batch_path,
        execute_experiment=_FakeExecutor(
            source_sha256="a" * 64, scenario_count=4
        ),
        output_dir=tmp_path / "output",
    )

    assert result is not None and result.success
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert all(row["candidate_evaluations"] == 5 for row in summary["cells"])
    assert all(row["logical_rollouts"] == 20 for row in summary["cells"])
    assert all(row["physical_rollouts"] == 20 for row in summary["cells"])
    assert all(
        group["statistics"]["compute.candidate_evaluations"]["mean"] == 5.0
        for group in summary["groups"]
    )
    assert all(
        group["statistics"]["compute.logical_rollouts"]["mean"] == 20.0
        for group in summary["groups"]
    )


def test_experiment_batch_does_not_infer_failure_from_cost_magnitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)

    result = run_experiment_batch(
        batch_path,
        execute_experiment=_FakeExecutor(
            source_sha256="a" * 64, selected_cost=1.0e300
        ),
        output_dir=tmp_path / "output",
    )

    assert result is not None and result.success
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert all(row["all_penalty_updates"] == 0 for row in summary["cells"])


def test_experiment_batch_rejects_noninteger_logical_rollout_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)
    executor = _FakeExecutor(
        source_sha256="a" * 64, logical_rollouts=5.9
    )

    result = run_experiment_batch(
        batch_path,
        execute_experiment=executor,
        output_dir=tmp_path / "output",
    )

    assert result is not None and not result.success
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert all(row["status"] == "failed" for row in summary["cells"])
    assert all("must be an integer" in row["failure"] for row in summary["cells"])


def test_experiment_batch_persists_failure_and_continues_other_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)
    executor = _FakeExecutor(source_sha256="a" * 64, fail_bfm="heft_pmg")

    result = run_experiment_batch(
        batch_path,
        execute_experiment=executor,
        output_dir=tmp_path / "output",
    )

    assert result is not None and not result.success and result.cells == 3
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    failed = [row for row in summary["cells"] if row["status"] == "failed"]
    successful = [row for row in summary["cells"] if row["status"] == "success"]
    assert len(failed) == 1 and failed[0]["bfm_id"] == "heft_pmg"
    assert set(failed[0]) == set(successful[0])
    assert failed[0]["file_sha256"]["report.json"] is None
    assert failed[0]["baseline"] is None
    assert failed[0]["logical_rollouts"] is None
    assert failed[0]["penalized_candidate_evaluations"] is None
    assert failed[0]["invalid_scenario_rollouts"] is None
    assert failed[0]["all_penalty_updates"] is None
    failure_path = result.output_dir / failed[0]["output"] / "failure.json"
    assert "deliberate failure" in json.loads(failure_path.read_text())["failure"]
    assert executor.executions == 3


@pytest.mark.parametrize("mismatch", (None, "reset", "reference"))
def test_user_batch_expands_local_motions_and_groups_resets_per_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str | None,
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1, 2])
    value = yaml.safe_load(batch_path.read_text())
    value.pop("clip_duration_seconds")
    templates = []
    for name in ("twist2", "heft"):
        fixture = "twist2_batch.yaml" if name == "twist2" else "heft_jump_qualification.yaml"
        resolved, _ = run_module.load_experiment(
            REPO / "tests" / "fixtures" / "experiments" / fixture
        )
        template = tmp_path / f"{name}.yaml"
        template.write_text(yaml.safe_dump(resolved))
        templates.append(str(template) if name == "twist2" else template.name)
    value["experiment_refs"] = templates
    optimizer_path = tmp_path / "optimizer.yaml"
    optimizer_path.write_text((CONFIGS / "optimizers" / "weighted_elites.yaml").read_text())
    value["optimizer_variants"][0]["ref"] = optimizer_path.name
    value["motions"] = [
        {"id": motion, "reference": {"motion_id": motion, "dataset_root": "motions"}}
        for motion in ("jump", "walk")
    ]
    batch_path.write_text(yaml.safe_dump(value))
    _clean_source(monkeypatch)
    _, config, cells = load_experiment_batch(batch_path)
    assert len(cells) == len({cell.cell_id for cell in cells}) == 8
    assert {cell.motion_id for cell in cells} == {"jump", "walk"}
    assert all(cell.clip_duration_seconds is None for cell in cells)
    assert all(
        cell.resolved["reference"]["dataset_root"] == str(tmp_path / "motions")
        for cell in cells
    )
    assert all("motion_artifact" not in cell.resolved["reference"] for cell in cells)
    executor = _FakeExecutor(
        source_sha256="a" * 64,
        different_reset_bfm="heft_pmg" if mismatch == "reset" else None,
        different_target_bfm="heft_pmg" if mismatch == "reference" else None,
    )
    result = run_experiment_batch(
        batch_path, execute_experiment=executor, output_dir=tmp_path / "output"
    )

    assert result is not None and result.success == (mismatch is None)
    assert executor.preflights == executor.executions == 8
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    batch = summary["experiment_batch"]
    assert batch["source_sha256"] == "a" * 64
    assert batch["release_version"] == "0.1.0"
    assert len(summary["groups"]) == 4
    assert all(group["n"] == group["successes"] == 2 for group in summary["groups"])
    if mismatch is None:
        assert batch["initial_physical_state_digests"] == {"jump": "a" * 64, "walk": "b" * 64}
        assert batch["invariant_failures"] == []
    else:
        assert len(batch["invariant_failures"]) == 2
        expected = "inconsistent initial" if mismatch == "reset" else "inconsistent reference"
        assert all(expected in failure for failure in batch["invariant_failures"])


def test_templates_can_repeat_bfm_for_different_motions() -> None:
    _, config, _ = load_experiment_batch(PROFILE)
    config = dataclasses.replace(
        config,
        experiment_refs=(
            "../tests/fixtures/experiments/heft_jump_qualification.yaml",
            "../tests/fixtures/experiments/heft_walk.yaml",
        ),
        budgets=(BudgetPoint("n5-i1", 5, 1),),
    )
    cells = _expand_cells(config, configs_root=CONFIGS.resolve())
    assert len(cells) == 2
    assert cells[0].bfm_id == cells[1].bfm_id
    assert cells[0].motion_id != cells[1].motion_id
    with pytest.raises(ConfigError, match="duplicate motion/bfm pair"):
        _expand_cells(
            dataclasses.replace(
                config,
                motions=(MotionReference("same", cells[0].resolved["reference"]),),
            ),
            configs_root=CONFIGS.resolve(),
        )


def test_all_penalty_cell_retains_available_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_path = _write_batch(tmp_path, seeds=[1])
    _clean_source(monkeypatch)
    executor = _FakeExecutor(
        source_sha256="a" * 64, all_penalty_bfm="twist2_1017_20k"
    )

    result = run_experiment_batch(
        batch_path,
        execute_experiment=executor,
        output_dir=tmp_path / "output",
    )

    assert result is not None and not result.success
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    (row,) = [
        cell for cell in summary["cells"] if cell["bfm_id"] == "twist2_1017_20k"
    ]
    assert row["status"] == "failed"
    assert row["all_penalty_updates"] == 1
    assert row["baseline"]["success"] is True
    assert row["hardware"]["gpu_uuid"] == "GPU-test"
    assert row["logical_rollouts"] == 5
    assert row["penalized_candidate_evaluations"] == 5
    assert row["invalid_scenario_rollouts"] == 5
    assert "all-penalty" in row["failure"]


def test_cli_returns_nonzero_for_persisted_failed_experiment_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import command_optimization.batch as batch_run_module

    monkeypatch.setattr(
        batch_run_module,
        "run_experiment_batch",
        lambda *_args, **_kwargs: ExperimentBatchResult(tmp_path, False, 3),
    )

    assert run_module.main(["batch", str(tmp_path / "experiment.batch.yaml")]) == 1


def _write_batch(root: Path, *, seeds: list[int]) -> Path:
    path = root / "experiment.batch.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "name": "test_response",
                "experiment_refs": [
                    "../tests/fixtures/experiments/twist2_batch.yaml",
                    "../tests/fixtures/experiments/heft_jump_qualification.yaml",
                    "../tests/fixtures/experiments/sonic_jump_qualification.yaml",
                ],
                "clip_duration_seconds": 0.4,
                "budgets": [
                    {"id": "n5-i1", "num_candidates": 5, "num_iterations": 1}
                ],
                "optimizer_variants": [
                    {
                        "id": "weighted_elites",
                        "ref": "optimizers/weighted_elites.yaml",
                        "overrides": {"noise_scale": 0.2},
                    }
                ],
                "seeds": seeds,
                "rollout_batch_ceiling": 5,
            },
            sort_keys=True,
        )
    )
    return path


def _clean_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from command_optimization.provenance import SourceIdentity
    monkeypatch.setattr(batch_run_module, "source_identity", lambda _repo: SourceIdentity("0.1.0", "a" * 64))


class _FakeExecutor:
    def __init__(
        self,
        *,
        source_sha256: str,
        fail_bfm: str | None = None,
        all_penalty_bfm: str | None = None,
        observed_nonbest_proposal: bool = False,
        logical_rollouts: int | float | None = None,
        scenario_count: int = 1,
        selected_cost: float | None = None,
        different_reset_bfm: str | None = None,
        different_target_bfm: str | None = None,
    ) -> None:
        self.source_sha256 = source_sha256
        self.fail_bfm = fail_bfm
        self.all_penalty_bfm = all_penalty_bfm
        self.observed_nonbest_proposal = observed_nonbest_proposal
        self.logical_rollouts = logical_rollouts
        self.scenario_count = scenario_count
        self.selected_cost = selected_cost
        self.different_reset_bfm = different_reset_bfm
        self.different_target_bfm = different_target_bfm
        self.preflights = 0
        self.executions = 0

    def __call__(
        self,
        experiment,
        *,
        output_dir,
        dry_run,
        **_kwargs,
    ):
        if dry_run:
            self.preflights += 1
            return None
        self.executions += 1
        config = yaml.safe_load(Path(experiment).read_text())
        bfm = config["bfm"]["instance"]
        if bfm == self.fail_bfm:
            raise RuntimeError("deliberate failure")
        output = Path(output_dir)
        output.mkdir(parents=True)
        resolved_experiment = output / RESOLVED_EXPERIMENT_FILE
        resolved_experiment.write_text(yaml.safe_dump(config, sort_keys=True))
        seed = config["planning"]["seed"]
        motion_id = config["reference"]["motion_id"]
        reset_digest = "a" * 64 if motion_id in {"jump", "0019"} else "b" * 64
        if bfm == self.different_reset_bfm:
            reset_digest = "d" * 64
        reference_digest = "e" * 64 if bfm == self.different_target_bfm else "t" * 64
        report = {
            "schema_version": 3,
            "claim_scope": "test",
            "identities": {
                "bfm_id": bfm,
                "reference_id": f"test:{motion_id}",
                "reference_digest": reference_digest,
                "checkpoint_digest": "c" * 64,
                "robot_model_digest": "r" * 64,
                "initial_physical_state_digest": reset_digest,
                "control_period": 0.02,
            },
            "baseline": _metrics(1.0),
            "optimized": _metrics(0.5 + 0.01 * seed),
        }
        (output / "report.json").write_text(json.dumps(report))
        all_penalty = bfm == self.all_penalty_bfm
        if all_penalty:
            selected_cost = float(config["planning"]["failure_penalty"])
        elif self.selected_cost is None:
            selected_cost = 1.0
        else:
            selected_cost = self.selected_cost
        logical_rollouts = (
            5 * self.scenario_count
            if self.logical_rollouts is None
            else self.logical_rollouts
        )
        planning_updates = [
            {
                "update_step": 0,
                "selected_cost": selected_cost,
                "best_observed_candidate_id": "best",
                "proposal_candidate_id": (
                    "observed-nonbest" if self.observed_nonbest_proposal else "best"
                ),
                "cost_quantiles": [[selected_cost] * 5],
                "candidate_deformation": {
                    "body_position_rmse_m": 0.01 * seed,
                    "joint_position_rmse_rad": 0.02,
                },
                "candidate_evaluations": 5,
                "scenario_count": self.scenario_count,
                "scenario_digest": (
                    None if self.scenario_count == 1 else "sha256:scenario"
                ),
                "logical_rollouts": logical_rollouts,
                "physical_rollouts": (
                    logical_rollouts if isinstance(logical_rollouts, int) else 5
                ),
                "penalized_candidate_evaluations": 5 if all_penalty else 0,
                "invalid_scenario_rollouts": (
                    5 * self.scenario_count if all_penalty else 0
                ),
                "update_wall_seconds": 0.1 + 0.01 * seed,
            }
        ]
        planning = {"schema_version": 3, "updates": planning_updates}
        (output / "planning.json").write_text(json.dumps(planning))
        randomization = {
            "schema_version": 3,
            "enabled": self.scenario_count > 1,
            "num_scenarios": self.scenario_count,
            "scenario_seed": 0,
            "risk": "mean",
            "ranges": {},
            "updates": [
                {
                    "update_step": item["update_step"],
                    "seed": None,
                    "scenario_digest": item["scenario_digest"],
                }
                for item in planning_updates
            ],
        }
        (output / "randomization.json").write_text(json.dumps(randomization))
        np.savez_compressed(
            output / "paired_traces.npz",
            schema_version=np.int64(3),
            identity=np.asarray(f"trace-{bfm}-{seed}"),
        )
        provenance = {
            "schema_version": 3,
            "source_sha256": self.source_sha256,
            "release_version": "0.1.0",
            "bfm_runtime": "device",
            "reference_sha256": reference_digest,
            "report_sha256": sha256_file(output / "report.json"),
            "planning_sha256": sha256_file(output / "planning.json"),
            "randomization_sha256": sha256_file(output / "randomization.json"),
            "resolved_experiment_sha256": sha256_file(resolved_experiment),
            "trace_sha256": sha256_file(output / "paired_traces.npz"),
            "hardware": {
                "available": True,
                "gpu_uuid": "GPU-test",
                "gpu_name": "NVIDIA GeForce RTX 4090",
            },
        }
        (output / "provenance.json").write_text(json.dumps(provenance))
        return None


def _metrics(body_error: float) -> dict:
    return {
        "success": True,
        "failure_reason": None,
        "failure_step": None,
        "total_steps": 20,
        "valid_steps": 20,
        "body_position_mean_error_m": body_error,
        "root_position_mean_error_m": body_error,
        "joint_position_rmse_rad": 0.1,
        "body_orientation_mean_error_rad": 0.2,
        "root_orientation_mean_error_rad": 0.2,
    }
