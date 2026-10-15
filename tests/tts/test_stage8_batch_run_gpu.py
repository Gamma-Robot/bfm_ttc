from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from command_optimization.run import run_experiment
from command_optimization.batch import EXPERIMENT_BATCH_SUMMARY_FILE, run_experiment_batch


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
FIXTURES = REPO / "tests" / "fixtures" / "experiments"


def _device() -> int:
    return int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))


def test_user_batch_runs_two_motions_and_two_bfms(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 3,
        "name": "user-batch",
        "experiment_refs": [
            str(FIXTURES / "heft_smoke.yaml"),
            str(FIXTURES / "sonic_smoke.yaml"),
        ],
        "motions": [
            {
                "id": motion,
                "reference": {
                    "dataset_root": str(REPO / "data"),
                    "motion_id": motion,
                },
            }
            for motion in ("0019", "walk")
        ],
        "clip_duration_seconds": 0.08,
        "budgets": [{"id": "short", "num_candidates": 3, "num_iterations": 1}],
        "optimizer_variants": [
            {"id": "weighted", "ref": "optimizers/weighted_elites.yaml", "overrides": {}},
        ],
        "seeds": [47],
        "rollout_batch_ceiling": 3,
    }
    path = tmp_path / "user.batch.yaml"
    path.write_text(yaml.safe_dump(manifest))
    result = run_experiment_batch(
        path,
        execute_experiment=run_experiment,
        device_index=_device(),
        output_dir=tmp_path / "experiment_batch-output",
        configs_root=CONFIGS,
    )

    assert result is not None and result.success and result.cells == 4
    summary = json.loads((result.output_dir / EXPERIMENT_BATCH_SUMMARY_FILE).read_text())
    assert summary["experiment_batch"]["release_version"]
    assert len(summary["experiment_batch"]["source_sha256"]) == 64
    assert summary["experiment_batch"]["invariant_failures"] == []
    assert {(row["motion_id"], row["bfm_id"]) for row in summary["cells"]} == {
        (motion, bfm)
        for motion in ("0019", "walk")
        for bfm in ("heft_pmg", "sonic_release_g1")
    }
    for motion in ("0019", "walk"):
        rows = [row for row in summary["cells"] if row["motion_id"] == motion]
        assert len({row["initial_physical_state_digest"] for row in rows}) == 1
    assert all(row["status"] == "success" for row in summary["cells"])
    assert all(row["all_penalty_updates"] == 0 for row in summary["cells"])
    for row in summary["cells"]:
        assert row["logical_rollouts"] > 0
        assert row["physical_rollouts"] >= row["logical_rollouts"]
        output = result.output_dir / row["output"]
        assert (output / "paired_traces.npz").is_file()
        assert (output / "planning.json").is_file()
