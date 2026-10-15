from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from bfm_adapters import MOTION_COMMAND_ARRAY_FIELDS as COMMAND_ARRAYS
from command_optimization.artifacts import sha256_file
from command_optimization.config import RESOLVED_EXPERIMENT_FILE
from command_optimization.run import (
    PLANNING_FILE,
    PROVENANCE_FILE,
    RANDOMIZATION_FILE,
    RENDER_FILE,
    RENDER_PROVENANCE_FILE,
    REPORT_FILE,
    TRACE_FILE,
    load_experiment,
    run_experiment,
)
from wave_invariance import assert_real_paired_wave_invariance


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "tests" / "fixtures" / "experiments" / "heft_smoke.yaml"


def test_heft_real_paired_run_is_wave_size_invariant(tmp_path: Path) -> None:
    assert_real_paired_wave_invariance(
        EXPERIMENT, tmp_path, selected_cost_rtol=5.0e-5
    )


def test_g3_real_heft_paired_experiment(tmp_path: Path) -> None:
    device = int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))
    output = tmp_path / "heft_smoke"
    resolved, _ = load_experiment(EXPERIMENT)
    resolved["visualization"]["enabled"] = True
    experiment = tmp_path / "heft_render.yaml"
    experiment.write_text(yaml.safe_dump(resolved, sort_keys=True))
    resolved, config = load_experiment(experiment)
    result = run_experiment(experiment, device_index=device, output_dir=output)

    assert result is not None
    assert result.output_dir == output.resolve()
    pair = result.pair
    baseline = pair.baseline
    optimized = pair.optimized
    for field in (
        "reference_id",
        "reference_digest",
        "bfm_id",
        "checkpoint_digest",
        "robot_model_digest",
        "initial_physical_state_digest",
        "control_period",
    ):
        assert getattr(baseline, field) == getattr(optimized, field)
    assert len(baseline.initial_physical_state_digest) == 64
    assert set(baseline.initial_physical_state_digest) <= set("0123456789abcdef")
    np.testing.assert_array_equal(baseline.trace.timeline, optimized.trace.timeline)
    for field in COMMAND_ARRAYS:
        np.testing.assert_array_equal(
            getattr(baseline.trace.reference_motion, field),
            getattr(optimized.trace.reference_motion, field),
        )

    assert baseline.trace.success and optimized.trace.success
    assert baseline.trace.valid_steps == baseline.trace.num_steps > 0
    assert optimized.trace.valid_steps == optimized.trace.num_steps
    assert baseline.trace.qpos.shape[0] == baseline.trace.num_steps
    assert optimized.trace.qpos.shape[0] == optimized.trace.num_steps
    assert baseline.trace.executed_trajectory is not None
    assert optimized.trace.executed_trajectory is not None

    expected_updates = tuple(
        range(0, optimized.trace.num_steps, config.planning.replanning_interval_steps)
    )
    assert tuple(item.update_step for item in optimized.planning) == expected_updates
    # With one elite, the proposal is the already evaluated best candidate.
    expected_logical = config.optimizer["num_candidates"] * config.optimizer[
        "num_iterations"
    ]
    assert all(item.logical_rollouts == expected_logical for item in optimized.planning)
    assert all(
        item.physical_rollouts >= item.logical_rollouts
        for item in optimized.planning
    )
    assert all(
        item.cost_quantiles.shape
        == (
            config.optimizer["num_iterations"],
            5,
        )
        for item in optimized.planning
    )

    expected_files = {
        RESOLVED_EXPERIMENT_FILE,
        TRACE_FILE,
        PLANNING_FILE,
        RANDOMIZATION_FILE,
        REPORT_FILE,
        PROVENANCE_FILE,
        RENDER_FILE,
        RENDER_PROVENANCE_FILE,
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    assert yaml.safe_load((output / RESOLVED_EXPERIMENT_FILE).read_text()) == resolved

    with np.load(output / TRACE_FILE, allow_pickle=False) as traces:
        np.testing.assert_array_equal(
            traces["baseline_timeline"], baseline.trace.timeline
        )
        np.testing.assert_array_equal(traces["baseline_qpos"], baseline.trace.qpos)
        np.testing.assert_array_equal(traces["optimized_qpos"], optimized.trace.qpos)
        assert traces["reference_qpos"].shape == baseline.trace.qpos.shape

    planning_payload = json.loads((output / PLANNING_FILE).read_text())
    assert planning_payload["schema_version"] == 3
    planning = planning_payload["updates"]
    assert [item["update_step"] for item in planning] == list(expected_updates)
    assert [item["logical_rollouts"] for item in planning] == [
        expected_logical
    ] * len(expected_updates)

    report = json.loads((output / REPORT_FILE).read_text())
    assert report["schema_version"] == 3
    assert report["baseline"]["success"] is True
    assert report["optimized"]["success"] is True
    assert report["baseline"]["total_steps"] == baseline.trace.num_steps
    assert report["optimized"]["total_steps"] == optimized.trace.num_steps
    metric_names = (
        "joint_position_rmse_rad",
        "body_position_mean_error_m",
        "body_orientation_mean_error_rad",
        "root_position_mean_error_m",
        "root_orientation_mean_error_rad",
    )
    assert all(
        np.isfinite(report[kind][metric])
        for kind in ("baseline", "optimized")
        for metric in metric_names
    )
    assert report["identities"] == {
        "reference_id": baseline.reference_id,
        "reference_digest": baseline.reference_digest,
        "bfm_id": baseline.bfm_id,
        "checkpoint_digest": baseline.checkpoint_digest,
        "robot_model_digest": baseline.robot_model_digest,
        "initial_physical_state_digest": baseline.initial_physical_state_digest,
        "control_period": baseline.control_period,
    }

    provenance = json.loads((output / PROVENANCE_FILE).read_text())
    assert provenance["reference_sha256"] == baseline.reference_digest
    assert provenance["trace_sha256"] == sha256_file(output / TRACE_FILE)
    assert provenance["report_sha256"] == sha256_file(output / REPORT_FILE)
    assert provenance["planning_sha256"] == sha256_file(output / PLANNING_FILE)
    assert provenance["randomization_sha256"] == sha256_file(
        output / RANDOMIZATION_FILE
    )
    assert provenance["resolved_experiment_sha256"] == sha256_file(
        output / RESOLVED_EXPERIMENT_FILE
    )
    assert len(provenance["mjcf_sha256"]) == 64
    assert set(provenance["mjcf_sha256"]) <= set("0123456789abcdef")
    assert provenance["visualization"] == resolved["visualization"]
    assert provenance["bfm_runtime"] == "device"
    hardware = provenance["hardware"]
    assert hardware["available"] is True
    assert hardware["logical_device_index"] == device
    assert hardware["gpu_uuid"].startswith("GPU-")
    assert hardware["gpu_name"]
    assert hardware["compute_capability"]
    assert hardware["driver_version"]

    render = json.loads((output / RENDER_PROVENANCE_FILE).read_text())
    assert render["source_sha256"] == provenance["source_sha256"]
    assert render["release_version"] == provenance["release_version"]
    assert render["reference_sha256"] == baseline.reference_digest
    assert render["trace_sha256"] == provenance["trace_sha256"]
    assert render["report_sha256"] == provenance["report_sha256"]
    assert render["mjcf_sha256"] == provenance["mjcf_sha256"]
    assert render["config"] == resolved["visualization"]
    assert render["view"] == "comparison"
    assert render["track_labels"] == ["baseline", "optimized"]
    assert render["frame_indices"]
    assert (output / RENDER_FILE).stat().st_size > 0
    assert render["output_sha256"] == sha256_file(output / RENDER_FILE)
