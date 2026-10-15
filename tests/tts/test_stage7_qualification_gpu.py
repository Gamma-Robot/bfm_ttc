from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from command_optimization.run import load_experiment, run_experiment


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


@pytest.mark.parametrize("family", ("twist2", "heft", "sonic"))
def test_bfm_completes_common_any4hdmi_qualification(
    family: str, tmp_path: Path
) -> None:
    experiment = REPO / "tests" / "fixtures" / "experiments" / f"{family}_jump_qualification.yaml"
    _, config = load_experiment(experiment)
    result = run_experiment(
        experiment,
        configs_root=CONFIGS,
        device_index=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")),
        output_dir=tmp_path / family,
    )

    assert result is not None
    pair = result.pair
    assert pair.baseline.trace.success
    assert pair.optimized.trace.success
    assert pair.baseline.trace.num_steps == pair.optimized.trace.num_steps == 20
    assert pair.baseline.reference_id == pair.optimized.reference_id == (
        "local:0019"
    )
    assert pair.baseline.reference_digest == pair.optimized.reference_digest
    assert pair.baseline.bfm_id == pair.optimized.bfm_id == (
        config.bfm["instance"]
    )
    assert pair.baseline.checkpoint_digest == pair.optimized.checkpoint_digest
    assert len(pair.baseline.checkpoint_digest) == 64
    assert pair.baseline.robot_model_digest == pair.optimized.robot_model_digest
    assert len(pair.optimized.planning) == 1
    update = pair.optimized.planning[0]
    assert update.logical_rollouts == 5
    assert update.physical_rollouts == 5
    assert update.best_observed_cost < config.planning.failure_penalty
    assert update.proposal_cost is not None
    assert update.proposal_cost < config.planning.failure_penalty
    assert np.isfinite(update.cost_quantiles).all()
