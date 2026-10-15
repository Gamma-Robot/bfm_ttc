"""Shared numerical bounds for real-family MJWarp wave conformance."""

import os
from pathlib import Path

import numpy as np
import yaml

from bfm_adapters import MOTION_COMMAND_ARRAY_FIELDS as COMMAND_ARRAYS
from command_optimization.run import load_experiment, run_experiment


# Fourteen repeated SONIC paired runs on one RTX 4090 observed a maximum
# absolute qpos drift of 1.74e-5 between B=5 and B=3; the remaining runs were
# bitwise equal. Keep a small explicit margin without weakening the physical
# trace check to candidate identity alone.
QPOS_DRIFT_ATOL = 3.0e-5


def assert_qpos_drift_bounded(
    left: np.ndarray, right: np.ndarray, *, atol: float = QPOS_DRIFT_ATOL
) -> None:
    np.testing.assert_allclose(left, right, rtol=0.0, atol=atol)


def assert_real_paired_wave_invariance(
    experiment: Path,
    tmp_path: Path,
    *,
    selected_cost_rtol: float,
    qpos_atol: float = QPOS_DRIFT_ATOL,
) -> None:
    """Run one real family at B=5/B=3 and compare semantic outputs."""
    device = int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))
    resolved, _ = load_experiment(experiment)
    results = []
    for max_worlds in (5, 3):
        cell = dict(resolved)
        cell["name"] = f"{experiment.stem}_b{max_worlds}"
        cell["planning"] = {**resolved["planning"], "max_worlds": max_worlds}
        config_path = tmp_path / f"b{max_worlds}.yaml"
        config_path.write_text(yaml.safe_dump(cell, sort_keys=True))
        result = run_experiment(
            config_path,
            configs_root=Path(__file__).resolve().parents[2] / "configs",
            device_index=device,
            output_dir=tmp_path / f"output_b{max_worlds}",
        )
        assert result is not None
        assert result.pair.baseline.trace.success
        assert result.pair.optimized.trace.success
        results.append(result.pair)

    one_wave, two_waves = results
    assert_qpos_drift_bounded(
        one_wave.baseline.trace.qpos,
        two_waves.baseline.trace.qpos,
        atol=qpos_atol,
    )
    assert_qpos_drift_bounded(
        one_wave.optimized.trace.qpos,
        two_waves.optimized.trace.qpos,
        atol=qpos_atol,
    )
    for field in COMMAND_ARRAYS:
        np.testing.assert_allclose(
            getattr(one_wave.optimized.trace.applied_command, field),
            getattr(two_waves.optimized.trace.applied_command, field),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    assert [item.selected_candidate_id for item in one_wave.optimized.planning] == [
        item.selected_candidate_id for item in two_waves.optimized.planning
    ]
    assert len(one_wave.optimized.planning) == len(two_waves.optimized.planning) > 0
    np.testing.assert_allclose(
        [item.selected_cost for item in one_wave.optimized.planning],
        [item.selected_cost for item in two_waves.optimized.planning],
        rtol=selected_cost_rtol,
        atol=1.0e-6,
    )
    assert one_wave.optimized.planning[0].logical_rollouts == 5
    assert two_waves.optimized.planning[0].logical_rollouts == 5
    assert one_wave.optimized.planning[0].physical_rollouts == 5
    assert two_waves.optimized.planning[0].physical_rollouts == 6
