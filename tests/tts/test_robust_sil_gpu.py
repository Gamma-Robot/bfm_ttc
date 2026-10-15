from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from bfm_adapters import MOTION_COMMAND_ARRAY_FIELDS as COMMAND_ARRAYS
from command_optimization.run import RANDOMIZATION_FILE, load_experiment, run_experiment
from wave_invariance import assert_qpos_drift_bounded


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
ROBUSTNESS = {
    "enabled": True,
    "num_scenarios": 4,
    "scenario_seed": 1313,
    "risk": "mean",
    "ranges": {
        "contact_friction_scale": [0.8, 1.2],
        "joint_armature_scale": [0.8, 1.2],
        "joint_frictionloss_scale": [0.8, 1.2],
        "actuation_strength_scale": [0.9, 1.1],
        "root_linear_velocity_std": 0.02,
        "root_angular_velocity_std": 0.05,
        "joint_velocity_std": 0.05,
    },
}


def _variant(
    tmp_path: Path,
    experiment: str,
    name: str,
    *,
    max_worlds: int,
    robustness: dict | None = None,
    optimizer: dict | None = None,
) -> Path:
    resolved, _ = load_experiment(REPO / "tests" / "fixtures" / "experiments" / experiment)
    resolved["name"] = name
    resolved["planning"] = {
        **resolved["planning"],
        "max_worlds": max_worlds,
    }
    if robustness is not None:
        resolved["robustness"] = robustness
    if optimizer is not None:
        resolved["optimizer"] = optimizer
    resolved["visualization"] = {**resolved["visualization"], "enabled": False}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=True))
    return path


def _run(path: Path, output: Path):
    result = run_experiment(
        path,
        configs_root=CONFIGS,
        device_index=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")),
        output_dir=output,
    )
    assert result is not None
    return result


def test_robust_heft_is_wave_size_invariant_and_auditable(tmp_path: Path) -> None:
    paths = (
        _variant(
            tmp_path,
            "heft_smoke.yaml",
            "robust_b20",
            max_worlds=20,
            robustness=ROBUSTNESS,
        ),
        _variant(
            tmp_path,
            "heft_smoke.yaml",
            "robust_b12",
            max_worlds=12,
            robustness=ROBUSTNESS,
        ),
    )
    results = tuple(
        _run(path, tmp_path / f"output_{index}")
        for index, path in enumerate(paths)
    )
    left, right = (result.pair.optimized for result in results)

    assert [item.scenario_digest for item in left.planning] == [
        item.scenario_digest for item in right.planning
    ]
    assert [item.selected_candidate_id for item in left.planning] == [
        item.selected_candidate_id for item in right.planning
    ]
    np.testing.assert_allclose(
        [item.selected_cost for item in left.planning],
        [item.selected_cost for item in right.planning],
        # Separate MJWarp batch shapes are not bitwise deterministic. The
        # observed bound in this real robust gate is 7.8e-4 relative while
        # candidate identity remains exact.
        rtol=1.0e-3,
        atol=1.0e-7,
    )
    assert all(item.candidate_evaluations == 5 for item in left.planning)
    assert all(item.logical_rollouts == 20 for item in left.planning)
    assert all(item.physical_rollouts == 20 for item in left.planning)
    assert all(item.physical_rollouts == 24 for item in right.planning)
    assert_qpos_drift_bounded(left.trace.qpos, right.trace.qpos)
    for field in COMMAND_ARRAYS:
        np.testing.assert_allclose(
            getattr(left.trace.applied_command, field),
            getattr(right.trace.applied_command, field),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    for result in results:
        evidence = json.loads(
            (result.output_dir / RANDOMIZATION_FILE).read_text()
        )
        assert evidence["num_scenarios"] == 4
        assert [item["scenario_digest"] for item in evidence["updates"]] == [
            item.scenario_digest for item in result.pair.optimized.planning
        ]


def test_zero_randomization_r1_matches_reference_path(tmp_path: Path) -> None:
    reference_path = _variant(
        tmp_path, "heft_smoke.yaml", "reference_motion", max_worlds=5
    )
    parity_path = _variant(
        tmp_path,
        "heft_smoke.yaml",
        "robust_r1_reference_motion",
        max_worlds=5,
        robustness={
            "enabled": True,
            "num_scenarios": 1,
            "scenario_seed": 13,
            "risk": "mean",
            "ranges": {},
        },
    )
    reference_motion = _run(reference_path, tmp_path / "reference_output").pair
    parity = _run(parity_path, tmp_path / "parity_output").pair

    assert [item.selected_candidate_id for item in reference_motion.optimized.planning] == [
        item.selected_candidate_id for item in parity.optimized.planning
    ]
    np.testing.assert_allclose(
        [item.selected_cost for item in reference_motion.optimized.planning],
        [item.selected_cost for item in parity.optimized.planning],
        rtol=1.0e-5,
        # Ten nominal repeats on an RTX 4070 Laptop varied by 6.2e-7;
        # use the same absolute bound as the paired wave-conformance tests.
        atol=1.0e-6,
    )
    assert_qpos_drift_bounded(
        reference_motion.optimized.trace.qpos, parity.optimized.trace.qpos
    )


@pytest.mark.parametrize(
    "experiment,optimizer",
    [
        ("twist2_smoke.yaml", None),
        ("heft_smoke.yaml", None),
        ("sonic_smoke.yaml", None),
        (
            "heft_smoke.yaml",
            {
                "algorithm": "mppi",
                "num_candidates": 5,
                "num_iterations": 1,
                "noise_level": 0.2,
                "temperature": 0.1,
            },
        ),
    ],
)
def test_robust_evaluation_crosses_bfm_and_optimizer_seams(
    tmp_path: Path, experiment: str, optimizer: dict | None
) -> None:
    name = f"robust_{Path(experiment).stem}_{'default' if optimizer is None else 'mppi'}"
    path = _variant(
        tmp_path,
        experiment,
        name,
        max_worlds=20,
        robustness=ROBUSTNESS,
        optimizer=optimizer,
    )

    result = _run(path, tmp_path / f"{name}_output")

    assert result.pair.baseline.trace.success
    assert result.pair.optimized.trace.success
    assert result.pair.optimized.planning
    expected_candidates = 5 if optimizer is None else 6
    for update in result.pair.optimized.planning:
        assert update.scenario_count == 4
        assert update.candidate_evaluations == expected_candidates
        assert update.logical_rollouts == expected_candidates * 4
        assert update.penalized_candidate_evaluations == 0
        assert update.invalid_scenario_rollouts == 0
