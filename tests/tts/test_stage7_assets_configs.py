"""Public input examples resolve against one shared optimization preset."""

from __future__ import annotations

from pathlib import Path

import pytest

from command_optimization.config import resolve_config
from command_optimization.run import load_experiment, run_experiment


REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "configs" / "experiments"


@pytest.mark.parametrize("name", ["single_motion", "local_motion"])
def test_public_motion_example_resolves_and_dry_runs(name: str) -> None:
    path = EXPERIMENTS / f"{name}.yaml"
    resolved, config = load_experiment(path)
    preset = resolve_config(EXPERIMENTS / "cem_fixed_budget.base.yaml", REPO / "configs")

    assert not _contains_ref(resolved)
    assert config.name == name
    assert config.bfm["family"] == "heft"
    for section in ("optimizer", "planning", "motion_tracking"):
        assert resolved[section] == preset[section]
    assert preset["robot"]["physics_timestep"] == 0.005
    assert resolved["robot"] == preset["robot"]
    assert config.bfm["control_period"] == 0.02
    assert config.reference.clip_duration_seconds is None
    assert set(resolved["output"]["formats"]) == {"npz", "any4hdmi"}
    assert set(resolved["output"]["trajectories"]) == {"execution", "command"}
    assert config.reference.dataset_root is not None
    assert Path(config.reference.dataset_root).is_absolute()
    assert config.reference.manifest_artifact is None
    assert config.reference.motion_artifact is None
    if name == "single_motion":
        assert Path(config.reference.dataset_root) == REPO / "data"
        assert config.reference.motion_id == "0019"

    assert run_experiment(path, dry_run=True) is None


def _contains_ref(value: object) -> bool:
    if isinstance(value, dict):
        return "ref" in value or any(_contains_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False
