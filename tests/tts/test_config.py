from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from command_optimization.config import (
    RESOLVED_EXPERIMENT_FILE,
    ConfigError,
    dataclass_from_dict,
    resolve_config,
    write_resolved_experiment,
)


@dataclass
class Inner:
    sigma: float
    knots: int = 4


@dataclass
class Outer:
    name: str
    inner: Inner
    seeds: list[int] = field(default_factory=list)


def _write(root: Path, rel: str, data: dict) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


def test_resolve_ref_with_overrides(tmp_path: Path) -> None:
    _write(tmp_path, "optimizers/mppi.yaml", {"temperature": 0.1, "noise": 0.3})
    exp = _write(
        tmp_path,
        "experiments/e1.yaml",
        {"optimizer": {"ref": "optimizers/mppi.yaml", "noise": 0.5}, "seed": 7},
    )
    resolved = resolve_config(exp, configs_root=tmp_path)
    assert resolved["optimizer"] == {"temperature": 0.1, "noise": 0.5}
    assert resolved["seed"] == 7


def test_resolve_nested_refs_and_deep_merge(tmp_path: Path) -> None:
    _write(tmp_path, "objectives/base.yaml", {"weights": {"a": 1.0, "b": 2.0}})
    _write(
        tmp_path,
        "optimizers/p.yaml",
        {"objective": {"ref": "objectives/base.yaml", "weights": {"b": 9.0}}},
    )
    exp = _write(tmp_path, "experiments/e2.yaml", {"optimizer": {"ref": "optimizers/p.yaml"}})
    resolved = resolve_config(exp, configs_root=tmp_path)
    assert resolved["optimizer"]["objective"]["weights"] == {"a": 1.0, "b": 9.0}


def test_resolve_ref_cycle_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", {"x": {"ref": "b.yaml"}})
    _write(tmp_path, "b.yaml", {"y": {"ref": "a.yaml"}})
    with pytest.raises(ConfigError, match="cycle"):
        resolve_config(tmp_path / "a.yaml", configs_root=tmp_path)


@pytest.mark.parametrize(
    "optimizer_path",
    sorted((Path(__file__).resolve().parents[2] / "configs" / "optimizers").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_optimizer_ref_replaces_inherited_component(
    tmp_path: Path, optimizer_path: Path
) -> None:
    configs = optimizer_path.parents[1]
    experiment = _write(
        tmp_path,
        "experiment.yaml",
        {
            "ref": "experiments/cem_fixed_budget.base.yaml",
            "optimizer": {
                "ref": f"optimizers/{optimizer_path.name}",
                "num_candidates": 80,
                "num_iterations": 3,
            },
        },
    )
    resolved = resolve_config(experiment, configs)
    expected = resolve_config(optimizer_path, configs)
    expected.update(num_candidates=80, num_iterations=3)
    assert resolved["optimizer"] == expected
    assert resolved["planning"]["horizon_steps"] == 40


def test_plain_override_preserves_inherited_component(tmp_path: Path) -> None:
    _write(tmp_path, "component.yaml", {"weights": {"a": 1, "b": 2}, "scale": 3})
    _write(tmp_path, "base.yaml", {"component": {"ref": "component.yaml"}})
    experiment = _write(
        tmp_path, "experiment.yaml",
        {"ref": "base.yaml", "component": {"weights": {"b": 9}}},
    )
    assert resolve_config(experiment, tmp_path)["component"] == {
        "weights": {"a": 1, "b": 9}, "scale": 3,
    }


def test_nested_ref_replaces_inherited_component(tmp_path: Path) -> None:
    _write(tmp_path, "old.yaml", {"old_parameter": 1})
    _write(tmp_path, "new.yaml", {"weights": {"a": 2, "b": 3}})
    _write(tmp_path, "base.yaml", {"outer": {"inner": {"ref": "old.yaml"}, "seed": 7}})
    experiment = _write(
        tmp_path, "experiment.yaml",
        {"ref": "base.yaml", "outer": {"inner": {"ref": "new.yaml", "weights": {"b": 9}}}},
    )
    assert resolve_config(experiment, tmp_path)["outer"] == {
        "inner": {"weights": {"a": 2, "b": 9}}, "seed": 7,
    }


def test_dataclass_from_dict_rejects_unknown_top_level_field() -> None:
    data = {"name": "e", "inner": {"sigma": 0.2}, "extra": True}

    with pytest.raises(ConfigError, match=r"Outer.*extra"):
        dataclass_from_dict(Outer, data)


def test_dataclass_from_dict_rejects_unknown_nested_field_with_path() -> None:
    data = {"name": "e", "inner": {"sigma": 0.2, "unknown": 1}}

    with pytest.raises(ConfigError, match=r"Outer\.inner.*unknown"):
        dataclass_from_dict(Outer, data)


def test_dataclass_from_dict_rejects_non_mapping_nested_field_with_path() -> None:
    data = {"name": "e", "inner": [0.2]}

    with pytest.raises(ConfigError, match=r"Outer\.inner must be a mapping"):
        dataclass_from_dict(Outer, data)


def test_dataclass_from_dict_missing_required_raises() -> None:
    with pytest.raises(ConfigError, match="inner"):
        dataclass_from_dict(Outer, {"name": "e"})


def test_write_resolved_experiment_round_trip(tmp_path: Path) -> None:
    resolved = {"optimizer": {"noise": 0.5}, "seed": 7}
    path = write_resolved_experiment(resolved, tmp_path)
    assert path.name == RESOLVED_EXPERIMENT_FILE
    assert yaml.safe_load(path.read_text()) == resolved
