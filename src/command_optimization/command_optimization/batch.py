"""Run and aggregate YAML-declared motion and bfm experiment batches."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bfm_adapters import instance_config_from_mapping
from sampling_optimizers import ConvergenceCriteria
from command_optimization.artifacts import default_repo_root, sha256_file
from command_optimization.provenance import SourceIdentity, source_identity
from command_optimization.config import (
    RESOLVED_EXPERIMENT_FILE,
    ConfigError,
    dataclass_from_dict,
    resolve_config,
)


RESOLVED_EXPERIMENT_BATCH_FILE = "resolved_experiment_batch.yaml"
EXPERIMENT_BATCH_SUMMARY_FILE = "experiment_batch_summary.json"
_EXPERIMENT_FILES = (
    RESOLVED_EXPERIMENT_FILE,
    "report.json",
    "planning.json",
    "randomization.json",
    "provenance.json",
    "paired_traces.npz",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
from command_optimization.schema import LIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class BudgetPoint:
    id: str
    num_candidates: int
    num_iterations: int | str

    def __post_init__(self) -> None:
        _require_id(self.id, "budget id")
        _require_int(self.num_candidates, "num_candidates", minimum=1)
        if self.num_iterations != "inf":
            _require_int(self.num_iterations, "num_iterations", minimum=1)


@dataclass(frozen=True)
class OptimizerVariant:
    id: str
    ref: str
    overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_id(self.id, "optimizer variant id")
        if not isinstance(self.ref, str) or not self.ref:
            raise ConfigError("optimizer variant ref must be a non-empty string")
        if not isinstance(self.overrides, Mapping):
            raise ConfigError("optimizer variant overrides must be a mapping")
        forbidden = {"algorithm", "num_candidates", "num_iterations"}.intersection(
            self.overrides
        )
        if forbidden:
            raise ConfigError(
                "optimizer variant overrides cannot set fields owned by ref/budget: "
                f"{sorted(forbidden)}"
            )
        object.__setattr__(self, "overrides", dict(self.overrides))


@dataclass(frozen=True)
class MotionReference:
    id: str
    reference: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_id(self.id, "motion id")
        if not isinstance(self.reference, Mapping):
            raise ConfigError("motion reference must be a mapping")
        object.__setattr__(self, "reference", dict(self.reference))


@dataclass(frozen=True)
class ExperimentBatchConfig:
    schema_version: int
    name: str
    experiment_refs: Sequence[str]
    budgets: Sequence[BudgetPoint | Mapping[str, Any]]
    optimizer_variants: Sequence[OptimizerVariant | Mapping[str, Any]]
    seeds: Sequence[int]
    rollout_batch_ceiling: int
    clip_duration_seconds: float | None = None
    motions: Sequence[MotionReference | Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != LIVE_SCHEMA_VERSION
        ):
            raise ConfigError(
                f"schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError("experiment batch name must be a non-empty string")
        _require_id(self.name, "experiment batch name")
        refs = _sequence(self.experiment_refs, "experiment_refs")
        budgets = _bind_sequence(BudgetPoint, self.budgets, "budgets")
        variants = _bind_sequence(
            OptimizerVariant, self.optimizer_variants, "optimizer_variants"
        )
        seeds = _sequence(self.seeds, "seeds")
        if not refs or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ConfigError("experiment_refs must contain non-empty strings")
        if len(set(refs)) != len(refs):
            raise ConfigError("experiment_refs must be unique")
        _unique_ids(budgets, "budgets")
        _unique_ids(variants, "optimizer_variants")
        _unique_nonnegative_ints(seeds, "seeds", allow_zero=True)
        motion_rows = _sequence(self.motions, "motions")
        motions = (
            _bind_sequence(MotionReference, motion_rows, "motions") if motion_rows else ()
        )
        if motions:
            _unique_ids(motions, "motions")
        seconds = self.clip_duration_seconds
        if seconds is not None:
            if isinstance(seconds, bool):
                raise ConfigError("clip_duration_seconds must be finite and > 0")
            try:
                seconds = float(seconds)
            except (TypeError, ValueError) as exc:
                raise ConfigError("clip_duration_seconds must be finite and > 0") from exc
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ConfigError("clip_duration_seconds must be finite and > 0")
        _require_int(
            self.rollout_batch_ceiling,
            "rollout_batch_ceiling",
            minimum=1,
        )
        object.__setattr__(self, "experiment_refs", refs)
        object.__setattr__(self, "clip_duration_seconds", seconds)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "optimizer_variants", variants)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "motions", motions)


@dataclass(frozen=True)
class ExperimentBatchCell:
    cell_id: str
    motion_id: str
    bfm_id: str
    optimizer_variant_id: str
    optimizer_algorithm: str
    budget_id: str
    candidate_count: int
    iteration_count: int | str
    seed: int
    clip_duration_seconds: float | None
    rollout_batch_size: int
    candidate_latent_bytes_per_iteration: int
    resolved: dict[str, Any]


@dataclass(frozen=True)
class ExperimentBatchResult:
    output_dir: Path
    success: bool
    cells: int


ExperimentExecutor = Callable[..., Any]


def load_experiment_batch(
    batch_path: str | Path, *, configs_root: str | Path | None = None
) -> tuple[dict[str, Any], ExperimentBatchConfig, tuple[ExperimentBatchCell, ...]]:
    """Load one strict Experiment Batch YAML and expand its cells."""

    repo = default_repo_root()
    root = (repo / "configs" if configs_root is None else Path(configs_root)).resolve()
    path = Path(batch_path)
    try:
        resolved = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"experiment batch config file not found: {path}") from exc
    if not isinstance(resolved, dict):
        raise ConfigError(
            f"top-level experiment batch config must be a mapping: {path}"
        )
    config = dataclass_from_dict(ExperimentBatchConfig, resolved)
    cells = _expand_cells(config, configs_root=root, batch_dir=path.resolve().parent)
    return resolved, config, cells


def run_experiment_batch(
    batch_path: str | Path,
    *,
    execute_experiment: ExperimentExecutor,
    device_index: int = 0,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    configs_root: str | Path | None = None,
) -> ExperimentBatchResult | None:
    """Preflight and execute a motion × bfm × optimizer × budget matrix."""

    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise ValueError("device_index must be a non-negative integer")
    if not callable(execute_experiment):
        raise TypeError("execute_experiment must be callable")
    batch_path = Path(batch_path)
    resolved, config, cells = load_experiment_batch(batch_path, configs_root=configs_root)
    repo = default_repo_root()
    root = (repo / "configs" if configs_root is None else Path(configs_root)).resolve()

    _preflight(cells, execute_experiment, device_index=device_index, configs_root=root)
    if dry_run:
        return None

    # Reuse output handling while keeping imports independent at module load.
    from command_optimization.run import _default_output_dir, _prepare_output_dir

    source = source_identity(repo)
    destination = (
        _default_output_dir(repo, f"experiment_batches/{config.name}", source.source_sha256)
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    _prepare_output_dir(destination)
    _write_yaml(destination / RESOLVED_EXPERIMENT_BATCH_FILE, resolved)
    resolved_dir = destination / "resolved_experiments"
    resolved_dir.mkdir()
    cells_dir = destination / "cells"
    cells_dir.mkdir()

    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_config = resolved_dir / f"{cell.cell_id}.yaml"
        cell_output = cells_dir / cell.cell_id
        _write_yaml(cell_config, cell.resolved)
        try:
            execute_experiment(
                cell_config,
                device_index=device_index,
                output_dir=cell_output,
                dry_run=False,
                configs_root=root,
            )
            row = _summarize_cell(
                cell,
                cell_output,
                batch_root=destination,
                source=source,
            )
        except Exception as exc:
            cell_output.mkdir(parents=True, exist_ok=True)
            failure = f"{type(exc).__name__}: {exc}"
            failure_path = cell_output / "failure.json"
            if not failure_path.exists():
                _write_json(failure_path, {"cell_id": cell.cell_id, "failure": failure})
            row = _failed_row(cell, cell_output, destination, failure)
        rows.append(row)

    successful_digests = {
        row["initial_physical_state_digest"]
        for row in rows
        if row["status"] == "success"
    }
    invariant_failures: list[str] = []
    motion_digests: dict[str, str | None] = {}
    for motion_id in dict.fromkeys(cell.motion_id for cell in cells):
        digests = {
            row["initial_physical_state_digest"]
            for row in rows
            if row["status"] == "success" and row["motion_id"] == motion_id
        }
        motion_digests[motion_id] = next(iter(digests)) if len(digests) == 1 else None
        if len(digests) > 1:
            invariant_failures.append(
                f"motion {motion_id!r} has inconsistent initial physical-state digests"
            )
        reference_digests = {
            row["reference_digest"]
            for row in rows
            if row["status"] == "success" and row["motion_id"] == motion_id
        }
        if len(reference_digests) > 1:
            invariant_failures.append(
                f"motion {motion_id!r} has inconsistent reference digests"
            )
    successful_hardware = {
        row["hardware"]["gpu_uuid"]
        for row in rows
        if row["status"] == "success"
    }
    if len(successful_hardware) != 1:
        invariant_failures.append("successful cells do not share one physical GPU")
    success = all(row["status"] == "success" for row in rows) and not invariant_failures
    summary = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "experiment_batch": {
            "name": config.name,
            "release_version": source.release_version,
            "source_sha256": source.source_sha256,
            "experiment_batch_yaml_sha256": sha256_file(batch_path),
            "resolved_experiment_batch_sha256": sha256_file(
                destination / RESOLVED_EXPERIMENT_BATCH_FILE
            ),
            "experiment_refs": list(config.experiment_refs),
            "motions": [
                {"id": motion.id, "reference": dict(motion.reference)}
                for motion in config.motions
            ],
            "clip_duration_seconds": config.clip_duration_seconds,
            "budgets": [
                {
                    "id": budget.id,
                    "num_candidates": budget.num_candidates,
                    "num_iterations": budget.num_iterations,
                }
                for budget in config.budgets
            ],
            "optimizer_variants": [
                {
                    "id": variant.id,
                    "ref": variant.ref,
                    "overrides": dict(variant.overrides),
                }
                for variant in config.optimizer_variants
            ],
            "seeds": list(config.seeds),
            "rollout_batch_ceiling": config.rollout_batch_ceiling,
            "initial_physical_state_digest": (
                next(iter(successful_digests)) if len(successful_digests) == 1 else None
            ),
            "initial_physical_state_digests": motion_digests,
            "invariant_failures": invariant_failures,
            "success": success,
        },
        "cells": rows,
        "groups": _group_rows(rows),
    }
    _write_json(destination / EXPERIMENT_BATCH_SUMMARY_FILE, summary)
    return ExperimentBatchResult(destination, success, len(rows))


def _expand_cells(
    batch_config: ExperimentBatchConfig,
    *,
    configs_root: Path,
    batch_dir: Path | None = None,
) -> tuple[ExperimentBatchCell, ...]:
    from command_optimization.run import ReferenceConfig, load_experiment

    user_dir = batch_dir or configs_root
    optimizers: list[tuple[OptimizerVariant, dict[str, Any]]] = []
    for variant in batch_config.optimizer_variants:
        path = _safe_optimizer_ref(variant.ref, configs_root, user_dir=user_dir)
        optimizer = resolve_config(path, configs_root)
        if not isinstance(optimizer, dict):
            raise ConfigError(f"optimizer ref must resolve to a mapping: {variant.ref!r}")
        optimizer = copy.deepcopy(optimizer)
        optimizer.update(variant.overrides)
        algorithm = optimizer.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm:
            raise ConfigError(
                f"optimizer variant {variant.id!r} has no non-empty algorithm"
            )
        optimizers.append((variant, optimizer))

    experiments: list[tuple[str, dict[str, Any], Any, Any]] = []
    common: dict[str, Any] | None = None
    common_control_period: float | None = None
    common_joints: tuple[str, ...] | None = None
    motion_bfms: set[tuple[str, str]] = set()
    for ref in batch_config.experiment_refs:
        path = _safe_experiment_ref(ref, configs_root, user_dir=user_dir)
        resolved, config = load_experiment(path, configs_root=configs_root)
        bfm = instance_config_from_mapping(config.bfm)
        comparable = _common_experiment_fields(resolved)
        if common is None:
            common = comparable
            common_control_period = bfm.control_period
            common_joints = bfm.command_joint_names
        elif comparable != common:
            raise ConfigError(
                f"experiment {ref!r} differs outside fields owned by the "
                "Experiment Batch"
            )
        if bfm.control_period != common_control_period:
            raise ConfigError(
                "experiment batch bfms must share one control period"
            )
        if bfm.command_joint_names != common_joints:
            raise ConfigError(
                "experiment batch bfms must share canonical command joints"
            )
        targets = (
            [(motion.id, dict(motion.reference)) for motion in batch_config.motions]
            if batch_config.motions
            else [(None, dict(resolved["reference"]))]
        )
        for motion_id, reference in targets:
            if batch_config.clip_duration_seconds is not None:
                reference["clip_duration_seconds"] = batch_config.clip_duration_seconds
            if batch_config.motions and reference.get("dataset_root") is not None:
                dataset_root = Path(reference["dataset_root"]).expanduser()
                reference["dataset_root"] = str(
                    ((batch_dir or Path.cwd()) / dataset_root).resolve()
                )
            dataclass_from_dict(ReferenceConfig, reference)
            if motion_id is None:
                digest = hashlib.sha256(
                    json.dumps(reference, sort_keys=True).encode()
                ).hexdigest()[:12]
                label = re.sub(
                    r"[^a-z0-9_-]+", "-", reference["motion_id"].lower()
                ).strip("-_")
                motion_id = f"{label[:48] or 'motion'}-{digest}"
            identity = (motion_id, bfm.instance)
            if identity in motion_bfms:
                raise ConfigError(
                    f"duplicate motion/bfm pair in experiment batch: {identity}"
                )
            motion_bfms.add(identity)
            value = copy.deepcopy(resolved)
            value["reference"] = reference
            experiments.append((motion_id, value, config, bfm))

    cells: list[ExperimentBatchCell] = []
    seen: set[str] = set()
    for motion_id, resolved, config, bfm in experiments:
        for variant, optimizer in optimizers:
            algorithm = str(optimizer["algorithm"])
            for budget in batch_config.budgets:
                for seed in batch_config.seeds:
                    cell_id = (
                        f"{motion_id}-{bfm.instance}-{variant.id}-{budget.id}-s{seed}"
                    )
                    if cell_id in seen:
                        raise ConfigError(
                            f"duplicate experiment batch cell id: {cell_id}"
                        )
                    seen.add(cell_id)
                    value = copy.deepcopy(resolved)
                    value["name"] = f"{batch_config.name}-{cell_id}"
                    value["optimizer"] = copy.deepcopy(optimizer)
                    value["optimizer"]["num_candidates"] = budget.num_candidates
                    value["optimizer"]["num_iterations"] = budget.num_iterations
                    if budget.num_iterations == "inf":
                        _batch_convergence(value)
                    elif "convergence" in value["optimizer"]:
                        raise ConfigError("optimizer.convergence is valid only with num_iterations: 'inf'")
                    value["planning"]["seed"] = seed
                    scenario_count = (
                        config.robustness.num_scenarios
                        if config.robustness.enabled
                        else 1
                    )
                    candidate_capacity = (
                        batch_config.rollout_batch_ceiling // scenario_count
                    )
                    if candidate_capacity < 1:
                        raise ConfigError(
                            "rollout_batch_ceiling must fit one complete "
                            "scenario group"
                        )
                    rollout_batch_size = min(
                        budget.num_candidates, candidate_capacity
                    ) * scenario_count
                    value["planning"]["max_worlds"] = rollout_batch_size
                    latent_dim = 6 + len(bfm.command_joint_names)
                    cells.append(
                        ExperimentBatchCell(
                            cell_id=cell_id,
                            motion_id=motion_id,
                            bfm_id=bfm.instance,
                            optimizer_variant_id=variant.id,
                            optimizer_algorithm=algorithm,
                            budget_id=budget.id,
                            candidate_count=budget.num_candidates,
                            iteration_count=budget.num_iterations,
                            seed=seed,
                            clip_duration_seconds=(
                                value["reference"].get("clip_duration_seconds")
                            ),
                            rollout_batch_size=rollout_batch_size,
                            candidate_latent_bytes_per_iteration=(
                                budget.num_candidates
                                * config.planning.num_knots
                                * latent_dim
                                * 8
                            ),
                            resolved=value,
                        )
                    )
    return tuple(cells)


def _safe_experiment_ref(
    ref: str, configs_root: Path, *, user_dir: Path | None = None
) -> Path:
    if user_dir is not None:
        return _user_ref(ref, user_dir, configs_root)
    relative = Path(ref)
    if relative.is_absolute() or relative.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"unsafe experiment ref: {ref!r}")
    candidate = (configs_root / relative).resolve()
    if not candidate.is_relative_to(configs_root) or not candidate.is_file():
        raise ConfigError(f"unsafe or missing experiment ref: {ref!r}")
    return candidate


def _safe_optimizer_ref(
    ref: str, configs_root: Path, *, user_dir: Path | None = None
) -> Path:
    if user_dir is not None:
        return _user_ref(ref, user_dir, configs_root)
    relative = Path(ref)
    optimizers_root = (configs_root / "optimizers").resolve()
    if relative.is_absolute() or relative.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"unsafe optimizer ref: {ref!r}")
    candidate = (configs_root / relative).resolve()
    if not candidate.is_relative_to(optimizers_root) or not candidate.is_file():
        raise ConfigError(f"unsafe or missing optimizer ref: {ref!r}")
    return candidate


def _user_ref(ref: str, user_dir: Path, configs_root: Path) -> Path:
    path = Path(ref).expanduser()
    if path.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"config ref must be a YAML file: {ref!r}")
    for candidate in ((user_dir / path).resolve(), (configs_root / path).resolve()):
        if candidate.is_file():
            return candidate
    raise ConfigError(f"config ref not found: {ref!r}")


def _common_experiment_fields(resolved: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(resolved))
    value.pop("name", None)
    value.pop("bfm", None)
    value.pop("optimizer", None)
    planning = value.get("planning")
    reference = value.get("reference")
    if not isinstance(planning, dict) or not isinstance(reference, dict):
        raise ConfigError(
            "Experiment Batch inputs require planning and reference mappings"
        )
    value.pop("reference")
    planning.pop("seed", None)
    planning.pop("max_worlds", None)
    return value


def _preflight(
    cells: Sequence[ExperimentBatchCell],
    executor: ExperimentExecutor,
    *,
    device_index: int,
    configs_root: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="bfm-ttc-experiment-batch-preflight-"
    ) as temp:
        root = Path(temp)
        for cell in cells:
            path = root / f"{cell.cell_id}.yaml"
            _write_yaml(path, cell.resolved)
            executor(
                path,
                device_index=device_index,
                output_dir=None,
                dry_run=True,
                configs_root=configs_root,
            )


def _batch_convergence(resolved: Mapping[str, Any]) -> ConvergenceCriteria:
    optimizer = resolved["optimizer"]
    if optimizer.get("algorithm") != "cem":
        raise ConfigError("convergence batches support only cem")
    if resolved["planning"].get("evaluate_proposal") is not False:
        raise ConfigError("convergence batches require evaluate_proposal: false")
    criteria = optimizer.get("convergence")
    if not isinstance(criteria, Mapping):
        raise ConfigError("convergence batches require explicit convergence criteria")
    return dataclass_from_dict(ConvergenceCriteria, dict(criteria))


def _converged_iteration_count(record: Mapping[str, Any], criteria: ConvergenceCriteria) -> int:
    """Validate termination from the observed iteration minima, not a guessed I."""
    rows = record.get("cost_quantiles")
    if not isinstance(rows, list) or not rows:
        raise ValueError("convergence ledger must contain iteration cost quantiles")
    best = math.inf
    stalled = 0
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError("convergence cost quantile rows are invalid")
        value = float(row[0])
        if not math.isfinite(value):
            raise ValueError("convergence iteration minimum must be finite")
        current = min(best, value)
        if math.isfinite(best):
            stalled = 0 if best - current > criteria.min_cost_improvement else stalled + 1
        best = current
        if stalled >= criteria.patience and index != len(rows) - 1:
            raise ValueError("convergence ledger continues after its stopping condition")
    if stalled < criteria.patience:
        raise ValueError("convergence ledger does not reach its stopping condition")
    return len(rows)


def _summarize_cell(
    cell: ExperimentBatchCell,
    output: Path,
    *,
    batch_root: Path,
    source: SourceIdentity,
) -> dict[str, Any]:
    paths = {name: output / name for name in _EXPERIMENT_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"cell output is missing required files: {missing}")
    report = _read_json(paths["report.json"])
    planning_payload = _read_json(paths["planning.json"])
    randomization = _read_json(paths["randomization.json"])
    provenance = _read_json(paths["provenance.json"])
    resolved_experiment = yaml.safe_load(
        paths[RESOLVED_EXPERIMENT_FILE].read_text()
    )
    if resolved_experiment != cell.resolved:
        raise ValueError(
            "persisted resolved experiment differs from the experiment batch cell"
        )
    if (
        not isinstance(report, dict)
        or not isinstance(planning_payload, dict)
        or not isinstance(randomization, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError(
            "cell report/planning/randomization/provenance schema is invalid"
        )
    for name, payload in (
        ("report.json", report),
        ("planning.json", planning_payload),
        ("randomization.json", randomization),
        ("provenance.json", provenance),
    ):
        if (
            isinstance(payload.get("schema_version"), bool)
            or not isinstance(payload.get("schema_version"), int)
            or payload.get("schema_version") != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"cell {name} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
    with np.load(paths["paired_traces.npz"], allow_pickle=False) as traces:
        trace_version = np.asarray(traces["schema_version"])
        if (
            trace_version.shape != ()
            or not np.issubdtype(trace_version.dtype, np.integer)
            or int(trace_version) != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                "cell paired_traces.npz schema_version must be exactly "
                f"{LIVE_SCHEMA_VERSION}"
            )
    planning = planning_payload.get("updates")
    if not isinstance(planning, list):
        raise ValueError("planning.json updates must be a list")
    if (
        provenance.get("release_version") != source.release_version
        or provenance.get("source_sha256") != source.source_sha256
    ):
        raise ValueError(
            "cell provenance differs from the Experiment Batch source state"
        )
    hardware = provenance.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("available") or not hardware.get(
        "gpu_uuid"
    ):
        raise ValueError("cell hardware provenance is incomplete")
    bfm_runtime = provenance.get("bfm_runtime")
    if bfm_runtime not in ("device", "host-reference"):
        raise ValueError("cell bfm runtime provenance is invalid")
    file_digests = {name: sha256_file(path) for name, path in paths.items()}
    for name, key in (
        ("report.json", "report_sha256"),
        ("planning.json", "planning_sha256"),
        ("randomization.json", "randomization_sha256"),
        (RESOLVED_EXPERIMENT_FILE, "resolved_experiment_sha256"),
        ("paired_traces.npz", "trace_sha256"),
    ):
        if provenance.get(key) != file_digests[name]:
            raise ValueError(f"cell provenance {key} does not match {name}")
    randomization_updates = randomization.get("updates")
    if not isinstance(randomization_updates, list) or len(
        randomization_updates
    ) != len(planning):
        raise ValueError("randomization updates disagree with planning ledger")
    for randomization_update, planning_update in zip(
        randomization_updates, planning, strict=True
    ):
        if not isinstance(randomization_update, dict) or (
            randomization_update.get("update_step")
            != planning_update.get("update_step")
            or randomization_update.get("scenario_digest")
            != planning_update.get("scenario_digest")
        ):
            raise ValueError("randomization identity disagrees with planning ledger")

    identities = report.get("identities")
    baseline = report.get("baseline")
    optimized = report.get("optimized")
    if not all(isinstance(value, dict) for value in (identities, baseline, optimized)):
        raise ValueError("cell report identities/metrics are invalid")
    assert isinstance(identities, dict)
    assert isinstance(baseline, dict)
    assert isinstance(optimized, dict)
    if identities.get("bfm_id") != cell.bfm_id:
        raise ValueError("cell bfm identity differs from Experiment Batch expansion")
    if provenance.get("reference_sha256") != identities.get("reference_digest"):
        raise ValueError("reference digest differs between report and provenance")
    execution_failures: list[str] = []
    expected_execution_steps = (
        round(cell.clip_duration_seconds / float(identities["control_period"]))
        if cell.clip_duration_seconds is not None
        else None
    )
    for label, metrics in (("baseline", baseline), ("optimized", optimized)):
        if not metrics.get("success") or metrics.get("valid_steps") != metrics.get(
            "total_steps"
        ):
            execution_failures.append(
                f"{label} execution did not complete all valid steps"
            )
        elif (
            expected_execution_steps is not None
            and int(metrics["total_steps"]) != expected_execution_steps
        ):
            execution_failures.append(
                f"{label} execution did not contain the declared clip steps"
            )
    if baseline.get("total_steps") != optimized.get("total_steps") or int(
        baseline["total_steps"]
    ) < 1:
        execution_failures.append("paired executions must have the same non-empty duration")

    if not planning:
        raise ValueError("cell planning ledger must not be empty")
    iterations = cell.iteration_count
    convergence = _batch_convergence(cell.resolved) if iterations == "inf" else None
    if cell.resolved["optimizer"]["num_iterations"] != iterations:
        raise ValueError("resolved optimizer iterations differ from Experiment Batch budget")
    evaluate_proposal = bool(cell.resolved["planning"]["evaluate_proposal"])
    all_penalty = 0
    for record in planning:
        if not isinstance(record, dict):
            raise ValueError("planning entries must be mappings")
        selected = float(record["selected_cost"])
        if not math.isfinite(selected):
            raise ValueError("planning selected cost must be finite")
        actual_iterations = (
            _converged_iteration_count(record, convergence)
            if convergence is not None else iterations
        )
        base_candidates = cell.candidate_count * actual_iterations
        candidate_value = record.get("candidate_evaluations")
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, int):
            raise ValueError("planning candidate ledger must be an integer")
        candidate_evaluations = candidate_value
        extra_evaluation = candidate_evaluations - base_candidates
        if extra_evaluation not in (0, 1):
            raise ValueError(
                "planning logical rollout ledger disagrees with Experiment Batch budget"
            )
        if not evaluate_proposal and extra_evaluation:
            raise ValueError("planning ledger contains a disabled proposal evaluation")
        if (
            record.get("proposal_candidate_id")
            == record.get("best_observed_candidate_id")
            and extra_evaluation
        ):
            raise ValueError("best-observed proposal must not be evaluated twice")
        scenario_count = record.get("scenario_count")
        logical_rollouts = record.get("logical_rollouts")
        if (
            isinstance(scenario_count, bool)
            or not isinstance(scenario_count, int)
            or scenario_count < 1
        ):
            raise ValueError("planning scenario_count ledger must be a positive integer")
        scenario_digest = record.get("scenario_digest")
        if scenario_digest is not None and (
            not isinstance(scenario_digest, str)
            or not scenario_digest.startswith("sha256:")
        ):
            raise ValueError("planning scenario_digest ledger is invalid")
        if isinstance(logical_rollouts, bool) or not isinstance(logical_rollouts, int):
            raise ValueError("planning logical rollout ledger must be an integer")
        if logical_rollouts != candidate_evaluations * scenario_count:
            raise ValueError("planning scenario rollout ledger is inconsistent")
        physical_rollouts = record.get("physical_rollouts")
        if (
            isinstance(physical_rollouts, bool)
            or not isinstance(physical_rollouts, int)
            or physical_rollouts < logical_rollouts
        ):
            raise ValueError("planning physical rollout ledger is invalid")
        penalized_candidates = record.get("penalized_candidate_evaluations")
        if (
            isinstance(penalized_candidates, bool)
            or not isinstance(penalized_candidates, int)
            or not 0 <= penalized_candidates <= candidate_evaluations
        ):
            raise ValueError("planning penalized candidate ledger is invalid")
        invalid_scenarios = record.get("invalid_scenario_rollouts")
        if (
            isinstance(invalid_scenarios, bool)
            or not isinstance(invalid_scenarios, int)
            or not 0 <= invalid_scenarios <= logical_rollouts
        ):
            raise ValueError("planning invalid scenario ledger is invalid")
        rows = record.get("cost_quantiles")
        expected_evaluations = actual_iterations + extra_evaluation
        if not isinstance(rows, list) or len(rows) != expected_evaluations:
            raise ValueError("planning cost_quantiles disagree with evaluator calls")
        if any(not isinstance(row, list) or len(row) != 5 for row in rows):
            raise ValueError("planning cost quantile rows are invalid")
        if penalized_candidates == candidate_evaluations:
            all_penalty += 1
    if all_penalty:
        execution_failures.append(
            f"cell contains {all_penalty} all-penalty planning updates"
        )

    deformations = _mean_deformation(planning)
    optimized_minus_baseline = {
        key: float(optimized[key]) - float(baseline[key])
        for key in _metric_keys(baseline, optimized)
    }
    candidate_evaluations = sum(
        int(record["candidate_evaluations"]) for record in planning
    )
    logical_rollouts = sum(int(record["logical_rollouts"]) for record in planning)
    physical_rollouts = sum(int(record["physical_rollouts"]) for record in planning)
    penalized_candidate_evaluations = sum(
        int(record["penalized_candidate_evaluations"]) for record in planning
    )
    invalid_scenario_rollouts = sum(
        int(record["invalid_scenario_rollouts"]) for record in planning
    )
    horizon_steps = int(cell.resolved["planning"]["horizon_steps"])
    return {
        **_cell_summary_fields(cell, output, batch_root),
        "status": "failed" if execution_failures else "success",
        "file_sha256": file_digests,
        "hardware": hardware,
        "bfm_runtime": bfm_runtime,
        "initial_physical_state_digest": identities[
            "initial_physical_state_digest"
        ],
        "reference_id": identities["reference_id"],
        "reference_digest": identities["reference_digest"],
        "checkpoint_digest": identities["checkpoint_digest"],
        "robot_model_digest": identities["robot_model_digest"],
        "control_period": identities["control_period"],
        "baseline": baseline,
        "optimized": optimized,
        "optimized_minus_baseline": optimized_minus_baseline,
        "mean_candidate_deformation": deformations,
        "candidate_evaluations": candidate_evaluations,
        "logical_rollouts": logical_rollouts,
        "physical_rollouts": physical_rollouts,
        "penalized_candidate_evaluations": penalized_candidate_evaluations,
        "invalid_scenario_rollouts": invalid_scenario_rollouts,
        "logical_rollout_steps": logical_rollouts * horizon_steps,
        "physical_rollout_steps": physical_rollouts * horizon_steps,
        "planning_wall_seconds": sum(
            float(record["update_wall_seconds"]) for record in planning
        ),
        "all_penalty_updates": all_penalty,
        "failure": "; ".join(execution_failures) if execution_failures else None,
    }


def _failed_row(
    cell: ExperimentBatchCell, output: Path, batch_root: Path, failure: str
) -> dict[str, Any]:
    return {
        **_cell_summary_fields(cell, output, batch_root),
        "status": "failed",
        "file_sha256": {
            name: sha256_file(output / name) if (output / name).is_file() else None
            for name in _EXPERIMENT_FILES
        },
        "initial_physical_state_digest": None,
        "hardware": None,
        "bfm_runtime": None,
        "reference_id": None,
        "reference_digest": None,
        "checkpoint_digest": None,
        "robot_model_digest": None,
        "control_period": None,
        "baseline": None,
        "optimized": None,
        "optimized_minus_baseline": None,
        "mean_candidate_deformation": None,
        "candidate_evaluations": None,
        "logical_rollouts": None,
        "physical_rollouts": None,
        "penalized_candidate_evaluations": None,
        "invalid_scenario_rollouts": None,
        "logical_rollout_steps": None,
        "physical_rollout_steps": None,
        "planning_wall_seconds": None,
        "all_penalty_updates": None,
        "failure": failure,
    }


def _cell_summary_fields(
    cell: ExperimentBatchCell, output: Path, batch_root: Path
) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "motion_id": cell.motion_id,
        "bfm_id": cell.bfm_id,
        "optimizer_variant_id": cell.optimizer_variant_id,
        "optimizer_algorithm": cell.optimizer_algorithm,
        "budget_id": cell.budget_id,
        "candidate_count": cell.candidate_count,
        "iteration_count": cell.iteration_count,
        "seed": cell.seed,
        "clip_duration_seconds": cell.clip_duration_seconds,
        "rollout_batch_size": cell.rollout_batch_size,
        "output": str(output.relative_to(batch_root)),
        "candidate_latent_bytes_per_iteration": (
            cell.candidate_latent_bytes_per_iteration
        ),
    }


def _mean_deformation(planning: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = sorted(
        {
            name
            for record in planning
            for name in record.get("candidate_deformation", {})
        }
    )
    if not names:
        raise ValueError("planning ledger must contain candidate deformation")
    return {
        name: statistics.fmean(
            float(record["candidate_deformation"][name]) for record in planning
        )
        for name in names
    }


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field} must be a sequence")
    return tuple(value)


def _metric_keys(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> tuple[str, ...]:
    excluded = {"total_steps", "valid_steps", "failure_step"}
    return tuple(
        key
        for key in sorted(set(baseline).intersection(optimized).difference(excluded))
        if isinstance(baseline[key], (int, float))
        and not isinstance(baseline[key], bool)
        and isinstance(optimized[key], (int, float))
        and not isinstance(optimized[key], bool)
    )


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = (
            str(row["motion_id"]),
            str(row["bfm_id"]),
            str(row["optimizer_variant_id"]),
            str(row["budget_id"]),
        )
        if key not in keys:
            keys.append(key)
    groups: list[dict[str, Any]] = []
    for motion_id, bfm_id, optimizer_variant_id, budget_id in keys:
        members = [
            row
            for row in rows
            if row["motion_id"] == motion_id
            and row["bfm_id"] == bfm_id
            and row["optimizer_variant_id"] == optimizer_variant_id
            and row["budget_id"] == budget_id
        ]
        successes = [row for row in members if row["status"] == "success"]
        scalar_names = sorted(
            {name for row in successes for name in _group_scalars(row)}
        )
        statistics_by_name = {}
        for name in scalar_names:
            values = [_group_scalars(row)[name] for row in successes]
            statistics_by_name[name] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        groups.append(
            {
                "motion_id": motion_id,
                "bfm_id": bfm_id,
                "optimizer_variant_id": optimizer_variant_id,
                "optimizer_algorithm": members[0]["optimizer_algorithm"],
                "budget_id": budget_id,
                "candidate_count": members[0]["candidate_count"],
                "iteration_count": members[0]["iteration_count"],
                "n": len(members),
                "successes": len(successes),
                "statistics": statistics_by_name,
            }
        )
    return groups


def _group_scalars(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in ("baseline", "optimized", "optimized_minus_baseline"):
        for name, value in row[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name not in ("total_steps", "valid_steps", "failure_step"):
                    values[f"{section}.{name}"] = float(value)
    for name, value in row["mean_candidate_deformation"].items():
        values[f"deformation.{name}"] = float(value)
    for name in (
        "candidate_evaluations",
        "logical_rollouts",
        "physical_rollouts",
        "penalized_candidate_evaluations",
        "invalid_scenario_rollouts",
        "logical_rollout_steps",
        "physical_rollout_steps",
        "candidate_latent_bytes_per_iteration",
        "planning_wall_seconds",
    ):
        values[f"compute.{name}"] = float(row[name])
    return values


def _unique_nonnegative_ints(
    values: Sequence[int], field: str, *, allow_zero: bool
) -> None:
    if not values:
        raise ConfigError(f"{field} must not be empty")
    minimum = 0 if allow_zero else 1
    for value in values:
        _require_int(value, field, minimum=minimum)
    if len(set(values)) != len(values):
        raise ConfigError(f"{field} must contain unique values")


def _bind_sequence(
    cls: type[Any], value: Any, field: str
) -> tuple[Any, ...]:
    rows = _sequence(value, field)
    if not rows:
        raise ConfigError(f"{field} must not be empty")
    return tuple(
        row if isinstance(row, cls) else dataclass_from_dict(cls, row)
        for row in rows
    )


def _unique_ids(values: Sequence[Any], field: str) -> None:
    ids = [value.id for value in values]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{field} must contain unique ids")


def _require_id(value: Any, field: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ConfigError(
            f"{field} must match {_ID_PATTERN.pattern!r}, got {value!r}"
        )


def _require_int(value: Any, field: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} values must be integers >= {minimum}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_yaml(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(yaml.safe_dump(value, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
