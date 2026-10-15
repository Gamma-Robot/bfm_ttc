"""Run one configured command-optimization experiment and expose the package console commands."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from contextlib import ExitStack
from datetime import datetime, timezone
import inspect
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sampling_optimizers import ConvergenceCriteria, IterationLimit, SamplingOptimizer, get_optimizer
from bfm_adapters import (
    DeviceBFMProvider,
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
    PreparedDeviceBFM,
    SerialReferenceBFM,
    BFMAdapter,
    BFMInstanceConfig,
    get_family,
    instance_config_from_mapping,
)
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import (
    LOCK_RELATIVE,
    ArtifactEntry,
    default_repo_root,
    ensure_artifact,
    load_lock,
    resolve,
    sha256_file,
)
from command_optimization.config import (
    RESOLVED_EXPERIMENT_FILE,
    dataclass_from_dict,
    resolve_config,
    write_resolved_experiment,
)
from command_optimization.motion_tracking.metrics import compute_paired_metrics
from command_optimization.motion_tracking.export import OutputConfig, export_motion
from command_optimization.motion_tracking.command import take_motion_command, reference_right_context
from command_optimization.motion_tracking.device_cost import MotionTrackingCostReducer
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.motion_tracking.search_space import KnotScales, QposKnotsSpace
from command_optimization.motion_tracking.any4hdmi import (
    load_reference_motion,
    resolve_any4hdmi_motion,
)
from command_optimization.planning.evaluation import RolloutEvaluator
from command_optimization.records import (
    ExecutionRunArtifact,
    ExecutionTrace,
    PairedRunArtifact,
    PlanningDiagnostics,
)
from command_optimization.loop import run_command_optimization
from command_optimization.simulation.command_execution import CommandExecutor
from command_optimization.simulation.measurements import MeasurementPrimitive
from command_optimization.simulation.mjwarp import MjwarpWorld
from command_optimization.simulation.randomization import (
    RobustnessConfig,
    ScenarioBatch,
    sample_scenarios,
)
from command_optimization.simulation.execution_state import ExecutionState, physical_state_digest
from command_optimization.bfm_preparation import prepare_bfm_artifact
from command_optimization.provenance import SourceIdentity, source_identity
from command_optimization.visualization import (
    ComparisonView,
    RenderConfig,
    RenderBindings,
    RenderTrack,
    render_mp4,
)


TRACE_FILE = "paired_traces.npz"
PLANNING_FILE = "planning.json"
REPORT_FILE = "report.json"
PROVENANCE_FILE = "provenance.json"
RENDER_FILE = "comparison.mp4"
RENDER_PROVENANCE_FILE = "render_provenance.json"
RANDOMIZATION_FILE = "randomization.json"
from command_optimization.schema import LIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class ReferenceConfig:
    manifest_artifact: str | None = None
    motion_artifact: str | None = None
    motion_id: str = ""
    clip_duration_seconds: float | None = None
    dataset_root: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.motion_id, str) or not self.motion_id:
            raise ValueError("reference.motion_id must be a non-empty string")
        if self.dataset_root is not None:
            if not isinstance(self.dataset_root, str) or not self.dataset_root:
                raise ValueError("reference.dataset_root must be a non-empty string")
            if self.manifest_artifact is not None or self.motion_artifact is not None:
                raise ValueError("reference must use dataset_root or artifact IDs, not both")
        else:
            for field in ("manifest_artifact", "motion_artifact"):
                if not isinstance(getattr(self, field), str) or not getattr(self, field):
                    raise ValueError(f"reference.{field} must be a non-empty string")
        if self.clip_duration_seconds is not None and (
            isinstance(self.clip_duration_seconds, bool)
            or not np.isfinite(self.clip_duration_seconds)
            or self.clip_duration_seconds <= 0.0
        ):
            raise ValueError("reference.clip_duration_seconds must be finite and positive")


@dataclass(frozen=True)
class PlanningConfig:
    horizon_steps: int
    replanning_interval_steps: int
    num_knots: int
    max_worlds: int
    failure_penalty: float
    seed: int
    evaluate_proposal: bool
    root_body_name: str
    scales: KnotScales

    def __post_init__(self) -> None:
        for field, minimum in (
            ("horizon_steps", 2),
            ("replanning_interval_steps", 1),
            ("num_knots", 2),
            ("max_worlds", 1),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"planning.{field} must be an integer >= {minimum}")
        if self.replanning_interval_steps > self.horizon_steps:
            raise ValueError("planning.replanning_interval_steps cannot exceed horizon_steps")
        if not np.isfinite(self.failure_penalty) or self.failure_penalty <= 0.0:
            raise ValueError("planning.failure_penalty must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("planning.seed must be a non-negative integer")
        if not isinstance(self.evaluate_proposal, bool):
            raise TypeError("planning.evaluate_proposal must be bool")
        if not isinstance(self.root_body_name, str) or not self.root_body_name:
            raise ValueError("planning.root_body_name must be non-empty")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    bfm: dict[str, Any]
    robot: dict[str, Any]
    optimizer: dict[str, Any]
    motion_tracking: dict[str, Any]
    reference: ReferenceConfig
    planning: PlanningConfig
    visualization: RenderConfig
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)
    robustness: RobustnessConfig = dataclasses.field(
        default_factory=RobustnessConfig
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("experiment name must be non-empty")
        for field in ("bfm", "robot", "optimizer", "motion_tracking"):
            if not isinstance(getattr(self, field), dict):
                raise TypeError(f"{field} config must be a mapping")


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    pair: PairedRunArtifact


@dataclass(frozen=True)
class _TargetAssets:
    dataset_root: Path
    reference_id: str
    reference_digest: str


class _RolloutRuntimePool:
    """Own persistent execution resources for every requested batch capacity."""

    def __init__(
        self,
        *,
        world_factory: Callable[[int], MjwarpWorld],
        adapter: BFMAdapter,
        bfm_config: BFMInstanceConfig,
        policy_path: Path,
        device_index: int,
        objective: MotionTrackingObjective,
        control_period: float,
        motion_command_query_offsets: tuple[int, ...],
    ) -> None:
        self._world_factory = world_factory
        self._adapter = adapter
        self._policy_path = policy_path
        self._objective = objective
        self._control_period = control_period
        self._motion_command_query_offsets = motion_command_query_offsets
        self._uses_device_bfm = isinstance(adapter, DeviceBFMProvider)
        self._host_policy = (
            None
            if self._uses_device_bfm
            else BatchedOrtCudaPolicy(
                policy_path, bfm_config, device_id=device_index
            )
        )
        self._worlds: dict[int, MjwarpWorld] = {}
        self._bfms: dict[int, PreparedDeviceBFM] = {}
        self._executors: dict[int, CommandExecutor] = {}
        self._reducers: dict[int, MotionTrackingCostReducer] = {}

    def world(self, batch: int) -> MjwarpWorld:
        if batch not in self._worlds:
            self._worlds[batch] = self._world_factory(batch)
        return self._worlds[batch]

    @property
    def implementation(self) -> str:
        return "device" if self._uses_device_bfm else "host-reference"

    @property
    def reducer_factory(self) -> Callable[[int], MotionTrackingCostReducer] | None:
        return self.reducer if self._uses_device_bfm else None

    def executor(self, batch: int) -> CommandExecutor:
        if batch not in self._executors:
            world = self.world(batch)
            if self._uses_device_bfm:
                bfm = self._adapter.build_device_bfm(
                    self._policy_path, world.state_view, world.command_view
                )
                if not isinstance(bfm, PreparedDeviceBFM):
                    raise TypeError(
                        "build_device_bfm must return a PreparedDeviceBFM"
                    )
                self._bfms[batch] = bfm
            else:
                assert self._host_policy is not None
                bfm = SerialReferenceBFM(
                    self._adapter, self._host_policy, batch_size=batch
                )
            self._executors[batch] = CommandExecutor(
                world,
                bfm,
                control_period=self._control_period,
                motion_command_query_offsets=self._motion_command_query_offsets,
            )
        return self._executors[batch]

    def reducer(self, batch: int) -> MotionTrackingCostReducer:
        if not self._uses_device_bfm:
            raise RuntimeError("host-reference runtime has no device cost reducer")
        if batch not in self._reducers:
            self.executor(batch)
            self._reducers[batch] = MotionTrackingCostReducer(
                self._objective, self._worlds[batch].state_view
            )
        return self._reducers[batch]

    def close(self) -> None:
        """Release policies before the world-owned CUDA buffers disappear."""
        error: Exception | None = None
        for bfm in reversed(tuple(self._bfms.values())):
            try:
                bfm.close()
            except Exception as exc:  # Preserve teardown of remaining capacities.
                if error is None:
                    error = exc
        self._bfms.clear()
        self._executors.clear()
        self._reducers.clear()
        self._worlds.clear()
        self._host_policy = None
        if error is not None:
            raise error


def load_experiment(
    path: str | Path, *, configs_root: str | Path | None = None
) -> tuple[dict[str, Any], ExperimentConfig]:
    repo = default_repo_root()
    root = repo / "configs" if configs_root is None else Path(configs_root)
    resolved = resolve_config(Path(path), root)
    dataset_root = resolved.get("reference", {}).get("dataset_root")
    if isinstance(dataset_root, str) and dataset_root:
        dataset = Path(dataset_root).expanduser()
        if not dataset.is_absolute():
            dataset = Path(path).resolve().parent / dataset
        resolved["reference"]["dataset_root"] = str(dataset.resolve())
    config = dataclass_from_dict(ExperimentConfig, resolved)
    return resolved, config


def build_optimizer(
    config: Mapping[str, Any], *, num_knots: int, dim: int
) -> tuple[SamplingOptimizer, IterationLimit]:
    """Construct any registered optimizer from a strict constructor mapping."""
    algorithm = config.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("optimizer.algorithm must be a non-empty string")
    iterations = config.get("num_iterations")
    if iterations == "inf":
        if algorithm != "cem":
            raise ValueError("optimizer convergence mode is supported only for cem")
        criteria = config.get("convergence")
        if not isinstance(criteria, Mapping):
            raise ValueError("optimizer.convergence must explicitly specify patience and min_cost_improvement")
        iterations = dataclass_from_dict(ConvergenceCriteria, dict(criteria))
    else:
        if "convergence" in config:
            raise ValueError("optimizer.convergence is valid only with num_iterations: 'inf'")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("optimizer.num_iterations must be a positive integer or 'inf'")
    optimizer_type = get_optimizer(algorithm)
    accepted = inspect.signature(optimizer_type).parameters
    reserved = {"num_knots", "dim"}.intersection(config)
    if reserved:
        raise ValueError(
            "optimizer config cannot set search-space fields: "
            f"{sorted(reserved)}"
        )
    unknown = set(config).difference({"algorithm", "num_iterations", "convergence", *accepted})
    if unknown:
        raise ValueError(
            f"unknown parameters for optimizer {algorithm!r}: {sorted(unknown)}"
        )
    values = {**dict(config), "num_knots": num_knots, "dim": dim}
    values.pop("algorithm", None)
    kwargs = {key: value for key, value in values.items() if key in accepted}
    return optimizer_type(**kwargs), iterations


def run_experiment(
    experiment_path: str | Path,
    *,
    device_index: int = 0,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    configs_root: str | Path | None = None,
) -> ExperimentResult | None:
    """Execute one paired baseline/optimized run from a user or repository YAML."""
    with ExitStack() as resources:
        return _execute_experiment(
            experiment_path,
            device_index=device_index,
            output_dir=output_dir,
            dry_run=dry_run,
            configs_root=configs_root,
            resources=resources,
        )


def _execute_experiment(
    experiment_path: str | Path,
    *,
    device_index: int,
    output_dir: str | Path | None,
    dry_run: bool,
    configs_root: str | Path | None,
    resources: ExitStack,
) -> ExperimentResult | None:
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    resolved, config = load_experiment(experiment_path, configs_root=configs_root)
    bfm_config = instance_config_from_mapping(config.bfm)
    objective = dataclass_from_dict(MotionTrackingObjective, config.motion_tracking)
    latent_dim = 6 + len(bfm_config.command_joint_names)
    optimizer, iterations = build_optimizer(
        config.optimizer, num_knots=config.planning.num_knots, dim=latent_dim
    )
    _validate_cross_config(config, bfm_config)
    if dry_run:
        return None

    repo = default_repo_root()
    source = source_identity(repo)
    destination = (
        _default_output_dir(repo, config.name, source.source_sha256)
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    _prepare_output_dir(destination)
    write_resolved_experiment(resolved, destination)

    entries = load_lock(repo / LOCK_RELATIVE)
    policy_entry = resolve(bfm_config.artifacts["policy"], lock=entries)
    robot_entry = resolve(str(config.robot["artifact"]), lock=entries)
    policy_path = ensure_artifact(policy_entry.id, root=repo, lock=entries)
    robot_root = ensure_artifact(robot_entry.id, root=repo, lock=entries)
    target_assets = _resolve_reference_assets(config.reference, entries, repo)
    model_path = (robot_root / str(config.robot["entrypoint"])).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"robot entrypoint does not exist: {model_path}")

    adapter = get_family(bfm_config.family)(bfm_config)

    def world(batch: int) -> MjwarpWorld:
        return _build_world(
            model_path,
            config.robot,
            bfm_config,
            body_names=objective.body_names,
            batch=batch,
            device_index=device_index,
            scenario_count=(
                config.robustness.num_scenarios
                if config.robustness.enabled and batch != 1
                else 1
            ),
        )

    runtimes = _RolloutRuntimePool(
        world_factory=world,
        adapter=adapter,
        bfm_config=bfm_config,
        policy_path=policy_path,
        device_index=device_index,
        objective=objective,
        control_period=bfm_config.control_period,
        motion_command_query_offsets=bfm_config.motion_command_query_offsets,
    )
    resources.callback(runtimes.close)
    execution_world = runtimes.world(1)
    reference = load_reference_motion(
        target_assets.dataset_root,
        config.reference.motion_id,
        joint_names=bfm_config.command_joint_names,
        body_names=execution_world.body_names,
        control_period=bfm_config.control_period,
        projector=execution_world,
        clip_duration_seconds=None,
    )
    target_manifest = json.loads(
        (target_assets.dataset_root / "manifest.json").read_text()
    )
    if config.reference.dataset_root is not None and (
        _resolve_reference_assets(config.reference, entries, repo).reference_digest
        != target_assets.reference_digest
    ):
        raise ValueError("local reference changed while loading; retry with unchanged input files")
    if len(reference.query_offsets) < 2:
        raise ValueError("reference must contain a reset frame and at least one future frame")
    target_qpos, target_qvel = execution_world.physical_trajectory(
        reference, bfm_config.control_period
    )
    reference_future = take_motion_command(reference, np.arange(1, len(reference.query_offsets)))
    execution_steps = (
        len(reference_future.query_offsets)
        if config.reference.clip_duration_seconds is None
        else round(config.reference.clip_duration_seconds / bfm_config.control_period)
    )
    if execution_steps > len(reference_future.query_offsets):
        raise ValueError("reference duration exceeds the available motion")
    execution_reference = take_motion_command(
        reference_future, np.arange(execution_steps)
    )
    past_context_steps = max(0, -min(bfm_config.motion_command_query_offsets))
    future_context_steps = max(0, max(bfm_config.motion_command_query_offsets))
    initial_execution_state = ExecutionState.capture(
        target_qpos[:, 0],
        target_qvel[:, 0],
        control_step=0,
        time=0.0,
        motion_command_index=0,
        command=None,
        bfm_state=adapter.initial_state(1),
        adapter=adapter,
        motion_command_history=_initial_motion_command_history(reference, past_context_steps),
    )

    execution = runtimes.executor(1)
    baseline_raw = execution.execute(
        initial_execution_state,
        execution_reference,
        num_steps=execution_steps,
        right_context=reference_right_context(
            reference_future, execution_steps, future_context_steps
        ),
        requires=frozenset(MeasurementPrimitive),
    )
    baseline_trace = ExecutionTrace(
        timeline=bfm_config.control_period
        * np.arange(1, execution_steps + 1, dtype=np.float64),
        reference_motion=execution_reference,
        applied_command=execution_reference,
        executed_trajectory=baseline_raw.measurements,
        qpos=baseline_raw.qpos[0],
        qvel=baseline_raw.qvel[0],
        valid=np.ones(execution_steps, dtype=np.bool_),
        root_body_name=config.planning.root_body_name,
    )

    def search_space(reference: MotionCommand) -> QposKnotsSpace:
        return QposKnotsSpace(
            reference,
            num_knots=config.planning.num_knots,
            control_period=bfm_config.control_period,
            scales=config.planning.scales,
            projector=execution_world,
            pelvis_body=config.planning.root_body_name,
        )

    def evaluator(
        space: QposKnotsSpace,
        reference_motion: MotionCommand,
        right_context: MotionCommand | None,
        execution_state: ExecutionState,
    ) -> RolloutEvaluator:
        scenarios = None
        scenario_configurer = None
        if config.robustness.enabled:
            num_joints = execution_state.qvel.shape[1] - 6
            if num_joints < 1:
                raise ValueError("robust evaluation requires one free root and joint DOFs")
            scenarios = sample_scenarios(
                config.robustness.ranges,
                count=config.robustness.num_scenarios,
                seed=config.robustness.scenario_seed + execution_state.motion_command_index,
                num_joints=num_joints,
            )

            def scenario_configurer(batch: int, values: ScenarioBatch) -> None:
                runtimes.world(batch).configure_scenarios(values)

        return RolloutEvaluator(
            search_space=space,
            objective=objective,
            reference_motion=reference_motion,
            execution_state=execution_state,
            bfm_adapter=adapter,
            executor_factory=runtimes.executor,
            max_worlds=config.planning.max_worlds,
            failure_penalty=config.planning.failure_penalty,
            right_context=right_context,
            reducer_factory=runtimes.reducer_factory,
            scenarios=scenarios,
            risk_strategy=config.robustness.risk,
            scenario_configurer=scenario_configurer,
        )

    optimized = run_command_optimization(
        body_names=objective.body_names,
        reference_future=reference_future,
        initial_execution_state=initial_execution_state,
        optimizer=optimizer,
        search_space_factory=search_space,
        evaluator_factory=evaluator,
        executor=execution,
        horizon_steps=config.planning.horizon_steps,
        replanning_interval_steps=config.planning.replanning_interval_steps,
        control_period=bfm_config.control_period,
        seed=config.planning.seed,
        iterations=iterations,
        root_body_name=config.planning.root_body_name,
        execution_steps=execution_steps,
        future_context_steps=future_context_steps,
        evaluate_proposal=config.planning.evaluate_proposal,
    )

    identities = dict(
        reference_id=target_assets.reference_id,
        reference_digest=target_assets.reference_digest,
        bfm_id=bfm_config.instance,
        checkpoint_digest=policy_entry.sha256,
        robot_model_digest=robot_entry.tree_sha256 or robot_entry.sha256,
        initial_physical_state_digest=physical_state_digest(
            initial_execution_state.qpos, initial_execution_state.qvel
        ),
        control_period=bfm_config.control_period,
    )
    pair = PairedRunArtifact(
        baseline=ExecutionRunArtifact(
            kind="baseline", trace=baseline_trace, planning=(), **identities
        ),
        optimized=ExecutionRunArtifact(
            kind="optimized",
            trace=optimized.trace,
            planning=optimized.planning,
            **identities,
        ),
    )
    metrics = compute_paired_metrics(pair, objective=objective)
    reference_qpos = target_qpos[0, 1 : execution_steps + 1]
    _persist_experiment(
        destination,
        pair=pair,
        metrics=metrics,
        reference_qpos=reference_qpos,
        source=source,
        model_path=model_path,
        reference_sha256=target_assets.reference_digest,
        config=config,
        device_index=device_index,
        bfm_runtime=runtimes.implementation,
    )
    if config.output.formats:
        initial_qpos = execution_world.canonical_qpos(initial_execution_state.qpos[0])
        trace = pair.optimized.trace
        for trajectory in config.output.trajectories:
            if trajectory == "execution":
                qpos = execution_world.canonical_qpos(trace.qpos)
            else:
                motion_command = trace.applied_command
                root_index = motion_command.body_names.index(config.planning.root_body_name)
                qpos = np.concatenate(
                    (
                        motion_command.body_position_world[0, :, root_index],
                        motion_command.body_orientation_world[0, :, root_index],
                        motion_command.joint_position[0],
                    ),
                    axis=-1,
                )
            export_motion(
                destination / "exports", trajectory=trajectory,
                qpos=np.concatenate((initial_qpos[None], qpos)),
                valid=np.concatenate(([True], trace.valid)),
                joint_names=bfm_config.command_joint_names,
                control_period=bfm_config.control_period,
                mjcf=target_manifest["mjcf"],
                formats=config.output.formats,
            )
    if config.visualization.enabled:
        _render_experiment(destination, model_path=model_path)
    return ExperimentResult(destination, pair)


def _validate_cross_config(config: ExperimentConfig, bfm: Any) -> None:
    joints = tuple(config.robot.get("joint_names", ()))
    if joints != bfm.command_joint_names:
        raise ValueError("robot and bfm command joint order must match exactly")
    if config.planning.root_body_name not in bfm.motion_command_body_names:
        raise ValueError("planning root body must be part of bfm motion_command bodies")
    if (
        config.robustness.enabled
        and config.planning.max_worlds < config.robustness.num_scenarios
    ):
        raise ValueError("planning.max_worlds must fit one complete scenario group")
    if config.reference.clip_duration_seconds is None:
        return
    duration_steps = round(config.reference.clip_duration_seconds / bfm.control_period)
    if duration_steps < 1 or not np.isclose(
        config.reference.clip_duration_seconds,
        duration_steps * bfm.control_period,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("reference.clip_duration_seconds must align with bfm control_period")


def _resolve_reference_assets(
    config: ReferenceConfig,
    entries: Mapping[str, ArtifactEntry],
    repo: Path,
) -> _TargetAssets:
    if config.dataset_root is not None:
        dataset_root = Path(config.dataset_root).expanduser().resolve()
        motion_path = resolve_any4hdmi_motion(dataset_root, config.motion_id)
        file_digests = {
            "manifest.json": sha256_file(dataset_root / "manifest.json"),
            "motion": sha256_file(motion_path),
        }
        digest = hashlib.sha256(
            json.dumps(file_digests, sort_keys=True).encode("ascii")
        ).hexdigest()
        return _TargetAssets(dataset_root, f"local:{config.motion_id}", digest)
    manifest_entry = resolve(config.manifest_artifact, lock=entries)
    motion_entry = resolve(config.motion_artifact, lock=entries)
    manifest_path = ensure_artifact(manifest_entry.id, root=repo, lock=entries)
    motion_path = ensure_artifact(motion_entry.id, root=repo, lock=entries)

    if manifest_entry.kind == motion_entry.kind == "file":
        dataset_root = manifest_path.parent.resolve()
        if manifest_path.resolve() != dataset_root / "manifest.json":
            raise ValueError("reference manifest artifact must be manifest.json")
        expected_motion = resolve_any4hdmi_motion(dataset_root, config.motion_id)
        if motion_path.resolve() != expected_motion:
            raise ValueError("reference artifact path does not match reference.motion_id")
        return _TargetAssets(dataset_root, motion_entry.id, motion_entry.sha256)

    if manifest_entry.kind == motion_entry.kind == "tar.gz":
        manifest_root = manifest_path.resolve()
        motion_root = motion_path.resolve()
        if manifest_root != motion_root:
            raise ValueError("reference tree artifacts must resolve to the same dataset root")
        resolve_any4hdmi_motion(motion_root, config.motion_id)
        if motion_entry.tree_sha256 is None:
            raise ValueError("reference tree artifact requires tree_sha256")
        return _TargetAssets(
            motion_root,
            f"{motion_entry.id}:{config.motion_id}",
            motion_entry.tree_sha256,
        )

    raise ValueError("reference manifest and motion artifacts must both be files or trees")


def _build_world(
    model_path: Path,
    robot: Mapping[str, Any],
    bfm: Any,
    *,
    body_names: tuple[str, ...],
    batch: int,
    device_index: int,
    scenario_count: int = 1,
) -> MjwarpWorld:
    return MjwarpWorld(
        model_path,
        batch_size=batch,
        joint_names=bfm.command_joint_names,
        body_names=tuple(dict.fromkeys((*body_names, *bfm.motion_command_body_names))),
        physics_timestep=float(robot["physics_timestep"]),
        joint_effort_limit=np.asarray(robot["joint_effort_limit"], dtype=np.float64),
        floor_friction=tuple(robot["floor_friction"]),
        joint_armature=np.asarray(robot["joint_armature"], dtype=np.float64),
        joint_frictionloss=np.asarray(robot["joint_frictionloss"], dtype=np.float64),
        motion_command_velocity_stencil=bfm.motion_command_velocity_stencil,
        scenario_count=scenario_count,
        device=f"cuda:{device_index}",
        nconmax=int(robot["nconmax"]),
        njmax=int(robot["njmax"]),
    )


def _initial_motion_command_history(
    reference: MotionCommand, num_steps: int
) -> MotionCommand | None:
    if num_steps == 0:
        return None
    reset = take_motion_command(reference, np.zeros(num_steps, dtype=np.int64))
    return MotionCommand(
        query_offsets=tuple(range(-num_steps, 0)),
        joint_names=reset.joint_names,
        body_names=reset.body_names,
        **{field: getattr(reset, field) for field in _COMMAND_ARRAYS},
    )


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(
                f"experiment output path is not a directory: {path}"
            )
        if any(path.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty experiment directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def _default_output_dir(repo: Path, experiment: str, source_sha256: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return repo / "outputs" / experiment / f"{timestamp}-{source_sha256[:8]}"


def _persist_experiment(
    output: Path,
    *,
    pair: PairedRunArtifact,
    metrics: Any,
    reference_qpos: np.ndarray,
    source: SourceIdentity,
    model_path: Path,
    reference_sha256: str,
    config: ExperimentConfig,
    device_index: int,
    bfm_runtime: str,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.int64(LIVE_SCHEMA_VERSION),
        "reference_qpos": reference_qpos,
    }
    reference_motion = pair.baseline.trace.reference_motion
    for field in _COMMAND_ARRAYS:
        arrays[f"reference_{field}"] = getattr(reference_motion, field)
    for label, run in (("baseline", pair.baseline), ("optimized", pair.optimized)):
        trace = run.trace
        arrays[f"{label}_timeline"] = trace.timeline
        arrays[f"{label}_valid"] = trace.valid
        arrays[f"{label}_qpos"] = trace.qpos
        arrays[f"{label}_qvel"] = trace.qvel
        for field in _COMMAND_ARRAYS:
            arrays[f"{label}_applied_command_{field}"] = getattr(
                trace.applied_command, field
            )
        if trace.executed_trajectory is not None:
            for primitive, value in trace.executed_trajectory.arrays.items():
                arrays[f"{label}_execution_{primitive.value}"] = value
    np.savez_compressed(output / TRACE_FILE, **arrays)
    planning_path = output / PLANNING_FILE
    _write_json(
        planning_path,
        {
            "schema_version": LIVE_SCHEMA_VERSION,
            "updates": [
                _planning_json(item) for item in pair.optimized.planning
            ],
        },
    )
    randomization_path = output / RANDOMIZATION_FILE
    _write_json(
        randomization_path,
        _randomization_json(config.robustness, pair.optimized.planning, config.robot),
    )
    report = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "claim_scope": "paired configured-motion tracking",
        "baseline": dataclasses.asdict(metrics.baseline),
        "optimized": dataclasses.asdict(metrics.optimized),
        "identities": {
            "reference_id": pair.baseline.reference_id,
            "reference_digest": pair.baseline.reference_digest,
            "bfm_id": pair.baseline.bfm_id,
            "checkpoint_digest": pair.baseline.checkpoint_digest,
            "robot_model_digest": pair.baseline.robot_model_digest,
            "initial_physical_state_digest": (
                pair.baseline.initial_physical_state_digest
            ),
            "control_period": pair.baseline.control_period,
        },
    }
    _write_json(output / REPORT_FILE, report)
    provenance = {
        "schema_version": LIVE_SCHEMA_VERSION,
        **dataclasses.asdict(source),
        "reference_sha256": reference_sha256,
        "mjcf_sha256": sha256_file(model_path),
        "trace_sha256": sha256_file(output / TRACE_FILE),
        "report_sha256": sha256_file(output / REPORT_FILE),
        "planning_sha256": sha256_file(planning_path),
        "randomization_sha256": sha256_file(randomization_path),
        "resolved_experiment_sha256": sha256_file(
            output / RESOLVED_EXPERIMENT_FILE
        ),
        "runtime_versions": {
            package: version(package)
            for package in ("mujoco", "mujoco-warp", "warp-lang", "onnxruntime-gpu")
        },
        "bfm_runtime": bfm_runtime,
        "hardware": _hardware_identity(device_index),
        "visualization": dataclasses.asdict(config.visualization),
    }
    _write_json(output / PROVENANCE_FILE, provenance)


def _hardware_identity(device_index: int) -> dict[str, Any]:
    """Identify the selected logical CUDA device through ``nvidia-smi``."""
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    base: dict[str, Any] = {
        "logical_device_index": device_index,
        "cuda_visible_devices": visible_raw,
    }
    if visible_raw is None:
        selector = str(device_index)
    else:
        visible = tuple(
            item.strip() for item in visible_raw.split(",") if item.strip()
        )
        if device_index >= len(visible):
            return {
                **base,
                "available": False,
                "error": "logical device is absent from CUDA_VISIBLE_DEVICES",
            }
        selector = visible[device_index]
    base["selector"] = selector
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=index,uuid,name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            **base,
            "available": False,
            "error": "nvidia-smi query failed",
        }
    rows = [line for line in output.splitlines() if line.strip()]
    fields = [item.strip() for item in rows[0].split(",")] if len(rows) == 1 else []
    if len(fields) != 5:
        return {
            **base,
            "available": False,
            "error": "unexpected nvidia-smi output",
        }
    physical_index, gpu_uuid, gpu_name, compute_capability, driver_version = fields
    return {
        **base,
        "available": True,
        "physical_device_index": int(physical_index),
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "driver_version": driver_version,
    }


def _planning_json(item: PlanningDiagnostics) -> dict[str, Any]:
    return {
        "update_step": item.update_step,
        "selected_source": item.selected_source,
        "selected_candidate_id": item.selected_candidate_id,
        "selected_cost": item.selected_cost,
        "best_observed_candidate_id": item.best_observed_candidate_id,
        "best_observed_cost": item.best_observed_cost,
        "proposal_candidate_id": item.proposal_candidate_id,
        "proposal_cost": item.proposal_cost,
        "cost_quantiles": item.cost_quantiles.tolist(),
        "candidate_deformation": dict(item.candidate_deformation),
        "candidate_evaluations": item.candidate_evaluations,
        "scenario_count": item.scenario_count,
        "scenario_digest": item.scenario_digest,
        "logical_rollouts": item.logical_rollouts,
        "physical_rollouts": item.physical_rollouts,
        "penalized_candidate_evaluations": (
            item.penalized_candidate_evaluations
        ),
        "invalid_scenario_rollouts": item.invalid_scenario_rollouts,
        "update_wall_seconds": item.update_wall_seconds,
    }


def _randomization_json(
    config: RobustnessConfig,
    planning: tuple[PlanningDiagnostics, ...],
    robot: Mapping[str, Any],
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    for item in planning:
        expected = None
        if config.enabled:
            expected = sample_scenarios(
                config.ranges,
                count=config.num_scenarios,
                seed=config.scenario_seed + item.update_step,
                num_joints=len(tuple(robot["joint_names"])),
            ).digest
        if item.scenario_digest != expected:
            raise RuntimeError("persisted planning scenario identity is inconsistent")
        updates.append(
            {
                "update_step": item.update_step,
                "seed": (
                    None
                    if not config.enabled
                    else config.scenario_seed + item.update_step
                ),
                "scenario_digest": item.scenario_digest,
            }
        )
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "enabled": config.enabled,
        "num_scenarios": config.num_scenarios,
        "scenario_seed": config.scenario_seed,
        "risk": config.risk,
        "ranges": dataclasses.asdict(config.ranges),
        "updates": updates,
    }


def _render_experiment(
    experiment_dir: Path, *, model_path: Path | None = None
) -> Path:
    resolved = yaml.safe_load(
        (experiment_dir / RESOLVED_EXPERIMENT_FILE).read_text()
    )
    experiment = dataclass_from_dict(ExperimentConfig, resolved)
    for name in (REPORT_FILE, PLANNING_FILE, RANDOMIZATION_FILE):
        payload = json.loads((experiment_dir / name).read_text())
        _require_schema_version(payload, name)
    provenance = json.loads((experiment_dir / PROVENANCE_FILE).read_text())
    _require_schema_version(provenance, PROVENANCE_FILE)
    if model_path is None:
        robot = experiment.robot
        root = ensure_artifact(str(robot["artifact"]), root=default_repo_root())
        model_path = root / str(robot["entrypoint"])
    with np.load(experiment_dir / TRACE_FILE, allow_pickle=False) as traces:
        trace_schema_version = np.asarray(traces["schema_version"])
        if (
            trace_schema_version.shape != ()
            or not np.issubdtype(trace_schema_version.dtype, np.integer)
            or int(trace_schema_version) != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{TRACE_FILE} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        render = render_mp4(
            mjcf_path=model_path,
            timeline=traces["baseline_timeline"],
            reference_qpos=traces["reference_qpos"],
            view=ComparisonView(
                left=RenderTrack("baseline", traces["baseline_qpos"]),
                right=RenderTrack("optimized", traces["optimized_qpos"]),
            ),
            output_path=experiment_dir / RENDER_FILE,
            config=dataclasses.replace(experiment.visualization, enabled=True),
            bindings=RenderBindings(
                release_version=provenance["release_version"],
                source_sha256=provenance["source_sha256"],
                reference_sha256=provenance["reference_sha256"],
                trace_sha256=sha256_file(experiment_dir / TRACE_FILE),
                report_sha256=sha256_file(experiment_dir / REPORT_FILE),
                mjcf_sha256=sha256_file(model_path),
            ),
        )
    _write_json(
        experiment_dir / RENDER_PROVENANCE_FILE, dataclasses.asdict(render)
    )
    return experiment_dir / RENDER_FILE


def _require_schema_version(payload: object, name: str) -> None:
    if (
        not isinstance(payload, Mapping)
        or isinstance(payload.get("schema_version"), bool)
        or not isinstance(payload.get("schema_version"), int)
        or payload.get("schema_version") != LIVE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{name} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
        )


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare_bfm(
    artifact_id: str,
    *,
    root: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> Path:
    """Rebuild one lock-declared derived bfm artifact."""
    return prepare_bfm_artifact(artifact_id, root=root, lock_path=lock_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bfm-ttc")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute one experiment YAML")
    run.add_argument("experiment", type=Path)
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--dry-run", action="store_true")
    batch = commands.add_parser("batch", help="execute an Experiment Batch YAML")
    batch.add_argument("experiment_batch", type=Path)
    batch.add_argument("--device-index", type=int, default=0)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--dry-run", action="store_true")
    render = commands.add_parser("render", help="render persisted paired traces")
    render.add_argument("experiment_dir", type=Path)
    prepare = commands.add_parser("prepare-bfm", help="prepare batched bfm ONNX")
    prepare.add_argument(
        "artifact_id",
        help="derived artifact identity from artifacts/lock.v1.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_experiment(
            args.experiment,
            device_index=args.device_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        if result is not None:
            print(result.output_dir)
    elif args.command == "batch":
        from command_optimization.batch import run_experiment_batch

        result = run_experiment_batch(
            args.experiment_batch,
            execute_experiment=run_experiment,
            device_index=args.device_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        if result is not None:
            print(result.output_dir)
            if not result.success:
                return 1
    elif args.command == "render":
        print(_render_experiment(args.experiment_dir.resolve()))
    else:
        print(prepare_bfm(args.artifact_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
