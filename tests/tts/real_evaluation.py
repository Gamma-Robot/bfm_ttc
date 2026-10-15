"""Shared construction of real host/device Candidate Rollout evaluation cases."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bfm_adapters import (
    SerialReferenceBFM,
    get_family,
    instance_config_from_mapping,
)
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import LOCK_RELATIVE, ensure_artifact, load_lock, resolve
from command_optimization.config import dataclass_from_dict
from command_optimization.motion_tracking.objective import MotionTrackingObjective
from command_optimization.motion_tracking.search_space import QposKnotsSpace
from command_optimization.motion_tracking.any4hdmi import load_reference_motion
from command_optimization.motion_tracking.command import reference_right_context, take_motion_command
from command_optimization.planning.evaluation import RolloutEvaluator
from command_optimization.run import (
    _RolloutRuntimePool,
    _build_world,
    _initial_motion_command_history,
    _resolve_reference_assets,
    load_experiment,
)
from command_optimization.simulation.command_execution import CommandExecutor
from command_optimization.simulation.execution_state import ExecutionState


REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


class HostRuntimePool:
    """Persistent serial-reference runtime used only as a conformance oracle."""

    def __init__(
        self,
        *,
        world_factory: Callable[[int], object],
        adapter: object,
        policy_path: Path,
        bfm_config: object,
        device_index: int,
    ) -> None:
        self._world_factory = world_factory
        self._adapter = adapter
        self._config = bfm_config
        self._policy = BatchedOrtCudaPolicy(
            policy_path, bfm_config, device_id=device_index
        )
        self._executors: dict[int, CommandExecutor] = {}

    def executor(self, batch: int) -> CommandExecutor:
        if batch not in self._executors:
            bfm = SerialReferenceBFM(
                self._adapter, self._policy, batch_size=batch
            )
            self._executors[batch] = CommandExecutor(
                self._world_factory(batch),
                bfm,
                control_period=self._config.control_period,
                motion_command_query_offsets=self._config.motion_command_query_offsets,
            )
        return self._executors[batch]

    def close(self) -> None:
        self._executors.clear()
        self._policy = None


@dataclass
class RealEvaluationCase:
    resolved: dict[str, Any]
    experiment: Any
    bfm_config: Any
    adapter: Any
    objective: MotionTrackingObjective
    space: QposKnotsSpace
    reference_motion: Any
    execution_state: ExecutionState
    right_context: Any
    device_pool: _RolloutRuntimePool
    host_pool: HostRuntimePool
    policy_entry: Any
    robot_entry: Any
    model_path: Path
    target_assets: Any
    device_index: int

    def evaluator(
        self, runtime: Literal["host-reference", "device"], *, max_worlds: int
    ) -> RolloutEvaluator:
        common = dict(
            search_space=self.space,
            objective=self.objective,
            reference_motion=self.reference_motion,
            execution_state=self.execution_state,
            bfm_adapter=self.adapter,
            max_worlds=max_worlds,
            failure_penalty=self.experiment.planning.failure_penalty,
            right_context=self.right_context,
        )
        if runtime == "host-reference":
            return RolloutEvaluator(
                **common,
                executor_factory=self.host_pool.executor,
            )
        return RolloutEvaluator(
            **common,
            executor_factory=self.device_pool.executor,
            reducer_factory=self.device_pool.reducer,
        )

    def close(self) -> None:
        try:
            self.device_pool.close()
        finally:
            self.host_pool.close()
            gc.collect()


def build_real_evaluation_case(
    experiment_name: str,
    *,
    horizon: int,
    num_knots: int,
    device_index: int,
) -> RealEvaluationCase:
    """Build one shared ExecutionState host/device case from a committed experiment."""
    resolved, experiment = load_experiment(REPO / "tests" / "fixtures" / "experiments" / experiment_name)
    bfm_config = instance_config_from_mapping(experiment.bfm)
    adapter = get_family(bfm_config.family)(bfm_config)
    objective = dataclass_from_dict(
        MotionTrackingObjective, experiment.motion_tracking
    )
    lock = load_lock(REPO / LOCK_RELATIVE)
    policy_entry = resolve(bfm_config.artifacts["policy"], lock=lock)
    policy_path = ensure_artifact(policy_entry.id, root=REPO, lock=lock)
    robot_entry = resolve(str(experiment.robot["artifact"]), lock=lock)
    robot_root = ensure_artifact(robot_entry.id, root=REPO, lock=lock)
    model_path = (robot_root / str(experiment.robot["entrypoint"])).resolve()
    target_assets = _resolve_reference_assets(experiment.reference, lock, REPO)

    def world(batch: int):
        return _build_world(
            model_path,
            experiment.robot,
            bfm_config,
            body_names=objective.body_names,
            batch=batch,
            device_index=device_index,
        )

    device_pool = _RolloutRuntimePool(
        world_factory=world,
        adapter=adapter,
        bfm_config=bfm_config,
        policy_path=policy_path,
        device_index=device_index,
        objective=objective,
        control_period=bfm_config.control_period,
        motion_command_query_offsets=bfm_config.motion_command_query_offsets,
    )
    host_pool: HostRuntimePool | None = None
    try:
        host_pool = HostRuntimePool(
            world_factory=world,
            adapter=adapter,
            policy_path=policy_path,
            bfm_config=bfm_config,
            device_index=device_index,
        )
        projection_world = device_pool.world(1)
        future_context_steps = max(0, max(bfm_config.motion_command_query_offsets))
        target = load_reference_motion(
            target_assets.dataset_root,
            experiment.reference.motion_id,
            joint_names=bfm_config.command_joint_names,
            body_names=projection_world.body_names,
            control_period=bfm_config.control_period,
            projector=projection_world,
            clip_duration_seconds=(horizon + future_context_steps)
            * bfm_config.control_period,
        )
        target_qpos, target_qvel = projection_world.physical_trajectory(
            target, bfm_config.control_period
        )
        reference_future = take_motion_command(
            target, np.arange(1, len(target.query_offsets))
        )
        reference_motion = take_motion_command(reference_future, np.arange(horizon))
        right_context = reference_right_context(
            reference_future, horizon, future_context_steps
        )
        past_context_steps = max(0, -min(bfm_config.motion_command_query_offsets))
        execution_state = ExecutionState.capture(
            target_qpos[:, 0],
            target_qvel[:, 0],
            control_step=0,
            time=0.0,
            motion_command_index=0,
            command=None,
            bfm_state=adapter.initial_state(1),
            adapter=adapter,
            motion_command_history=_initial_motion_command_history(
                target, past_context_steps
            ),
        )
        space = QposKnotsSpace(
            reference_motion,
            num_knots=num_knots,
            control_period=bfm_config.control_period,
            scales=experiment.planning.scales,
            projector=projection_world,
            pelvis_body=experiment.planning.root_body_name,
        )
        return RealEvaluationCase(
            resolved=resolved,
            experiment=experiment,
            bfm_config=bfm_config,
            adapter=adapter,
            objective=objective,
            space=space,
            reference_motion=reference_motion,
            execution_state=execution_state,
            right_context=right_context,
            device_pool=device_pool,
            host_pool=host_pool,
            policy_entry=policy_entry,
            robot_entry=robot_entry,
            model_path=model_path,
            target_assets=target_assets,
            device_index=device_index,
        )
    except Exception:
        try:
            device_pool.close()
        finally:
            if host_pool is not None:
                host_pool.close()
        raise
