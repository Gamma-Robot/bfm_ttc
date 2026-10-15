"""HEFT-PMG bfm adapter over common bfm currencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bfm_adapters.batched_bfm import (
    PreparedDeviceBFM,
    RobotStateView,
)
from bfm_adapters.contracts import (
    DevicePDCommand,
    MotionCommand,
    NamedTensors,
    PDCommand,
    RobotState,
)
from bfm_adapters.families.heft.command import action_to_pd
from bfm_adapters.families.heft.observation import (
    ACTION_DIM,
    CONTEXT_DIM,
    HISTORY_LAGS,
    HISTORY_LENGTH,
    MOTION_COMMAND_DIM,
    PREVIOUS_ACTIONS,
    PROPRIOCEPTION_DIM,
    QUERY_OFFSETS,
    TARGET_MOTION_DIM,
    append_newest,
    pack_motion_command,
    pack_proprioception,
    pack_target_motion,
    projected_gravity,
    root_angular_velocity_body,
)
from bfm_adapters.registry import BFMInstanceConfig, available, register

FAMILY = "heft"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _HeftConfig:
    root_body: str
    boot_indicator_max: int
    history_lags: tuple[int, ...]
    previous_action_count: int
    action_clip: float
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _HeftConfig:
        required = {
            "root_body",
            "boot_indicator_max",
            "history_lags",
            "previous_action_count",
            "action_clip",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                f"heft family_config keys must be {sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("heft root_body must be non-empty")
        boot = raw["boot_indicator_max"]
        if not isinstance(boot, int) or boot != 25:
            raise ValueError("heft boot_indicator_max must be 25")
        lags = tuple(raw["history_lags"])  # type: ignore[arg-type]
        if lags != HISTORY_LAGS:
            raise ValueError(f"heft history_lags must be {HISTORY_LAGS}")
        previous = raw["previous_action_count"]
        if not isinstance(previous, int) or previous != PREVIOUS_ACTIONS:
            raise ValueError(f"heft previous_action_count must be {PREVIOUS_ACTIONS}")
        clip = float(raw["action_clip"])  # type: ignore[arg-type]
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("heft action_clip must be finite and positive")

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"heft {name} must contain {ACTION_DIM} finite values")
            return value.copy()

        return cls(
            root_body=root,
            boot_indicator_max=boot,
            history_lags=lags,
            previous_action_count=previous,
            action_clip=clip,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class HeftAdapter:
    """PMG observation/state semantics without simulator or policy execution."""

    def __init__(self, config: BFMInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"HeftAdapter requires family {FAMILY!r}")
        expected_inputs = {
            "context": (CONTEXT_DIM,),
            "motion_command": (MOTION_COMMAND_DIM,),
            "target_motion": (TARGET_MOTION_DIM,),
            "proprioception": (PROPRIOCEPTION_DIM,),
        }
        actual_inputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_inputs
        }
        if actual_inputs != expected_inputs:
            raise ValueError(f"heft policy_inputs must be {expected_inputs}")
        actual_outputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_outputs
        }
        if actual_outputs != {"action": (ACTION_DIM,)}:
            raise ValueError("heft policy_outputs must contain action[29]")
        if config.motion_command_query_offsets != QUERY_OFFSETS:
            raise ValueError(f"heft motion_command_query_offsets must be {QUERY_OFFSETS}")
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"heft requires {ACTION_DIM} policy joints")
        family = _HeftConfig.from_mapping(config.family_config)
        if family.root_body not in config.motion_command_body_names:
            raise ValueError("heft motion_command bodies must contain root_body")
        self._config = config
        self._family = family
        self._root_index = config.motion_command_body_names.index(family.root_body)
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "boot": np.full(
                batch_size, self._family.boot_indicator_max, dtype=np.int32
            ),
            "root_angular_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, 3), np.float32
            ),
            "projected_gravity": np.zeros((batch_size, HISTORY_LENGTH, 3), np.float32),
            "joint_position": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "joint_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "previous_actions": np.zeros(
                (batch_size, PREVIOUS_ACTIONS, ACTION_DIM), np.float32
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), np.float32),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "heft expand_state requires one lane and positive batch_size"
            )
        return {
            name: np.repeat(value, batch_size, axis=0)
            for name, value in state.items()
        }

    def build_policy_inputs(
        self,
        state: RobotState,
        motion_command: MotionCommand,
        bfm_state: dict[str, np.ndarray],
    ) -> tuple[NamedTensors, dict[str, np.ndarray]]:
        if motion_command.query_offsets != QUERY_OFFSETS:
            raise ValueError("heft motion_command offsets differ from the bfm config")
        batch = state.joint_position.shape[0]
        if (
            motion_command.joint_position.shape[0] != batch
            or bfm_state["boot"].shape != (batch,)
        ):
            raise ValueError("HEFT state/motion_command/bfm-state batch sizes differ")
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _order(motion_command.joint_names, self._config.policy_joint_names)
        guide_bodies = _order(motion_command.body_names, self._config.motion_command_body_names)
        state_bodies = _order(state.body_names, self._config.motion_command_body_names)

        joint_position = state.joint_position[:, state_joints].astype(
            np.float32, copy=False
        )
        joint_velocity = state.joint_velocity[:, state_joints].astype(
            np.float32, copy=False
        )
        robot_root_orientation = state.body_orientation_world[
            :, state_bodies[self._root_index]
        ].astype(np.float32, copy=False)
        robot_root_angular_velocity = state.body_angular_velocity_world[
            :, state_bodies[self._root_index]
        ].astype(np.float32, copy=False)
        reference_joint_position = motion_command.joint_position[:, :, guide_joints].astype(
            np.float32, copy=False
        )
        root_column = guide_bodies[self._root_index]
        reference_root_position = motion_command.body_position_world[
            :, :, root_column
        ].astype(np.float32, copy=False)
        reference_root_orientation = motion_command.body_orientation_world[
            :, :, root_column
        ].astype(np.float32, copy=False)

        next_state = self.clone_state(bfm_state)
        next_state["boot"] = np.maximum(bfm_state["boot"] - 1, 0).astype(np.int32)
        next_state["root_angular_velocity"] = append_newest(
            bfm_state["root_angular_velocity"],
            root_angular_velocity_body(
                robot_root_orientation, robot_root_angular_velocity
            ),
        )
        next_state["projected_gravity"] = append_newest(
            bfm_state["projected_gravity"],
            projected_gravity(robot_root_orientation),
        )
        next_state["joint_position"] = append_newest(
            bfm_state["joint_position"], joint_position
        )
        next_state["joint_velocity"] = append_newest(
            bfm_state["joint_velocity"], joint_velocity
        )
        next_state["previous_actions"] = append_newest(
            bfm_state["previous_actions"], bfm_state["last_action"]
        )

        inputs = {
            "context": (
                next_state["boot"].astype(np.float32)[:, None]
                / np.float32(self._family.boot_indicator_max)
            ),
            "motion_command": pack_motion_command(
                reference_root_position,
                reference_root_orientation,
                robot_root_orientation,
            ),
            "target_motion": pack_target_motion(
                reference_joint_position,
                joint_position,
                reference_root_position,
                reference_root_orientation,
            ),
            "proprioception": pack_proprioception(
                next_state["root_angular_velocity"],
                next_state["projected_gravity"],
                next_state["joint_position"],
                next_state["joint_velocity"],
                next_state["previous_actions"],
            ),
        }
        return (
            {
                name: value.astype(self._input_dtypes[name], copy=False)
                for name, value in inputs.items()
            },
            next_state,
        )

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        bfm_state: dict[str, np.ndarray],
    ) -> tuple[PDCommand, dict[str, np.ndarray]]:
        command, clipped = action_to_pd(
            outputs["action"],
            clip=self._family.action_clip,
            policy_joint_names=self._config.policy_joint_names,
            command_joint_names=self._config.command_joint_names,
            default_joint_position=self._family.default_joint_position,
            action_scale=self._family.action_scale,
            proportional_gain=self._family.proportional_gain,
            derivative_gain=self._family.derivative_gain,
        )
        if clipped.shape != bfm_state["last_action"].shape:
            raise ValueError("HEFT action batch differs from bfm state")
        next_state = self.clone_state(bfm_state)
        next_state["last_action"] = clipped
        return command, next_state

    def build_device_bfm(
        self,
        policy_path: str | Path,
        state_view: RobotStateView,
        command_view: DevicePDCommand,
    ) -> PreparedDeviceBFM:
        """Build the optional CUDA runtime without importing it on the host path."""

        from bfm_adapters.families.heft.device import HeftDeviceBFM

        return HeftDeviceBFM(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_heft(config: BFMInstanceConfig) -> HeftAdapter:
    return HeftAdapter(config)


assert FAMILY in available()
