"""Released native-G1 SONIC adapter over common bfm currencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bfm_adapters.contracts import MotionCommand, NamedTensors, PDCommand, RobotState
from bfm_adapters.families.sonic.command import action_to_pd
from bfm_adapters.families.sonic.observation import (
    ACTION_DIM,
    G1_INPUT_DIM,
    HISTORY_LENGTH,
    PROPRIOCEPTION_DIM,
    QUERY_COUNT,
    append_history,
    heading_offset,
    pack_g1_input,
    pack_proprioception,
    projected_gravity,
    root_angular_velocity_body,
)
from bfm_adapters.registry import BFMInstanceConfig, available, register

if TYPE_CHECKING:
    from bfm_adapters.batched_bfm import PreparedDeviceBFM, RobotStateView
    from bfm_adapters.contracts import DevicePDCommand

FAMILY = "sonic"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _SonicConfig:
    root_body: str
    history_steps: tuple[int, ...]
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _SonicConfig:
        required = {
            "root_body",
            "history_steps",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                "sonic family_config keys must be "
                f"{sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("sonic root_body must be non-empty")
        steps = tuple(raw["history_steps"])  # type: ignore[arg-type]
        if steps != tuple(range(HISTORY_LENGTH)):
            raise ValueError(
                f"sonic history_steps must be {tuple(range(HISTORY_LENGTH))}"
            )

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"sonic {name} must contain {ACTION_DIM} finite values"
                )
            return value.copy()

        return cls(
            root_body=root,
            history_steps=steps,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class SonicAdapter:
    """SONIC observation/state semantics without simulator or policy execution."""

    def __init__(self, config: BFMInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"SonicAdapter requires family {FAMILY!r}")
        expected_inputs = {
            "g1_input": (G1_INPUT_DIM,),
            "proprioception": (PROPRIOCEPTION_DIM,),
        }
        actual_inputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_inputs
        }
        if actual_inputs != expected_inputs:
            raise ValueError(f"sonic policy_inputs must be {expected_inputs}")
        actual_outputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_outputs
        }
        if actual_outputs != {"action": (ACTION_DIM,)}:
            raise ValueError("sonic runtime policy_outputs must contain only action[29]")
        if len(config.motion_command_query_offsets) != QUERY_COUNT:
            raise ValueError(
                f"sonic motion_command_query_offsets must contain {QUERY_COUNT} offsets"
            )
        if any(offset < 0 for offset in config.motion_command_query_offsets):
            raise ValueError("sonic motion_command_query_offsets must be non-negative")
        if (
            config.motion_command_query_offsets[0] != 0
            or config.motion_command_query_offsets
            != tuple(sorted(config.motion_command_query_offsets))
        ):
            raise ValueError(
                "sonic motion_command_query_offsets must start at zero and be strictly increasing"
            )
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"sonic requires {ACTION_DIM} policy joints")
        family = _SonicConfig.from_mapping(config.family_config)
        if family.root_body not in config.motion_command_body_names:
            raise ValueError("sonic motion_command bodies must contain root_body")
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
            "root_angular_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, 3), np.float32
            ),
            "joint_position": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "joint_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "previous_actions": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "projected_gravity": np.zeros(
                (batch_size, HISTORY_LENGTH, 3), np.float32
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), np.float32),
            "history_initialized": np.zeros(batch_size, dtype=bool),
            "heading_offset": np.zeros((batch_size, 4), np.float32),
            "heading_initialized": np.zeros(batch_size, dtype=bool),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "sonic expand_state requires one lane and positive batch_size"
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
        if motion_command.query_offsets != self._config.motion_command_query_offsets:
            raise ValueError("sonic motion_command offsets differ from the bfm config")
        batch = state.joint_position.shape[0]
        if (
            motion_command.joint_position.shape[0] != batch
            or bfm_state["history_initialized"].shape != (batch,)
        ):
            raise ValueError("SONIC state/motion_command/bfm-state batch sizes differ")
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _order(motion_command.joint_names, self._config.policy_joint_names)
        state_bodies = _order(state.body_names, self._config.motion_command_body_names)
        guide_bodies = _order(motion_command.body_names, self._config.motion_command_body_names)

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
        reference_joint_velocity = motion_command.joint_velocity[:, :, guide_joints].astype(
            np.float32, copy=False
        )
        reference_root_orientation = motion_command.body_orientation_world[
            :, :, guide_bodies[self._root_index]
        ].astype(np.float32, copy=False)

        next_state = self.clone_state(bfm_state)
        initialized = bfm_state["history_initialized"]
        values = {
            "root_angular_velocity": root_angular_velocity_body(
                robot_root_orientation, robot_root_angular_velocity
            ),
            "joint_position": joint_position
            - self._family.default_joint_position[None],
            "joint_velocity": joint_velocity,
            "previous_actions": bfm_state["last_action"],
            "projected_gravity": projected_gravity(robot_root_orientation),
        }
        for name, value in values.items():
            next_state[name] = append_history(
                bfm_state[name], value, initialized
            )
        next_state["history_initialized"] = np.ones(batch, dtype=bool)

        current_heading = heading_offset(
            robot_root_orientation, reference_root_orientation[:, 0]
        )
        next_state["heading_offset"] = np.where(
            bfm_state["heading_initialized"][:, None],
            bfm_state["heading_offset"],
            current_heading,
        ).astype(np.float32, copy=False)
        next_state["heading_initialized"] = np.ones(batch, dtype=bool)

        inputs = {
            "g1_input": pack_g1_input(
                reference_joint_position,
                reference_joint_velocity,
                reference_root_orientation,
                robot_root_orientation,
                next_state["heading_offset"],
            ),
            "proprioception": pack_proprioception(
                next_state["root_angular_velocity"],
                next_state["joint_position"],
                next_state["joint_velocity"],
                next_state["previous_actions"],
                next_state["projected_gravity"],
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
        action = np.asarray(outputs["action"], dtype=np.float32)
        if action.shape != bfm_state["last_action"].shape:
            raise ValueError("SONIC action batch differs from bfm state")
        command = action_to_pd(
            action,
            policy_joint_names=self._config.policy_joint_names,
            command_joint_names=self._config.command_joint_names,
            default_joint_position=self._family.default_joint_position,
            action_scale=self._family.action_scale,
            proportional_gain=self._family.proportional_gain,
            derivative_gain=self._family.derivative_gain,
        )
        next_state = self.clone_state(bfm_state)
        next_state["last_action"] = action.copy()
        return command, next_state

    def build_device_bfm(
        self,
        policy_path: str | Path,
        state_view: RobotStateView,
        command_view: DevicePDCommand,
    ) -> PreparedDeviceBFM:
        """Build the optional CUDA runtime without importing it on the host path."""

        from bfm_adapters.families.sonic.device import SonicDeviceBFM

        return SonicDeviceBFM(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_sonic(config: BFMInstanceConfig) -> SonicAdapter:
    return SonicAdapter(config)


assert FAMILY in available()
