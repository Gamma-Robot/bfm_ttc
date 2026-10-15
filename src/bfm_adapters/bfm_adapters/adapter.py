"""Interface implemented by each bfm family."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bfm_adapters.contracts import (
    MotionCommand,
    NamedTensors,
    PDCommand,
    RobotState,
    BFMStateBatch,
)


@runtime_checkable
class BFMAdapter(Protocol):
    """Pure bfm semantics; policy execution and physics live elsewhere."""

    def initial_state(self, batch_size: int) -> BFMStateBatch:
        """Create family-owned state for independent rollout lanes."""

    def clone_state(self, state: BFMStateBatch) -> BFMStateBatch:
        """Return an independent copy of family-owned state."""

    def expand_state(
        self, state: BFMStateBatch, batch_size: int
    ) -> BFMStateBatch:
        """Copy one execution-lane state into independent rollout lanes."""

    def build_policy_inputs(
        self,
        state: RobotState,
        motion_command: MotionCommand,
        bfm_state: BFMStateBatch,
    ) -> tuple[NamedTensors, BFMStateBatch]:
        """Build named inputs and return state advanced by observation history."""

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        bfm_state: BFMStateBatch,
    ) -> tuple[PDCommand, BFMStateBatch]:
        """Map named outputs to a command and the next family-owned state."""
