"""Dense motion_command slices and immutable reference_motion context."""

from __future__ import annotations

import numpy as np

from bfm_adapters import MOTION_COMMAND_ARRAY_FIELDS, MotionCommand


def take_motion_command(motion_command: MotionCommand, indices: np.ndarray) -> MotionCommand:
    """Select frames into a dense zero-based query grid, preserving array dtype."""
    return MotionCommand(
        query_offsets=tuple(range(len(indices))),
        joint_names=motion_command.joint_names,
        body_names=motion_command.body_names,
        **{
            name: np.asarray(getattr(motion_command, name))[:, indices]
            for name in MOTION_COMMAND_ARRAY_FIELDS
        },
    )


def motion_command_window(
    motion_command: MotionCommand, start: int, num_steps: int
) -> MotionCommand:
    """Select a fixed-size window, repeating the final frame past the end."""
    indices = np.minimum(
        np.arange(start, start + num_steps), len(motion_command.query_offsets) - 1
    )
    return take_motion_command(motion_command, indices)


def reference_right_context(
    motion_command: MotionCommand, start: int, num_steps: int
) -> MotionCommand | None:
    """Copy edge-padded reference_motion context into owned, read-only arrays."""
    if num_steps == 0:
        return None
    context = motion_command_window(motion_command, start, num_steps)
    arrays = {}
    for name in MOTION_COMMAND_ARRAY_FIELDS:
        value = np.array(getattr(context, name), copy=True, order="C")
        value.flags.writeable = False
        arrays[name] = value
    return MotionCommand(
        query_offsets=context.query_offsets,
        joint_names=context.joint_names,
        body_names=context.body_names,
        **arrays,
    )
