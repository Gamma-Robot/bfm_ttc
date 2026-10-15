"""Complete application-owned state for synchronized simulation lanes."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from bfm_adapters import (
    MOTION_COMMAND_ARRAY_FIELDS as _COMMAND_ARRAYS,
    MotionCommand,
    PDCommand,
    BFMAdapter,
    BFMStateBatch,
)


_PHYSICAL_RESET_DIGEST_DOMAIN = b"bfm-ttc.physical-reset.v1\0"


def _physical(value: np.ndarray, field: str) -> None:
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{field} must be a floating-point numpy array")
    if value.ndim != 2 or value.shape[0] < 1:
        raise ValueError(
            f"{field} must have shape (batch, dim) with batch >= 1, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{field} must be finite")


def _immutable_copy(value: np.ndarray) -> np.ndarray:
    copy = np.array(value, order="C", copy=True)
    copy.flags.writeable = False
    return copy


def _immutable_motion_command(value: MotionCommand) -> MotionCommand:
    arrays = {name: _immutable_copy(getattr(value, name)) for name in _COMMAND_ARRAYS}
    return MotionCommand(
        query_offsets=value.query_offsets,
        joint_names=value.joint_names,
        body_names=value.body_names,
        **arrays,
    )


def physical_state_digest(qpos: np.ndarray, qvel: np.ndarray) -> str:
    """Return the v1 content digest of a physical reset state.

    BFM state, commands, motion_command, and application counters are excluded by
    construction. Arrays are encoded as little-endian float64 so equivalent
    floating-point values have one identity across dtype and memory layout.
    """

    _physical(qpos, "qpos")
    _physical(qvel, "qvel")
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError("qpos and qvel must share one batch size")

    digest = hashlib.sha256(_PHYSICAL_RESET_DIGEST_DOMAIN)
    for name, value in (("qpos", qpos), ("qvel", qvel)):
        encoded_name = name.encode("ascii")
        canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", canonical.ndim))
        digest.update(struct.pack(f"<{canonical.ndim}Q", *canonical.shape))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ExecutionState:
    """Complete reset or continuation point for synchronized execution lanes.

    Physical arrays are `(B, nq)` / `(B, nv)` blocks owned by `command_optimization`;
    `bfm_state` is an opaque family-owned value that is captured and
    restored only through the `BFMAdapter`, never inspected here.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    control_step: int
    time: float
    motion_command_index: int
    command: PDCommand | None
    bfm_state: BFMStateBatch
    motion_command_history: MotionCommand | None = None
    qacc_warmstart: np.ndarray | None = None

    def __post_init__(self) -> None:
        _physical(self.qpos, "qpos")
        _physical(self.qvel, "qvel")
        if self.qpos.shape[0] != self.qvel.shape[0]:
            raise ValueError(
                "qpos and qvel must share one batch size, got "
                f"{self.qpos.shape[0]} and {self.qvel.shape[0]}"
            )
        if self.qacc_warmstart is not None:
            _physical(self.qacc_warmstart, "qacc_warmstart")
            if self.qacc_warmstart.shape != self.qvel.shape:
                raise ValueError("qacc_warmstart must have the same shape as qvel")
            object.__setattr__(
                self, "qacc_warmstart", _immutable_copy(self.qacc_warmstart)
            )
        if self.control_step < 0:
            raise ValueError(f"control_step must be >= 0, got {self.control_step}")
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError(f"time must be finite and >= 0, got {self.time}")
        if self.motion_command_index < 0:
            raise ValueError(f"motion_command_index must be >= 0, got {self.motion_command_index}")
        history = self.motion_command_history
        if history is not None:
            history_steps = len(history.query_offsets)
            if history.query_offsets != tuple(range(-history_steps, 0)):
                raise ValueError(
                    "motion_command_history must use a dense negative query grid"
                )
            if history.joint_position.shape[0] != self.qpos.shape[0]:
                raise ValueError("motion_command_history batch size must match qpos")
            for field in _COMMAND_ARRAYS:
                if not np.isfinite(getattr(history, field)).all():
                    raise ValueError(f"motion_command_history {field} must be finite")
            object.__setattr__(self, "motion_command_history", _immutable_motion_command(history))

    @classmethod
    def capture(
        cls,
        qpos: np.ndarray,
        qvel: np.ndarray,
        control_step: int,
        time: float,
        motion_command_index: int,
        command: PDCommand | None,
        bfm_state: BFMStateBatch,
        adapter: BFMAdapter,
        motion_command_history: MotionCommand | None = None,
        qacc_warmstart: np.ndarray | None = None,
    ) -> ExecutionState:
        """Capture with defensive physical copies and adapter-cloned state."""
        return cls(
            qpos=_immutable_copy(qpos),
            qvel=_immutable_copy(qvel),
            control_step=control_step,
            time=time,
            motion_command_index=motion_command_index,
            command=command,
            bfm_state=adapter.clone_state(bfm_state),
            motion_command_history=motion_command_history,
            qacc_warmstart=qacc_warmstart,
        )

    def restore_bfm_state(self, adapter: BFMAdapter) -> BFMStateBatch:
        """Return an independent bfm state; every restore yields a new clone."""
        return adapter.clone_state(self.bfm_state)
