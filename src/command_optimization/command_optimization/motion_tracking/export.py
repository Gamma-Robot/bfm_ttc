"""Export one completed trajectory without rerunning dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from command_optimization.motion_tracking.any4hdmi import ROOT_QPOS_NAMES


@dataclass(frozen=True)
class OutputConfig:
    formats: tuple[str, ...] = ()
    trajectories: tuple[str, ...] = ("execution",)

    def __post_init__(self) -> None:
        for name, choices in (
            ("formats", {"npz", "any4hdmi"}),
            ("trajectories", {"execution", "command"}),
        ):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or value not in choices for value in values
            ):
                raise ValueError(f"output.{name} must be a list drawn from {sorted(choices)}")
            if len(set(values)) != len(values):
                raise ValueError(f"output.{name} must not contain duplicates")
            object.__setattr__(self, name, tuple(values))
        if self.formats and not self.trajectories:
            raise ValueError("output.trajectories must not be empty when exporting data")


def export_motion(
    destination: Path,
    *,
    trajectory: str,
    qpos: np.ndarray,
    valid: np.ndarray,
    joint_names: tuple[str, ...],
    control_period: float,
    mjcf: str,
    formats: tuple[str, ...],
) -> None:
    """Include the reset frame at t=0; never label padded/failed frames as motion."""
    OutputConfig(formats=formats, trajectories=(trajectory,))
    values = np.array(qpos, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 7 + len(joint_names) or len(values) < 2:
        raise ValueError("export qpos must contain reset plus motion frames in canonical order")
    if np.shape(valid) != (len(values),) or not np.asarray(valid, dtype=bool).all():
        raise ValueError("cannot export an incomplete trajectory; inspect paired_traces.npz")
    if not np.isfinite(values).all() or not np.isfinite(control_period) or control_period <= 0:
        raise ValueError("export requires finite qpos and a positive control period")
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("export root quaternions must be normalized")
    values[:, 3:7] /= norms[:, None]
    names = (*ROOT_QPOS_NAMES, *joint_names)
    destination.mkdir(parents=True, exist_ok=True)
    if "npz" in formats:
        np.savez_compressed(
            destination / f"{trajectory}.npz", qpos=values,
            timeline=np.arange(len(values), dtype=np.float64) * control_period,
            qpos_names=np.asarray(names), trajectory=np.asarray(trajectory),
        )
    if "any4hdmi" in formats:
        root = destination / f"{trajectory}.any4hdmi"
        (root / "motions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "motions" / f"{trajectory}.npz", qpos=values.astype(np.float32)
        )
        manifest = {
            "format_version": 2, "dataset_name": f"bfm-ttc-{trajectory}",
            "mjcf": mjcf, "motions_subdir": "motions", "timestep": control_period,
            "qpos_dim": len(names), "qpos_names": names, "num_motions": 1,
            "source": {"tool": "bfm-ttc", "trajectory": trajectory},
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
