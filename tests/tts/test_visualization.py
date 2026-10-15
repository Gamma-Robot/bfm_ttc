from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import command_optimization.visualization.render as rendering
from command_optimization.visualization import (
    ComparisonView,
    RenderConfig,
    RenderBindings,
    RenderTrack,
    SingleView,
    render_mp4,
)


def _bindings() -> RenderBindings:
    return RenderBindings(
        release_version="0.1.0",
        source_sha256="a" * 64,
        reference_sha256="1" * 64,
        trace_sha256="2" * 64,
        report_sha256="3" * 64,
        mjcf_sha256="4" * 64,
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    mjcf = tmp_path / "g1.xml"
    mjcf.write_text("<mujoco/>")
    reference_motion = np.zeros((6, 7), dtype=np.float64)
    return {
        "mjcf_path": mjcf,
        "timeline": np.arange(6, dtype=np.float64) * 0.02,
        "reference_qpos": reference_motion,
        "view": ComparisonView(
            left=RenderTrack("baseline", reference_motion[:3].copy()),
            right=RenderTrack("optimized", reference_motion[:5].copy()),
        ),
        "output_path": tmp_path / "comparison.mp4",
        "config": RenderConfig(fps=25, width=8, height=4),
        "bindings": _bindings(),
    }


def test_render_validates_and_returns_bound_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _inputs(tmp_path)
    captured: dict[str, object] = {}

    def fake_frames(*args: object) -> object:
        captured["frame_indices"] = args[5]
        yield np.zeros((4, 8, 3), dtype=np.uint8)

    def fake_encode(
        frames: object,
        output_path: Path,
        *,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        captured["shape"] = next(iter(frames)).shape  # type: ignore[arg-type]
        captured["encoding"] = (fps, width, height)
        output_path.write_bytes(b"mp4-evidence")

    monkeypatch.setattr(rendering, "_render_frames", fake_frames)
    monkeypatch.setattr(rendering, "_encode_mp4", fake_encode)
    provenance = render_mp4(**arguments)  # type: ignore[arg-type]

    assert captured == {
        "frame_indices": (0, 2, 4, 5),
        "shape": (4, 8, 3),
        "encoding": (25, 8, 4),
    }
    assert provenance.renderer_contract_version == "bfm-ttc-render-v2"
    assert provenance.frame_indices == (0, 2, 4, 5)
    assert provenance.release_version == "0.1.0"
    assert provenance.source_sha256 == "a" * 64
    assert provenance.reference_sha256 == "1" * 64
    assert provenance.config.width == 8
    assert provenance.view == "comparison"
    assert provenance.track_labels == ("baseline", "optimized")
    assert provenance.output_sha256 == hashlib.sha256(b"mp4-evidence").hexdigest()


def test_single_view_explicitly_renders_one_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _inputs(tmp_path)
    comparison = arguments["view"]
    assert isinstance(comparison, ComparisonView)
    arguments["view"] = SingleView(comparison.right)

    def fake_frames(*_args: object) -> object:
        yield np.zeros((4, 8, 3), dtype=np.uint8)

    def fake_encode(frames: object, output_path: Path, **_kwargs: object) -> None:
        next(iter(frames))  # type: ignore[arg-type]
        output_path.write_bytes(b"single")

    monkeypatch.setattr(rendering, "_render_frames", fake_frames)
    monkeypatch.setattr(rendering, "_encode_mp4", fake_encode)

    provenance = render_mp4(**arguments)  # type: ignore[arg-type]

    assert provenance.view == "single"
    assert provenance.track_labels == ("optimized",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timeline", np.array([0.0, 0.02, 0.01]), "strictly increasing"),
        ("reference_qpos", np.zeros((5, 7)), "length must equal"),
        ("baseline_qpos", np.zeros((2, 8)), "same nq"),
        ("optimized_qpos", np.full((2, 7), np.nan), "finite"),
    ],
)
def test_render_rejects_inconsistent_persisted_arrays(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    arguments = _inputs(tmp_path)
    if field in {"baseline_qpos", "optimized_qpos"}:
        view = arguments["view"]
        assert isinstance(view, ComparisonView)
        arguments["view"] = ComparisonView(
            left=RenderTrack(
                "baseline", value if field == "baseline_qpos" else view.left.qpos
            ),
            right=RenderTrack(
                "optimized", value if field == "optimized_qpos" else view.right.qpos
            ),
        )
    else:
        arguments[field] = value
    with pytest.raises(ValueError, match=message):
        render_mp4(**arguments)  # type: ignore[arg-type]


def test_render_does_not_overwrite_evidence(tmp_path: Path) -> None:
    arguments = _inputs(tmp_path)
    output = arguments["output_path"]
    assert isinstance(output, Path)
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_mp4(**arguments)  # type: ignore[arg-type]


def test_render_reports_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _inputs(tmp_path)
    monkeypatch.setattr(rendering.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        render_mp4(**arguments)  # type: ignore[arg-type]


def test_config_and_bindings_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="width must be even"):
        RenderConfig(width=1279)
    with pytest.raises(ValueError, match="height must be even"):
        RenderConfig(height=719)
    with pytest.raises(ValueError, match="camera_mode must be one of"):
        RenderConfig(camera_mode="orbit")
    with pytest.raises(TypeError, match="show_root_error must be bool"):
        RenderConfig(show_root_error=1)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        RenderBindings("0.1.0", "a" * 64, "A" * 64, "2" * 64, "3" * 64, "4" * 64)


def test_config_accepts_every_declared_camera_mode() -> None:
    for mode in rendering.CAMERA_MODES:
        assert RenderConfig(camera_mode=mode).camera_mode == mode


def test_frame_indices_use_nearest_recorded_pose() -> None:
    timeline = np.arange(11, dtype=np.float64) * 0.02
    assert rendering._frame_indices(timeline, fps=25) == (0, 2, 4, 6, 8, 10)
