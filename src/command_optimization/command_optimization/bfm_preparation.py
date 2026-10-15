"""Rebuild lock-declared derived bfm artifacts from verified sources."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from bfm_adapters.preparation import apply_transform
from command_optimization.artifacts import (
    LOCK_RELATIVE,
    default_repo_root,
    ensure_artifact,
    load_lock,
    resolve,
    sha256_file,
    verify,
)


def prepare_bfm_artifact(
    artifact_id: str,
    *,
    root: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> Path:
    """Build and verify one derived ONNX file without mutating its lock."""
    import onnx

    repo = default_repo_root() if root is None else Path(root).resolve()
    path = repo / LOCK_RELATIVE if lock_path is None else Path(lock_path)
    entries = load_lock(path)
    entry = resolve(artifact_id, lock=entries)
    if (
        entry.kind != "file"
        or entry.path is None
        or entry.source_id is None
        or entry.transform is None
        or entry.transform_version is None
    ):
        raise ValueError(f"artifact {artifact_id!r} is not a complete derived recipe")

    source = ensure_artifact(entry.source_id, root=repo, lock=entries)
    output = (repo / entry.path).resolve()
    if output != repo and repo not in output.parents:
        raise ValueError(f"artifact {artifact_id!r} path escapes repository root")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    apply_transform(entry.transform, entry.transform_version, model)

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        onnx.save(model, str(candidate))
        if candidate.stat().st_size != entry.bytes or sha256_file(candidate) != entry.sha256:
            raise ValueError(
                f"derived artifact {artifact_id!r} does not match its locked identity"
            )
        os.replace(candidate, output)
    return verify(entry, root=repo)
