"""Restore and verify repository-wide external inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

LOCK_RELATIVE = Path("artifacts/lock.v1.json")


@dataclass(frozen=True)
class ArtifactEntry:
    """One immutable file or extracted tree declared by the artifact lock."""

    id: str
    kind: str
    sha256: str | None = None
    bytes: int | None = None
    path: str | None = None
    uri: str | None = None
    cache_path: str | None = None
    extract_to: str | None = None
    tree_root: str | None = None
    tree_files: int | None = None
    tree_bytes: int | None = None
    tree_sha256: str | None = None
    source_id: str | None = None
    transform: str | None = None
    transform_version: int | None = None
    source_url: str | None = None


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_lock(path: str | Path | None = None) -> dict[str, ArtifactEntry]:
    lock_path = Path(path) if path is not None else default_repo_root() / LOCK_RELATIVE
    try:
        payload = json.loads(lock_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"artifact lock not found: {lock_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported artifact lock schema: {lock_path}")
    raw_entries = payload.get("artifacts")
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise ValueError(f"artifact lock has no artifacts: {lock_path}")
    return {
        str(artifact_id): _parse_entry(str(artifact_id), raw)
        for artifact_id, raw in raw_entries.items()
    }


def resolve(
    artifact_id: str,
    *,
    lock: Mapping[str, ArtifactEntry] | None = None,
    root: str | Path | None = None,
) -> ArtifactEntry:
    entries = lock if lock is not None else load_lock(
        None if root is None else Path(root) / LOCK_RELATIVE
    )
    try:
        return entries[artifact_id]
    except KeyError:
        raise KeyError(
            f"unknown artifact id {artifact_id!r}; available: {sorted(entries)}"
        ) from None


def verify(entry: ArtifactEntry, *, root: str | Path | None = None) -> Path:
    """Verify an installed artifact and return its file or tree root."""

    repo_root = Path(root) if root is not None else default_repo_root()
    if entry.kind == "file":
        path = _repo_path(repo_root, _required(entry.path, entry, "path"))
        _verify_file(path, entry)
        return path
    if entry.kind in {"tar.gz", "tree"}:
        tree = _repo_path(repo_root, _required(entry.tree_root, entry, "tree_root"))
        _verify_tree(tree, entry)
        return tree
    raise ValueError(f"unsupported artifact kind {entry.kind!r}")


def ensure_artifact(
    artifact_id: str,
    *,
    root: str | Path | None = None,
    lock: Mapping[str, ArtifactEntry] | None = None,
) -> Path:
    """Verify a local artifact, or restore an explicitly downloadable input."""

    repo_root = Path(root) if root is not None else default_repo_root()
    entry = resolve(artifact_id, root=repo_root, lock=lock)
    try:
        return verify(entry, root=repo_root)
    except (FileNotFoundError, ValueError) as exc:
        if entry.kind == "tar.gz":
            cache = _repo_path(repo_root, _required(entry.cache_path, entry, "cache_path"))
            if _file_matches(cache, entry):
                _install_tree(cache, entry, repo_root)
                return verify(entry, root=repo_root)
        if entry.uri is None:
            if entry.source_id is not None:
                instruction = (
                    f"Install source artifact {entry.source_id!r}, then run "
                    f"'bfm-ttc prepare-bfm {entry.id}'."
                )
            else:
                instruction = "Install the exact locked artifact locally."
                if entry.source_url is not None:
                    instruction += f" Official source: {entry.source_url}."
            raise type(exc)(
                f"{exc}. {instruction} See docs/assets.md for paths and preparation status."
            ) from exc

    if entry.kind == "file":
        target = _repo_path(repo_root, _required(entry.path, entry, "path"))
        _download(entry.uri, target)
    elif entry.kind == "tar.gz":
        cache = _repo_path(
            repo_root, _required(entry.cache_path, entry, "cache_path")
        )
        if not _file_matches(cache, entry):
            _download(entry.uri, cache)
        _verify_file(cache, entry)
        _install_tree(cache, entry, repo_root)
    else:
        raise ValueError(f"unsupported artifact kind {entry.kind!r}")
    return verify(entry, root=repo_root)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode())
        total_bytes += path.stat().st_size
    return len(files), total_bytes, digest.hexdigest()


def _verify_file(path: Path, entry: ArtifactEntry) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"artifact {entry.id!r} missing at {path}")
    if path.stat().st_size != entry.bytes:
        raise ValueError(f"artifact {entry.id!r} size mismatch")
    if sha256_file(path) != entry.sha256:
        raise ValueError(f"artifact {entry.id!r} sha256 mismatch")


def _file_matches(path: Path, entry: ArtifactEntry) -> bool:
    try:
        _verify_file(path, entry)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _verify_tree(path: Path, entry: ArtifactEntry) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"artifact tree {entry.id!r} missing at {path}")
    expected = (entry.tree_files, entry.tree_bytes, entry.tree_sha256)
    if any(value is None for value in expected):
        raise ValueError(f"artifact {entry.id!r} has incomplete tree metadata")
    if _tree_stats(path) != expected:
        raise ValueError(f"artifact tree {entry.id!r} digest mismatch")


def _download(uri: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        downloaded = Path(temporary) / target.name
        with urllib.request.urlopen(uri) as source, downloaded.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        os.replace(downloaded, target)


def _install_tree(archive: Path, entry: ArtifactEntry, repo_root: Path) -> None:
    extract_to = _repo_path(
        repo_root, _required(entry.extract_to, entry, "extract_to")
    )
    tree_root = _repo_path(
        repo_root, _required(entry.tree_root, entry, "tree_root")
    )
    try:
        relative_tree = tree_root.relative_to(extract_to)
    except ValueError as exc:
        raise ValueError("tree_root must be inside extract_to") from exc
    extract_to.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=extract_to.parent) as temporary:
        temporary_root = Path(temporary)
        _safe_extract(archive, temporary_root)
        candidate = temporary_root / relative_tree
        _verify_tree(candidate, entry)
        tree_root.parent.mkdir(parents=True, exist_ok=True)
        if tree_root.is_symlink() or tree_root.is_file():
            tree_root.unlink()
        elif tree_root.exists():
            shutil.rmtree(tree_root)
        shutil.move(str(candidate), tree_root)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member: {member.name}")
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(destination, filter="data")
        else:  # pragma: no cover - Python 3.10 compatibility
            bundle.extractall(destination)


def _repo_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes repository root: {relative}")
    return path


def _required(value: str | None, entry: ArtifactEntry, field: str) -> str:
    if not value:
        raise ValueError(f"artifact {entry.id!r} requires {field}")
    return value


def _parse_entry(artifact_id: str, raw: Any) -> ArtifactEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"artifact {artifact_id!r} must be a mapping")
    try:
        entry = ArtifactEntry(
            id=artifact_id,
            kind=str(raw["kind"]),
            path=_optional_str(raw.get("path")),
            uri=_optional_str(raw.get("uri")),
            cache_path=_optional_str(raw.get("cache_path")),
            sha256=_optional_str(raw.get("sha256")),
            bytes=_optional_int(raw.get("bytes")),
            extract_to=_optional_str(raw.get("extract_to")),
            tree_root=_optional_str(raw.get("tree_root")),
            tree_files=_optional_int(raw.get("tree_files")),
            tree_bytes=_optional_int(raw.get("tree_bytes")),
            tree_sha256=_optional_str(raw.get("tree_sha256")),
            source_id=_optional_str(raw.get("source_id")),
            transform=_optional_str(raw.get("transform")),
            transform_version=_optional_int(raw.get("transform_version")),
            source_url=_optional_str(raw.get("source_url")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid artifact entry {artifact_id!r}") from exc
    if entry.kind == "tree":
        valid = (
            entry.tree_root is not None
            and entry.tree_files is not None and entry.tree_files >= 0
            and entry.tree_bytes is not None and entry.tree_bytes >= 0
            and entry.tree_sha256 is not None and len(entry.tree_sha256) == 64
            and entry.uri is None
        )
    else:
        valid = (
            entry.kind in {"file", "tar.gz"}
            and entry.sha256 is not None and len(entry.sha256) == 64
            and entry.bytes is not None and entry.bytes >= 0
        )
    if not valid:
        raise ValueError(f"invalid artifact entry {artifact_id!r}")
    return entry


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
