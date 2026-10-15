"""Release identity derived from distributed source, without repository metadata.

Hash Python and package TOML files below ``src/`` plus the root project and
locked dependencies. Configurations and assets have their own persisted hashes.
Hidden files, bytecode, build products, and runtime outputs are not source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

RELEASE_VERSION = "0.1.0"
_EXCLUDED = frozenset({"__pycache__", "build", "dist", "outputs"})


@dataclass(frozen=True)
class SourceIdentity:
    release_version: str
    source_sha256: str


def source_identity(root: str | Path) -> SourceIdentity:
    """Hash sorted relative source paths and bytes with unambiguous framing.

    The caller supplies the source distribution root; no parent traversal or Git
    discovery occurs. Missing source is an error, never a guessed identity.
    """
    root = Path(root)
    source = root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    paths = [
        path for path in source.rglob("*")
        if path.suffix in (".py", ".toml")
        and not any(part.startswith(".") or part in _EXCLUDED
                    for part in path.relative_to(root).parts)
        and path.is_file()
    ]
    if not any(path.suffix == ".py" for path in paths):
        raise ValueError(f"source directory has no Python files: {source}")
    project = root / "pyproject.toml"
    if not project.is_file():
        raise FileNotFoundError(f"source project metadata does not exist: {project}")
    paths.append(project)
    lock = root / "uv.lock"
    if lock.is_file():
        paths.append(lock)
    digest = hashlib.sha256(b"bfm-ttc-source-v1\0")
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        source_parents = (parent for parent in path.parents if parent != root and root in parent.parents)
        if path.is_symlink() or any(parent.is_symlink() for parent in source_parents):
            raise ValueError(f"source identity does not accept symbolic links: {path}")
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return SourceIdentity(RELEASE_VERSION, digest.hexdigest())
