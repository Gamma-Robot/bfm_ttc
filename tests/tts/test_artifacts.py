"""CPU tests for repository-wide artifact restore and verification."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from command_optimization.artifacts import ensure_artifact, load_lock, resolve, verify


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_lock(root: Path, artifacts: dict[str, object]) -> None:
    path = root / "artifacts" / "lock.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}))


def test_lock_covers_policy_and_robot() -> None:
    entries = load_lock()
    source = resolve("heft-pmg-policy", lock=entries)
    prepared = resolve("heft-pmg-policy.batched", lock=entries)
    robot = resolve("g1-paper-model", lock=entries)

    assert source.kind == "file"
    assert source.path == "artifacts/store/heft/pmg/policy.onnx"
    assert source.sha256 == (
        "ae22fd44c9d073e98a19b36e88d6a598119a2a900e9dba410ffea77a05ce784d"
    )
    assert source.uri is None
    assert all(entry.uri is None or not entry.uri.startswith("bos:") for entry in entries.values())
    assert prepared.source_id == source.id
    assert prepared.transform == "relax_leading"
    assert prepared.transform_version == 1
    assert robot.kind == "tree"
    assert robot.tree_files == 37
    assert robot.tree_sha256 == (
        "101ee02653225b276abe709efe74a2bdae0436a438dd7a8a5b86719e70c08c11"
    )


def test_verify_rejects_tampered_file(tmp_path: Path) -> None:
    _write_lock(
        tmp_path,
        {
            "demo": {
                "kind": "file",
                "path": "store/demo.bin",
                "sha256": "0" * 64,
                "bytes": 4,
            }
        },
    )
    target = tmp_path / "store" / "demo.bin"
    target.parent.mkdir()
    target.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify(resolve("demo", root=tmp_path), root=tmp_path)


def test_ensure_restores_file_uri(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"locked")
    _write_lock(
        tmp_path,
        {
            "demo": {
                "kind": "file",
                "uri": source.as_uri(),
                "path": "store/demo.bin",
                "sha256": _digest(b"locked"),
                "bytes": 6,
            }
        },
    )

    restored = ensure_artifact("demo", root=tmp_path)

    assert restored.read_bytes() == b"locked"


@pytest.mark.parametrize("prepared", [False, True])
def test_missing_local_artifact_explains_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: bool
) -> None:
    entry = {
        "kind": "file",
        "path": "store/policy.onnx",
        "sha256": "0" * 64,
        "bytes": 1,
        "source_url": "https://example.org/official-model",
    }
    if prepared:
        entry.update(source_id="source", transform="relax_leading", transform_version=1)
    _write_lock(tmp_path, {"policy": entry})

    def unexpected_download(*args, **kwargs):
        pytest.fail("source_url is documentation, not a download URI")

    monkeypatch.setattr("command_optimization.artifacts._download", unexpected_download)
    expected = "prepare-bfm policy" if prepared else "https://example.org/official-model"
    with pytest.raises(FileNotFoundError, match=expected) as error:
        ensure_artifact("policy", root=tmp_path)
    assert "store/policy.onnx" in str(error.value)
    assert "docs/assets.md" in str(error.value)


@pytest.mark.parametrize("local_archive", [False, True])
def test_ensure_extracts_and_verifies_tree(tmp_path: Path, local_archive: bool) -> None:
    source = tmp_path / "tree.tar.gz"
    content = b"model"
    with tarfile.open(source, "w:gz") as bundle:
        member = tarfile.TarInfo("bundle/model.xml")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    tree_digest = hashlib.sha256(
        f"{_digest(content)}  model.xml\n".encode()
    ).hexdigest()
    _write_lock(
        tmp_path,
        {
            "tree": {
                "kind": "tar.gz",
                "uri": None if local_archive else source.as_uri(),
                "cache_path": "artifacts/cache/tree.tar.gz",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
                "extract_to": "artifacts/store/tree",
                "tree_root": "artifacts/store/tree/bundle",
                "tree_files": 1,
                "tree_bytes": len(content),
                "tree_sha256": tree_digest,
            }
        },
    )
    if local_archive:
        cached = tmp_path / "artifacts/cache/tree.tar.gz"
        cached.parent.mkdir()
        cached.write_bytes(source.read_bytes())

    tree = ensure_artifact("tree", root=tmp_path)

    assert tree == (tmp_path / "artifacts/store/tree/bundle").resolve()
    assert (tree / "model.xml").read_bytes() == content


def test_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    source = tmp_path / "bad.tar.gz"
    with tarfile.open(source, "w:gz") as bundle:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    _write_lock(
        tmp_path,
        {
            "bad": {
                "kind": "tar.gz",
                "uri": source.as_uri(),
                "cache_path": "artifacts/cache/bad.tar.gz",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
                "extract_to": "artifacts/store/bad",
                "tree_root": "artifacts/store/bad/bundle",
                "tree_files": 1,
                "tree_bytes": 1,
                "tree_sha256": "0" * 64,
            }
        },
    )

    with pytest.raises(ValueError, match="escapes extraction root"):
        ensure_artifact("bad", root=tmp_path)


def test_prepared_tree_verifies_without_an_archive(tmp_path: Path) -> None:
    content = b"model"
    tree = tmp_path / "store/robot"
    tree.mkdir(parents=True)
    (tree / "model.xml").write_bytes(content)
    digest = _digest(f"{_digest(content)}  model.xml\n".encode())
    _write_lock(tmp_path, {"robot": {
        "kind": "tree", "tree_root": "store/robot",
        "tree_files": 1, "tree_bytes": len(content), "tree_sha256": digest,
        "source_url": "https://example.org/official-robot",
    }})
    assert ensure_artifact("robot", root=tmp_path) == tree
    (tree / "model.xml").write_bytes(b"other")
    with pytest.raises(ValueError, match="digest mismatch"):
        ensure_artifact("robot", root=tmp_path)
    (tree / "model.xml").unlink()
    tree.rmdir()
    with pytest.raises(FileNotFoundError, match="official-robot"):
        ensure_artifact("robot", root=tmp_path)
