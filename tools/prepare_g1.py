#!/usr/bin/env python3
"""Prepare the pinned G1 robot from an official motion_tracking checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "robots" / "g1"


def _verified_bytes(root: Path, entry: dict) -> bytes:
    path = root / entry["path"]
    data = path.read_bytes()
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"G1 asset identity mismatch: {path}")
    return data


def prepare_g1(source_checkout: Path, destination: Path) -> dict:
    """Verify official source bytes and install the minimal historical model.

    The destination must not exist. No network access or Git invocation is used.
    The bundled XML retains the experiment's physical model; source XML and
    mesh hashes identify the official revision used for reconstruction.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"G1 destination already exists: {destination}")
    manifest = json.loads((ASSET_DIRECTORY / "manifest.json").read_text())
    source_robot = source_checkout / manifest["source_robot_directory"]
    _verified_bytes(source_robot, manifest["source_xml"])
    _verified_bytes(source_checkout, manifest["source_license"])
    files = {entry["path"]: _verified_bytes(source_robot, entry) for entry in manifest["meshes"]}
    files.update({entry["path"]: _verified_bytes(ASSET_DIRECTORY, entry) for entry in manifest["bundled_files"]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g1-prepare-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "robot"
        staged.mkdir()
        for relative, data in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        staged.rename(destination)
    digest = hashlib.sha256()
    for relative, data in sorted(files.items()):
        digest.update(f"{hashlib.sha256(data).hexdigest()}  {relative}\n".encode())
    return {
        "tree_files": len(files),
        "tree_bytes": sum(len(data) for data in files.values()),
        "tree_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkout", type=Path, help="Official motion_tracking checkout root")
    parser.add_argument("destination", type=Path, help="New directory for prepared XML and meshes")
    arguments = parser.parse_args()
    print(json.dumps(prepare_g1(arguments.source_checkout, arguments.destination), indent=2))


if __name__ == "__main__":
    main()
