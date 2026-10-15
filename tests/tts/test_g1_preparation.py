from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def _entry(path: str, data: bytes) -> dict:
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = Path(__file__).resolve().parents[2] / "tools" / "prepare_g1.py"
    spec = importlib.util.spec_from_file_location("prepare_g1_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundled = tmp_path / "bundled"
    source = tmp_path / "official"
    robot = source / "robot"
    (robot / "meshes").mkdir(parents=True)
    bundled.mkdir()
    xml = b'<mujoco><asset><mesh name="pelvis" file="pelvis.STL"/></asset></mujoco>\n'
    upstream = b'<mujoco model="upstream"/>\n'
    mesh = b"fixture mesh bytes"
    license_text = b"fixture upstream license\n"
    (robot / "g1.xml").write_bytes(upstream)
    (robot / "meshes" / "pelvis.STL").write_bytes(mesh)
    (robot / "meshes" / "unused.STL").write_bytes(b"unused")
    (source / "LICENSE").write_bytes(license_text)
    (bundled / "g1-mode_13_15.xml").write_bytes(xml)
    (bundled / "LICENSE").write_bytes(license_text)
    manifest = {
        "source_robot_directory": "robot",
        "source_xml": _entry("g1.xml", upstream),
        "source_license": _entry("LICENSE", license_text),
        "meshes": [_entry("meshes/pelvis.STL", mesh)],
        "bundled_files": [_entry("g1-mode_13_15.xml", xml), _entry("LICENSE", license_text)],
    }
    (bundled / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "ASSET_DIRECTORY", bundled)
    return module, source, tmp_path / "prepared", bundled


def test_preparation_installs_only_pinned_files_and_reports_tree_identity(preparation) -> None:
    module, source, destination, bundled = preparation
    result = module.prepare_g1(source, destination)
    assert sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()) == [
        "LICENSE", "g1-mode_13_15.xml", "meshes/pelvis.STL",
    ]
    assert (destination / "g1-mode_13_15.xml").read_bytes() == (bundled / "g1-mode_13_15.xml").read_bytes()
    digest = hashlib.sha256()
    size = 0
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        data = path.read_bytes()
        digest.update(f"{hashlib.sha256(data).hexdigest()}  {path.relative_to(destination).as_posix()}\n".encode())
        size += len(data)
    assert result == {"tree_files": 3, "tree_bytes": size, "tree_sha256": digest.hexdigest()}


@pytest.mark.parametrize("damaged", ["source_xml", "mesh", "license", "bundled_xml"])
def test_preparation_rejects_changed_assets_before_installing(preparation, damaged: str) -> None:
    module, source, destination, bundled = preparation
    paths = {
        "source_xml": source / "robot/g1.xml", "mesh": source / "robot/meshes/pelvis.STL",
        "license": source / "LICENSE", "bundled_xml": bundled / "g1-mode_13_15.xml",
    }
    paths[damaged].write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        module.prepare_g1(source, destination)
    assert not destination.exists()


def test_preparation_reports_missing_mesh(preparation) -> None:
    module, source, destination, _ = preparation
    (source / "robot/meshes/pelvis.STL").unlink()
    with pytest.raises(FileNotFoundError):
        module.prepare_g1(source, destination)
    assert not destination.exists()


def test_preparation_preserves_existing_destination(preparation) -> None:
    module, source, destination, _ = preparation
    destination.mkdir()
    sentinel = destination / "existing"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        module.prepare_g1(source, destination)
    assert sentinel.read_bytes() == b"keep"


def test_preparation_cli_requires_explicit_paths_and_emits_identity(preparation, monkeypatch, capsys) -> None:
    module, source, destination, _ = preparation
    monkeypatch.setattr(sys, "argv", ["prepare_g1.py", str(source), str(destination)])
    module.main()
    assert json.loads(capsys.readouterr().out)["tree_files"] == 3


def test_bundled_robot_manifest_covers_exact_xml_dependencies() -> None:
    bundled = Path(__file__).resolve().parents[2] / "assets/robots/g1"
    manifest = json.loads((bundled / "manifest.json").read_text())
    xml = ET.parse(bundled / "g1-mode_13_15.xml").getroot()
    assert xml.find("compiler").get("meshdir") == "meshes"
    required = {"meshes/" + mesh.get("file") for mesh in xml.findall("asset/mesh")}
    assert {entry["path"] for entry in manifest["meshes"]} == required
    for entry in manifest["bundled_files"]:
        assert _entry(entry["path"], (bundled / entry["path"]).read_bytes()) == entry
