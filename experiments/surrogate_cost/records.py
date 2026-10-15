"""Portable run records, excluding host names, user paths, and Git metadata."""
import hashlib
import json
import platform
from importlib.metadata import version

from omegaconf import OmegaConf

from datasets import ROOT
from parameters import named_parameters


def write_record(path, parameters, cfg):
    files = sorted([*ROOT.glob("*.py"), ROOT / "config.yaml", ROOT / "uv.lock",
                    ROOT / "data/manifest.json", *ROOT.glob("data/*.csv")])
    record = {
        "mode": cfg.mode, "device": str(cfg.device),
        "budgets": list(cfg.budgets), "parameters": named_parameters(parameters),
        "fit_settings": OmegaConf.to_container(cfg.fit, resolve=True),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "numpy", "scipy", "pandas", "matplotlib", "hydra-core")},
        "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    record["fit_settings"].pop("output_dir")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
