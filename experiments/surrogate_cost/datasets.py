from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIGURATION = ["I", "N", "H", "E"]


def load_data():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    schema = {
        "solver": (293, ["algorithm", *CONFIGURATION, "cost", "cost_std"]),
        "horizon": (24, ["H", "E", "cost", "cost_std"]),
        "joint_train": (100, [*CONFIGURATION, "cost"]),
        "joint_test": (50, [*CONFIGURATION, "cost"]),
        "random": (1200, ["budget", *CONFIGURATION, "cost"]),
        "proposed": (6, ["budget", *CONFIGURATION, "cost"]),
        "nominal": (1, ["cost"]),
    }
    data = {}
    for name, (count, columns) in schema.items():
        path = ROOT / "data" / f"{name}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Data identity mismatch: {name}")
        rows = pd.read_csv(ROOT / "data" / f"{name}.csv", float_precision="round_trip")
        if len(rows) != count or list(rows.columns) != columns:
            raise ValueError(f"Unexpected dataset shape: {name}")
        numbers = rows.drop(columns=["algorithm"], errors="ignore").to_numpy()
        if not np.isfinite(numbers).all() or (numbers < 0).any() or (rows.cost <= 0).any():
            raise ValueError(f"Invalid measurements: {name}")
        coordinates = [c for c in CONFIGURATION if c in columns]
        if coordinates:
            values = rows[coordinates].to_numpy()
            if (values < 1).any() or (values != np.floor(values)).any() or (rows.E > rows.H).any():
                raise ValueError(f"Invalid configuration: {name}")
            keys = (["algorithm"] if name == "solver" else []) + coordinates
            if rows.duplicated(keys).any():
                raise ValueError(f"Duplicate configuration: {name}")
        data[name] = rows
    counts = data["solver"].groupby("algorithm").size().to_dict()
    if counts != {"CBO": 59, "CEM": 59, "CMA-ES": 57, "MPPI-CMA": 59, "PS": 59}:
        raise ValueError("Incorrect solver groups")
    if len(data["joint_train"].merge(data["joint_test"], on=CONFIGURATION)):
        raise ValueError("Joint training and test sets overlap")
    for name in ("random", "proposed"):
        rows = data[name]
        expected = 200 if name == "random" else 1
        if rows.groupby("budget").size().to_dict() != {k: expected for k in (256, 512, 1024, 2048, 4096, 8192)}:
            raise ValueError(f"Incorrect budget groups: {name}")
        if ((rows.I > 512) | (rows.N < 2) | (rows.I * rows.N < 256) |
            (rows.I * rows.N > 8192) | (rows.E < 5) | (rows.E > 40) |
            (rows.H > 10 * rows.E) | (rows.N * rows.H > 327680) |
            (rows.I * rows.N * rows.H > rows.budget * rows.E)).any():
            raise ValueError(f"Configuration outside the search domain: {name}")
        if name == "random" and (10 * rows.I * rows.N * rows.H < 9 * rows.budget * rows.E).any():
            raise ValueError("Random configuration outside the budget band")
    if len(data["random"].merge(data["proposed"], on=CONFIGURATION)):
        raise ValueError("Historical proposals must be distinct from random configurations")
    return data
