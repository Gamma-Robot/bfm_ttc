"""Named parameters with explicit order at the numerical model boundary."""
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

PARAMETER_NAMES = {
    "horizon": ("J_inf", "C_H", "alpha_H", "C_E"),
    "solver": ("J_floor", "C_I", "C_N", "alpha_I", "alpha_N", "beta"),
    "full": ("J_inf", "C_H", "alpha_H", "C_E", "C_I", "C_N",
             "alpha_I", "alpha_N", "eta", "gamma"),
}


def load_parameters(path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) != set(PARAMETER_NAMES):
        raise ValueError("Expected horizon, solver, and full parameter mappings")
    result = {}
    for component, names in PARAMETER_NAMES.items():
        values = raw[component]
        if not isinstance(values, dict) or set(values) != set(names):
            raise ValueError(f"Incorrect named parameters for {component}")
        ordered = np.array([values[name] for name in names], dtype=float)
        if not np.isfinite(ordered).all() or (ordered <= 0).any():
            raise ValueError("Parameters must be finite and positive")
        result[component] = ordered.tolist()
    return result


def named_parameters(parameters):
    return {component: dict(zip(names, parameters[component], strict=True))
            for component, names in PARAMETER_NAMES.items()}


def require_paper_parameters(parameters):
    frozen = load_parameters(Path(__file__).with_name("parameters.yaml"))
    if parameters != frozen:
        raise ValueError("Paper reproduction requires frozen parameters; use mode=refit for new parameters")
