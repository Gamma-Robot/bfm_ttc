from itertools import product
import json

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import least_squares

from datasets import ROOT, load_data
from model import cost, inputs, restrict
from parameters import PARAMETER_NAMES, named_parameters
from records import write_record


def fit(x, observed, starts, bounds, component, cfg):
    device = torch.device(cfg.fit.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Fitting requires a CUDA device")
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(observed, dtype=torch.float64, device=device)

    def residual_tensor(values):
        return (cost(restrict(values, component), x) / y).log()

    derivative = torch.func.jacfwd(residual_tensor)

    def residual(values):
        return residual_tensor(torch.as_tensor(values, device=device)).cpu().numpy()

    def jacobian(values):
        return derivative(torch.as_tensor(values, device=device)).cpu().numpy()

    lower, upper = np.log(bounds).T
    successful = []
    trials = []
    for index, start in enumerate(starts):
        result = least_squares(
            residual, np.clip(np.log(start), lower + 1e-12, upper - 1e-12),
            jac=jacobian, bounds=(lower, upper), loss="linear", x_scale="jac",
            max_nfev=cfg.fit.max_nfev, ftol=cfg.fit.ftol,
            xtol=cfg.fit.xtol, gtol=cfg.fit.gtol,
        )
        if result.success and np.isfinite(result.fun).all():
            successful.append(result)
        trials.append({"start": list(map(float, start)), "success": bool(result.success),
                       "status": int(result.status), "message": str(result.message),
                       "nfev": int(result.nfev), "njev": int(result.njev),
                       "log_mse": float(np.mean(result.fun ** 2)),
                       "optimality": float(result.optimality)})
        print(f"{component}: {index + 1}/{len(starts)}, log MSE {np.mean(result.fun ** 2):.12g}", flush=True)
    if not successful:
        raise RuntimeError(f"No successful {component} fit")
    best = min(successful, key=lambda result: float(result.fun @ result.fun))
    return np.exp(best.x).tolist(), {"starts": trials, "successful_starts": len(successful),
                                    "best_log_mse": float(np.mean(best.fun ** 2))}


def solver_starts(minimum, cfg):
    s = cfg.starts.component
    floors = [s.floors[0], minimum * s.floors[1]]
    starts = [[c, *s.scales, u, v, g]
              for c, u, v, g in product(floors, s.powers, s.powers, s.exponents)]
    starts.extend([*values, g] for values in s.extra for g in s.extra_exponents)
    return starts


def joint_training_data(data):
    """Assemble the 24 horizon, 59 CEM, and 100 joint-training points."""
    horizon = data["horizon"]
    solver = data["solver"].loc[data["solver"].algorithm.eq("CEM")]
    joint = data["joint_train"]
    x = np.concatenate((inputs(horizon, "horizon"), inputs(solver, "full"), inputs(joint, "full")))
    y = np.concatenate((horizon.cost, solver.cost, joint.cost))
    return x, y


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    torch.set_num_threads(1)
    if torch.device(cfg.fit.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Fitting requires a CUDA device")
    output = ROOT / cfg.fit.output_dir
    if output.exists():
        raise FileExistsError("Fit output directory already exists; choose a new fit.output_dir")
    data = load_data()
    horizon = data["horizon"]
    solver = data["solver"].loc[data["solver"].algorithm.eq("CEM")]
    bounds = np.array([cfg.bounds[name] for name in PARAMETER_NAMES["full"]], dtype=float)
    output.mkdir(parents=True)
    reports = {}

    def save_reports():
        (output / "fit_report.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")

    h, reports["horizon"] = fit(inputs(horizon, "horizon"), horizon.cost.to_numpy(),
            list(cfg.starts.horizon), bounds[:4], "horizon", cfg)
    save_reports()
    starts = [[*h, *s, *t] for s in cfg.starts.solver for t in cfg.starts.transfer]
    x, y = joint_training_data(data)
    full, reports["full"] = fit(x, y, starts, bounds, "full", cfg)
    save_reports()
    minimum = float(solver.cost.min())
    solver_bounds = [[1e-12, minimum * (1 - 1e-10)], [np.exp(-20), 10],
                     [np.exp(-20), 10], [.005, 8], [.005, 8], [.25, 10]]
    s, reports["solver"] = fit(inputs(solver, "solver"), solver.cost.to_numpy(),
            solver_starts(minimum, cfg), solver_bounds, "solver", cfg)
    save_reports()
    parameters = {"horizon": h, "solver": s, "full": full}
    OmegaConf.save(OmegaConf.create(named_parameters(parameters)), output / "parameters.yaml")
    fit_cfg = OmegaConf.merge(cfg, {"mode": "refit", "device": cfg.fit.device})
    write_record(output / "fit_run.json", parameters, fit_cfg)
    print("Completed fits:", {name: len(report["starts"]) for name, report in reports.items()})


if __name__ == "__main__":
    main()
