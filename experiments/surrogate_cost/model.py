import numpy as np
import torch


def cost(log_parameters, inputs):
    p = log_parameters.exp()
    i, n, h, e, converged = inputs.unbind(-1)
    horizon = p[0] + p[1] * torch.exp(-p[2] * (2 * h - e)) + p[3] * (e - 1)
    log_solver = torch.logaddexp(log_parameters[4] - p[6] * i.log(),
                                log_parameters[5] - p[7] * n.log())
    sampling = torch.exp(p[8] / h * log_solver + h * log_parameters[9])
    return horizon + torch.where(converged.bool(), torch.zeros_like(sampling), sampling)


def restrict(parameters, component):
    if component == "full":
        return parameters
    if component == "horizon":
        return torch.cat((parameters, parameters.new_zeros(6)))
    if component == "solver":
        c, a, b, u, v, g = parameters.unbind()
        zero = parameters.new_tensor(float("-inf"))
        one = parameters.new_tensor(0.)
        return torch.stack((c, zero, one, zero, a, b, u, v,
                            g + parameters.new_tensor(40.).log(), one))
    raise ValueError(f"Unknown component: {component}")


def inputs(rows, component):
    if component == "horizon":
        return np.column_stack((np.ones((len(rows), 2)), rows.H, rows.E, np.ones(len(rows))))
    if component in ("solver", "full"):
        return np.column_stack((rows[["I", "N", "H", "E"]], np.zeros(len(rows))))
    raise ValueError(f"Unknown component: {component}")


def predict(parameters, x, component, device):
    p, x = np.asarray(parameters, dtype=float), np.asarray(x, dtype=float)
    if p.shape != ({"horizon": 4, "solver": 6, "full": 10}[component],):
        raise ValueError("Incorrect parameter count")
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Parameters must be finite and positive")
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all():
        raise ValueError("Expected finite I, N, H, E, converged columns")
    if (x[:, :4] <= 0).any() or (x[:, 3] > x[:, 2]).any():
        raise ValueError("Require I, N, H, E > 0 and E <= H")
    if not np.isin(x[:, 4], [0, 1]).all():
        raise ValueError("Converged must be 0 or 1")
    if component == "horizon" and (x[:, 4] != 1).any():
        raise ValueError("The horizon restriction requires zero sampling error")
    if component == "solver" and not (x[:, 2:] == [40, 20, 0]).all():
        raise ValueError("The solver restriction requires H=40 and E=20")
    with torch.no_grad():
        logs = torch.as_tensor(np.log(p), dtype=torch.float64, device=device)
        values = torch.as_tensor(x, dtype=torch.float64, device=device)
        return cost(restrict(logs, component), values).cpu().numpy()
