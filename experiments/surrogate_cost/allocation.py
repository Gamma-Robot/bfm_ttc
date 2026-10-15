import numpy as np
import pandas as pd

from datasets import CONFIGURATION
from model import inputs, predict


def search(parameters, budgets, device):
    records = []
    for budget in budgets:
        candidates = []
        for e in range(5, 41):
            h, i = np.meshgrid(np.arange(e, 10 * e + 1), np.arange(1, 513))
            n = np.minimum(np.minimum(8192 // i, budget * e // (i * h)), 327680 // h)
            valid = n >= np.maximum(2, (256 + i - 1) // i)
            rows = pd.DataFrame({"I": i[valid], "N": n[valid], "H": h[valid], "E": e})
            if rows.empty:
                continue
            rows["predicted_cost"] = predict(parameters, inputs(rows, "full"), "full", device)
            rows["compute"] = rows.I * rows.N * rows.H / rows.E
            columns = ["predicted_cost", "compute", "H", "E", "I", "N"]
            candidates.append(rows.sort_values(columns, ascending=[True, False, True, True, True, True]).iloc[0])
        if not candidates:
            raise ValueError(f"No feasible configuration at budget {budget}")
        best = pd.DataFrame(candidates).sort_values(columns, ascending=[True, False, True, True, True, True]).iloc[0]
        records.append({"budget": budget, **best.to_dict()})
    rows = pd.DataFrame(records)
    rows[CONFIGURATION] = rows[CONFIGURATION].astype(int)
    return rows[["budget", *CONFIGURATION, "compute", "predicted_cost"]]
