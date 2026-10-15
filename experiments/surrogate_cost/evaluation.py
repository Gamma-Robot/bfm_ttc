"""Prediction metrics and measured costs matched to the selected configuration."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from allocation import search
from datasets import CONFIGURATION
from model import inputs, predict


def ranking_metrics(observed, predicted):
    relevance = len(observed) + 1 - rankdata(observed, method="average")
    discount = np.log2(np.arange(2, 7))
    chosen = np.argsort(predicted, kind="stable")[:5]
    ideal = np.sort(relevance)[::-1][:5]
    return {
        "kendall_tau": float(kendalltau(observed, predicted, variant="b").statistic),
        "spearman_rho": float(spearmanr(observed, predicted).statistic),
        "ndcg_at_5": float(np.sum(relevance[chosen] / discount) / np.sum(ideal / discount)),
    }


def evaluate_selections(data, parameters, selections, device, *, paper):
    records = []
    for selected in selections.itertuples(index=False):
        random = data["random"].loc[data["random"].budget.eq(selected.budget)]
        historical = data["proposed"].loc[data["proposed"].budget.eq(selected.budget)]
        if len(random) != 200 or len(historical) != 1:
            raise ValueError("Each budget requires 200 random and one historical proposed configuration")
        coordinates = [getattr(selected, name) for name in CONFIGURATION]
        if paper and not historical[CONFIGURATION].eq(coordinates).all(axis=1).all():
            raise ValueError(f"Selected configuration differs from the paper at budget {selected.budget}")
        # Regret is measured against the saved 201-point reference set S_K.
        # A new prediction cannot borrow the cost of the historical proposal.
        reference = pd.concat([random, historical], ignore_index=True)
        matched = reference.loc[reference[CONFIGURATION].eq(coordinates).all(axis=1)]
        if len(matched) > 1:
            raise ValueError("Ambiguous measurements for selected configuration")
        measured = float(matched.cost.iloc[0]) if len(matched) else np.nan
        best = float(reference.cost.min())
        predicted = predict(parameters["full"], inputs(random, "full"), "full", device)
        records.append({
            "budget": selected.budget,
            **ranking_metrics(random.cost.to_numpy(), predicted),
            "regret_percent": 100 * (measured / best - 1),
            "proposed_cost": measured, "best_measured_cost": best,
            "measurement_status": "matched" if len(matched) else "unmeasured",
        })
    return pd.DataFrame(records)


def evaluate(data, parameters, budgets, device, *, paper):
    selections = search(parameters["full"], budgets, device)
    return selections, evaluate_selections(data, parameters, selections, device, paper=paper)


def component_results(data, parameters, device):
    solver = data["solver"].loc[data["solver"].algorithm.eq("CEM")]
    groups = [("solver", solver), ("horizon", data["horizon"]), ("full", data["joint_test"])]
    errors = {}
    for component, rows in groups:
        prediction = predict(parameters[component], inputs(rows, component), component, device)
        errors[f"{component}_mare_percent"] = float(100 * np.abs(prediction / rows.cost - 1).mean())
    sweep = solver.assign(O=solver.I * solver.N)
    costs = sweep.loc[sweep.O.eq(8192)].set_index("I").cost
    errors["solver_cost_reduction_percent"] = float(100 * (1 - costs[32] / costs[1]))
    horizon = data["horizon"]
    for ratio in [4, 10]:
        costs = horizon.loc[horizon.H.eq(ratio * horizon.E)].set_index("E").cost
        errors[f"horizon_ratio_{ratio}_change_percent"] = float(100 * (costs[10] / costs[5] - 1))
    return errors
