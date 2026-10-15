import json

import hydra
import numpy as np
import pandas as pd
import torch

from datasets import ROOT, load_data
from figures import allocation_figure, strategy_figure
from evaluation import evaluate, component_results
from parameters import load_parameters, require_paper_parameters
from records import write_record


def write_table(metrics, output):
    lines = [r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
             r"$K$ & $\tau$ & $\rho$ & NDCG@5 & Regret (\%) \\", r"\midrule"]
    rows = [(str(int(row.budget)), row) for row in metrics.itertuples()]
    means = metrics.select_dtypes(include="number").mean()
    if not metrics.measurement_status.eq("matched").all():
        means["regret_percent"] = np.nan
    rows.append(("Mean", means))
    for label, row in rows:
        lines.append(f"{label} & {row.kendall_tau:.3f} & {row.spearman_rho:.3f} & "
                     f"{row.ndcg_at_5:.3f} & " + (f"{row.regret_percent:.2f}" if np.isfinite(row.regret_percent) else "--") + " " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "compute_strategy.tex").write_text("\n".join(lines) + "\n")


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg):
    torch.set_num_threads(1)
    if cfg.mode not in ("paper", "refit"):
        raise ValueError("mode must be paper or refit")
    output = ROOT / cfg.output_dir
    generated = ("allocations.csv", "metrics.csv", "component_metrics.json", "run.json",
                 "compute_allocation.pdf", "compute_strategy.pdf", "compute_strategy.tex")
    if any((output / name).exists() for name in generated):
        raise FileExistsError("Reproduction outputs already exist; choose a new output_dir")
    data = load_data()
    parameters = load_parameters(ROOT / cfg.parameters)
    if cfg.mode == "paper":
        require_paper_parameters(parameters)
    budgets = list(cfg.budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(b not in (256, 512, 1024, 2048, 4096, 8192) for b in budgets):
        raise ValueError("Use distinct budgets from the six paper budgets")
    selections, metrics = evaluate(data, parameters, budgets, cfg.device, paper=cfg.mode == "paper")
    components = component_results(data, parameters, cfg.device)
    output.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output / "allocations.csv", index=False)
    table = metrics.copy()
    means = metrics.select_dtypes(include="number").mean().to_dict()
    means.update(budget="Mean", measurement_status=f"{metrics.measurement_status.eq('matched').sum()}/{len(metrics)} matched")
    if not metrics.measurement_status.eq("matched").all():
        means.update(regret_percent=np.nan, proposed_cost=np.nan)
    table = pd.concat([table, pd.DataFrame([means])], ignore_index=True)
    table.to_csv(output / "metrics.csv", index=False)
    (output / "component_metrics.json").write_text(json.dumps(components, indent=2) + "\n")
    write_table(metrics, output)
    allocation_figure(data, parameters, output, cfg.device)
    strategy_figure(metrics, output)
    write_record(output / "run.json", parameters, cfg)
    print(table.to_string(index=False))
    print(json.dumps(components, indent=2))


if __name__ == "__main__":
    main()
