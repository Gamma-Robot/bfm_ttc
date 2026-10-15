import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from model import inputs, predict


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight", metadata={"Creator": "", "Producer": "", "CreationDate": None})
    plt.close(figure)


def allocation_figure(data, parameters, output, device):
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout="constrained")
    solver, horizon = data["solver"].copy(), data["horizon"].copy()
    solver["O"] = solver.I * solver.N
    horizon["ratio"] = horizon.H / horizon.E
    best = solver.loc[solver.groupby(["algorithm", "O"]).cost.idxmin()]
    for name in ["CEM", "PS", "CMA-ES", "CBO", "MPPI-CMA"]:
        rows = best.loc[best.algorithm.eq(name)].sort_values("O")
        axes[0].errorbar(rows.O, rows.cost, rows.cost_std, marker="o", label=name, capsize=2)
    cem = solver.loc[solver.algorithm.eq("CEM")]
    for index, (budget, rows) in enumerate(cem.groupby("O")):
        curve = pd.DataFrame({"I": np.geomspace(rows.I.min(), rows.I.max(), 240)})
        curve["N"], curve["H"], curve["E"] = budget / curve.I, 40, 20
        color = plt.cm.Blues(.4 + index * .11)
        axes[1].plot(curve.I, predict(parameters["solver"], inputs(curve, "solver"), "solver", device), color=color)
        axes[1].errorbar(rows.I, rows.cost, rows.cost_std, fmt="o", color=color, label=str(budget), capsize=2)
    best = horizon.loc[horizon.groupby("ratio").cost.idxmin()].sort_values("ratio")
    axes[2].errorbar(best.ratio, best.cost, best.cost_std, marker="o", capsize=2)
    for index, (ratio, rows) in enumerate(horizon.groupby("ratio")):
        curve = pd.DataFrame({"E": np.linspace(rows.E.min(), rows.E.max(), 240)})
        curve["H"] = ratio * curve.E
        color = plt.cm.Blues(.4 + index * .11)
        axes[3].plot(curve.E, predict(parameters["horizon"], inputs(curve, "horizon"), "horizon", device), color=color)
        axes[3].errorbar(rows.E, rows.cost, rows.cost_std, fmt="o", color=color, label=f"{ratio:g}", capsize=2)
    for index, (axis, label) in enumerate(zip(axes, [r"$O=IN$", r"$I$", r"$K_h=H/E$", r"$E$"])):
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel(r"Cost $J$")
        axis.set_title(f"({chr(97 + index)})", loc="left")
        nominal = float(data["nominal"].cost.iloc[0])
        axis.axhline(nominal, color="gray", linestyle="--")
        axis.annotate("Nominal", (.98, nominal), xycoords=("axes fraction", "data"), ha="right", va="bottom")
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[:2]:
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_formatter(ScalarFormatter())
    for axis, title in [(axes[0], ""), (axes[1], r"$O$"), (axes[3], r"$K_h$")]:
        axis.legend(title=title, loc="lower center", bbox_to_anchor=(.5, 1.06), ncol=3, frameon=False)
    save_figure(figure, output / "compute_allocation.pdf")


def strategy_figure(metrics, output):
    figure, axis = plt.subplots(figsize=(5, 3.5), layout="constrained")
    axis.plot(metrics.budget, metrics.proposed_cost, "o-", label="Selected (measured)")
    axis.plot(metrics.budget, metrics.best_measured_cost, "s--", color="gray", label="Measured best")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([256, 1024, 8192])
    axis.set_yticks([.04, .06, .1])
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.minorticks_off()
    axis.set_xlabel(r"Compute budget $K$")
    axis.set_ylabel(r"Cost $J$")
    axis.grid(alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    if not metrics.measurement_status.eq("matched").all():
        axis.set_title("Unmeasured selections omitted", fontsize="small")
    save_figure(figure, output / "compute_strategy.pdf")
