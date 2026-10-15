# Surrogate cost model

This module fits the surrogate cost model, selects integer compute allocations,
and reproduces the analysis figures and tables from the supplied measurements
of configuration costs. It covers Sections 4.2–4.3 and Appendices C.3–C.4 of
the [paper](https://arxiv.org/abs/2610.14896).

## Run the analysis

The module has its own Python 3.12 environment and dependency lock. From the
BFM-TTC repository root:

```sh
cd experiments/surrogate_cost
uv sync --frozen
uv run python reproduce.py
```

By default, the analysis runs on the CPU in double precision with the seven
included CSV files and the frozen parameters in `parameters.yaml`. It searches
six budgets and evaluates the selected configurations against the saved
measurements. Results are written to `output/paper/`:

| File | Content |
| --- | --- |
| `allocations.csv` | Selected integer configurations, compute use, and predicted costs. |
| `metrics.csv` | Ranking metrics, measured selection costs, allocation regret, and coverage. |
| `component_metrics.json` | Component and joint-test prediction errors, plus sweep comparisons. |
| `compute_allocation.pdf` | Optimizer and horizon comparisons with fitted component curves. |
| `compute_strategy.pdf` | Measured costs of the selected configurations and the best saved reference costs. |
| `compute_strategy.tex` | Budget comparison table. |
| `run.json` | Parameters, input/code hashes, dependency versions, and execution settings. |

Existing outputs are never overwritten, so write another run to a new directory:

```sh
uv run python reproduce.py output_dir=output/another-run
```

The [measurement data](data/README.md) page describes where the measurements
come from, how they are aggregated, and how they are split for training and
evaluation.

## Refit the model

Fitting runs PyTorch in double precision on an NVIDIA CUDA GPU, together with
SciPy's bounded least-squares solver. From this module's directory:

```sh
uv run python fit.py
uv run python reproduce.py mode=refit parameters=output/refit/parameters.yaml
```

The fit uses 5 starts for the horizon model, 60 for the joint model, and 104
for the solver component. The parameters, per-start diagnostics, and a run
record are written to `output/refit/`. Use `fit.device=cuda:1` to select
another GPU, or `fit.output_dir=output/another-fit` to write the fit elsewhere.

When evaluating a refit, each selected configuration is matched to the
measurements at the same budget. A selection with no matching measurement gets
the status `unmeasured`, its measured-cost and regret fields stay empty, and
the table shows them as `--`.

## Code

| File | Responsibility |
| --- | --- |
| `model.py` | Full surrogate and horizon/solver restrictions. |
| `parameters.py`, `parameters.yaml` | Explicit parameter order and frozen values. |
| `datasets.py`, `data/` | Measurement tables, identity checks, and experimental splits. |
| `fit.py` | Training-point assembly and multi-start fitting. |
| `allocation.py` | Budget-constrained integer configuration search. |
| `evaluation.py` | Prediction/ranking metrics and configuration-matched measurements. |
| `figures.py`, `reproduce.py` | Plotting and analysis entry point. |
| `records.py` | Execution records. |
| `config.yaml` | Devices, outputs, budgets, bounds, and initial values. |

## Surrogate

For configuration `(I,N,H,E)`, the full prediction is

```text
B(H,E) = J_inf + C_H * exp(-alpha_H * (2*H-E)) + C_E * (E-1)
P(I,N) = C_I * I^(-alpha_I) + C_N * N^(-alpha_N)
J_hat  = B(H,E) + P(I,N)^(eta/H) * gamma^H
```

For the converged horizon measurements, the model uses `B(H,E)` alone, which
assumes zero solver error. The separate solver-component curve is
`J_floor + (C_I*I^(-alpha_I) + C_N*N^(-alpha_N))^beta` at `H=40,E=20`.
Its six parameters are fitted independently of the ten parameters of the full
model. `parameters.yaml` stores the parameters as named positive values, and
`parameters.py` defines their numerical order.

## Fitting

The joint objective weights 183 configurations equally: 24 horizon points,
59 CEM solver points, and 100 joint-training points. Each residual is the
natural log of the predicted cost divided by the measured cost. Horizon points
use the converged restriction, and solver and joint points use the full model.

The optimizer works on the natural log of the parameters, with the bounds and
initial values in `config.yaml`. The best of five horizon starts initializes
the horizon terms. Five solver initializations crossed with twelve
`(eta,gamma)` initializations give 60 joint starts, and all ten joint
parameters stay free. The independent solver-component fit uses 104 starts.

SciPy `least_squares` runs with a linear loss, Jacobian scaling, at most 50,000
function evaluations per start, and `ftol=xtol=gtol=1e-11`. PyTorch computes
the residuals and forward-mode Jacobians on CUDA in float64. Of the starts that
succeed with finite values, the one with the lowest sum of squared residuals is
kept. `fit_report.json` records the status, number of evaluations, and log MSE
of every start.

With the frozen parameters, the joint test MARE over the 50 test points is
about 24.61872944%, which rounds to the 24.62% reported in the paper.

## Compute allocation

Compute use is `K = I*N*H/E`, and the six budgets are 256, 512, 1024, 2048,
4096, and 8192. The search runs over positive integers subject to

```text
1 <= I <= 512; N >= 2; 256 <= I*N <= 8192
5 <= E <= 40; E <= H <= 10*E; N*H <= 327680
I*N*H <= budget*E
```

For each feasible `(E,H,I)`, the search takes the largest feasible candidate
count `N = min(8192//I, (budget*E)//(I*H), 327680//H)`, which must be at least
`max(2, ceil(256/I))`. Since the prediction decreases with `N` at fixed
`I,H,E`, this finds the surrogate minimum over the whole feasible domain.
Exact ties are broken in favor of greater compute use, then smaller `H`, `E`,
`I`, and `N`, in that order. The saved random configurations also use at least
90% of their budget.

## Evaluation and measured configurations

Kendall tau-b, Spearman rho, and NDCG@5 measure prediction quality on the 200
saved random configurations per budget, where lower cost is better. NDCG uses
linear relevance `201 - rank(cost)`, average ranks for tied costs, and
logarithmic discounts. Allocation regret is
`100*(selected_measured_cost/reference_min-1)`, where the reference set is the
200 saved random points plus the frozen-model proposal for that budget.

In `mode=paper`, evaluation checks that the parameters equal the supplied
frozen parameters and that the search reproduces the frozen-model proposals.
In `mode=refit`, each selected configuration must match a configuration in
the same budget's reference set exactly. Selections without a match are marked
`unmeasured` and have no measured cost or regret. Ranking metrics evaluate the
refitted model on the saved random points. The figure shows only measured
selections, and mean regret is reported only when every requested budget has
a measured selection.
