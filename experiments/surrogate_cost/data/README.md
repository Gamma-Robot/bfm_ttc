# Measurement data

The analysis starts from seven CSV tables of measured configuration costs.
`manifest.json` records the SHA-256, row count, and columns of each file, and
the loader validates the files against it.

Appendices C.3–C.4 of the [paper](https://arxiv.org/abs/2610.14896) describe
the measurement and analysis protocols.

| File | Rows | Use |
| --- | ---: | --- |
| `solver.csv` | 293 | Five optimizers; only the 59 CEM rows are used to fit the surrogate. |
| `horizon.csv` | 24 | CEM sweep over horizon and replanning interval at convergence. |
| `joint_train.csv` | 100 | Joint fitting over all four compute settings. |
| `joint_test.csv` | 50 | Held-out configurations for measuring prediction error. |
| `random.csv` | 1,200 | 200 measured random configurations per budget. |
| `proposed.csv` | 6 | The configuration selected by the frozen model for each budget, with its measured cost. |
| `nominal.csv` | 1 | Baseline cost with the nominal command, used in the sweep figure. |

`I` is the number of iterations per planning update, `N` the number of
candidates per iteration, `H` the rollout horizon in control steps, and `E` the
number of steps executed before replanning. `algorithm` identifies the
optimizer group. `budget` is the upper bound on compute use `K = I*N*H/E`.

All measurements use SONIC on ten AMASS-S1 motions with simulation seeds
0, 1, and 2. The source motions come from [AMASS](https://amass.is.tue.mpg.de/).

For each configuration and seed, the executed running cost of each motion is
divided by its number of executed steps, and the ten motion means are averaged
with equal weight. `cost` is the average of the three seed means. Where present,
`cost_std` is their sample standard deviation (`ddof=1`). Local terminal
penalties are not included in this full-sequence cost.

The final fit uses 24 horizon, 59 CEM solver, and 100 joint-training
configurations. The 50 joint-test configurations are held out from it, although
23 of them were used to evaluate earlier versions of the model. Because all
sets share the same motions, the test measures how well predictions transfer
to new configurations on a fixed motion set. Six further joint configurations
that were measured outside the train/test split are not included.

One random configuration, `(I,N,H,E)=(11,141,13,5)` at budget 4096, also
appears in joint training. The other 1,205 budget-evaluation configurations,
including the six frozen-model proposals, overlap neither joint training nor
the 59 fixed-horizon CEM configurations. No frozen-model proposal appears among
the random configurations.

For each budget, the reference minimum is the lowest measured cost among the
200 saved random configurations and the frozen-model proposal. Allocation
regret is computed relative to this finite reference set.
