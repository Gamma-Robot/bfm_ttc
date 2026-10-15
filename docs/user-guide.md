# Usage

This guide assumes you have installed the environment from the
[README](../README.md) and prepared the official assets described in the
[asset guide](assets.md).

## Use a local motion

To prepare your own data, follow the [motion preparation guide](motions.md).

The runtime reads a retargeted G1 dataset in
[Any4HDMI v2 format](https://github.com/EGalahad/any4hdmi/blob/main/docs/dataset_format.md). The dataset
contains `manifest.json` and either `motions/<motion_id>.npz` or
`motions/<motion_id>/motion.npz`. The manifest declares the names, timestep,
and coordinate layout. Each NPZ contains a `qpos` array of shape `(T, 36)`:
root position, scalar-first root quaternion, and 29 joint positions in the
order declared by the manifest.

For a dataset directory named `data`, create an experiment YAML next to it:

```yaml
ref: experiments/cem_fixed_budget.base.yaml
name: walking
bfm:
  ref: bfms/heft_pmg.yaml
reference:
  dataset_root: ./data
  motion_id: walking
  clip_duration_seconds: null
output:
  formats: [npz, any4hdmi]
  trajectories: [execution, command]
visualization:
  enabled: false
```

```sh
MUJOCO_GL=egl uv run bfm-ttc run /path/to/walking.yaml \
  --device-index 0 --output-dir outputs/walking
```

The `ref` values point to files under the repository's `configs/` directory.
A relative `dataset_root` is resolved from the directory of the experiment YAML.
A motion ID is the motion's path relative to the dataset and may include
subdirectories, spaces, and parentheses. If several sequences share a filename,
use the full ID. The loader reorders joints by their declared names and
resamples the motion to the BFM's control period. With
`clip_duration_seconds: null`, the full motion is executed. A positive duration
executes only the beginning of the motion, and the later frames remain
available as context.

## Settings

| Setting | Meaning |
| --- | --- |
| `optimizer.num_candidates` | Number of candidates `N` per optimization iteration. |
| `optimizer.num_iterations` | Number of iterations `I` in each fixed-budget planning update. |
| `planning.horizon_steps` | Rollout horizon `H`, in control steps. |
| `planning.replanning_interval_steps` | Number of steps `E` executed before replanning. |
| `planning.num_knots` | Number of control knots that parameterize a candidate command. |
| `planning.max_worlds` | Maximum number of worlds simulated at once. Candidates beyond this capacity are evaluated in further waves, so it does not change `N`. |
| `planning.seed` | Random seed for the optimizer. |

Choose an optimizer with `optimizer.ref` and a BFM with `bfm.ref`. The
referenced files hold the algorithm- and model-specific parameters. An explicit
`ref` replaces the inherited component with the referenced configuration; fields
alongside it override that configuration. Without a new `ref`, fields override
the inherited parameters. `E` must
satisfy `1 <= E <= H`.

To use EvoSax's separable CMA-ES, install the extra and run JAX on the CPU in
float64:

```sh
uv sync --frozen --all-packages --extra evosax
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
```

## Batch execution

```sh
uv run bfm-ttc batch configs/experiments/motion_set.batch.yaml --dry-run
MUJOCO_GL=egl uv run bfm-ttc batch configs/experiments/motion_set.batch.yaml \
  --device-index 0 --output-dir outputs/motion-set
```

The supplied batch runs SONIC on the four included motions with the default
CEM settings. The walking motion is limited to its first 10 seconds, and the
other three run in full. Each cell gets its own directory with paired traces
and a report. To render a cell's comparison video, run
`MUJOCO_GL=egl uv run bfm-ttc render <cell-directory>`.

A batch configuration lists the input motions, experiment references,
optimizer variants, budgets, and seeds. Every cell is run by the same runner
as a single experiment.

## Outputs and visualization

- `resolved_experiment.yaml`: the configuration that was actually executed.
- `report.json`: tracking errors, costs, completion, and experiment identities.
- `planning.json`: optimization records and the evaluation work actually performed.
- `paired_traces.npz`: reference, applied command, execution, and validity arrays.
- `provenance.json`: identities of the source code, inputs, and environment.
- The requested `execution` and `command` exports: two separate trajectories,
  both including the reset frame.

To render the comparison video from the saved traces:

```sh
MUJOCO_GL=egl uv run bfm-ttc render outputs/walking
```

The video shows the baseline and optimized executions against the same
reference overlay. Rendering replays the saved trajectories and does not rerun
the simulation. Set `visualization.enabled: true` to render automatically after
each run.
