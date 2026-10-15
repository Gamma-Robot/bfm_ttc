# command_optimization

The runtime for test-time motion command optimization. It combines reference
motions, sampling optimizers, frozen BFM adapters, and physics simulation, and
runs each experiment twice: once with the baseline command and once with the
optimized command.

`run.py` and `batch.py` run configured experiments, and `loop.py` implements
the receding-horizon loop. Command interpolation and tracking objectives are in
`motion_tracking/`, candidate evaluation is in `planning/`, and physics and
state continuation are in `simulation/`. Export and visualization work from
the recorded results.

See [usage](../../docs/user-guide.md) and [official assets](../../docs/assets.md).
