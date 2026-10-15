# sampling_optimizers

Sampling-based black-box optimizers with a NumPy interface. Each optimizer
samples normalized candidates, gets their scalar costs from a callback you
provide, and updates its sampling distribution. Costs must be finite, and lower
is better. Candidate arrays passed through the public interface are float64.

The package does not depend on BFM adapters, robot geometry, or simulation.
The built-in optimizers are CEM, predictive sampling, MPPI, MPPI-CMA, CBO,
DIAL, SPIDER, and weighted elites. An optional EvoSax adapter supports
separable CMA-ES; EvoSax and JAX are imported only when it is selected.

See the [main README](../../README.md) for installation and experiment usage.
