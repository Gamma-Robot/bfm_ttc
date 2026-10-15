# bfm_adapters

Adapters for the frozen HEFT, SONIC, and TWIST2 BFMs. They build observations,
run policy inference, keep history state, and map actions to commands. SONIC
has a release and a low-latency variant, each with its own assets and control
settings.

Each `BFMAdapter` implements the observation and action conventions and the
state of one model. `SerialReferenceBFM` runs the policy on the host and serves
as the reference implementation. An optional prepared device BFM keeps policy
inputs and history on the simulation device. Both produce the `PDCommand` that
the runtime expects.

This package does not depend on the optimizer or runtime packages.
See [assets](../../docs/assets.md) for preparing the official models.
