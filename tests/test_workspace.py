import sampling_optimizers
import bfm_adapters
import command_optimization


def test_packages_import() -> None:
    for pkg in (sampling_optimizers, bfm_adapters, command_optimization):
        assert pkg.__doc__
