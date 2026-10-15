import subprocess
import sys

import pytest

import sampling_optimizers.registry as registry
from sampling_optimizers.algs.cem import CEM


def test_import_sampling_optimizers_does_not_import_optional_runtime() -> None:
    code = (
        "import sys; import sampling_optimizers; "
        "assert 'jax' not in sys.modules; assert 'evosax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_optional_dependency_has_actionable_error() -> None:
    name = "missing-optional-test"
    registry._register_optional(
        name,
        module="does_not_matter",
        attribute="DoesNotMatter",
        dependency="definitely_missing_sampling_optimizers_test_dependency",
        extra="test-extra",
    )
    try:
        assert name not in registry.available()
        with pytest.raises(ModuleNotFoundError, match=r"sampling-optimizers\[test-extra\]"):
            registry.get_optimizer(name)
    finally:
        registry._OPTIONAL_REGISTRY.pop(name)


def test_optional_entry_loads_and_caches_optimizer_class() -> None:
    name = "optional-cem-test"
    registry._register_optional(
        name,
        module="sampling_optimizers.algs.cem",
        attribute="CEM",
        dependency="numpy",
        extra="test-extra",
    )
    try:
        assert name in registry.available()
        assert registry.get_optimizer(name) is CEM
        assert registry.get_optimizer(name) is CEM
    finally:
        registry._REGISTRY.pop(name)
        registry._OPTIONAL_REGISTRY.pop(name)
