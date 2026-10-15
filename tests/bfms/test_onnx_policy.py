from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from bfm_adapters.contracts import TensorSpec
from bfm_adapters.registry import load_instance_config


class _Session:
    def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
        self._inputs = [SimpleNamespace(name=name) for name in inputs]
        self._outputs = [SimpleNamespace(name=name) for name in outputs]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def _policy(
    *,
    inputs: tuple[str, ...] = ("observation",),
    outputs: tuple[str, ...] = ("action",),
) -> BatchedOrtCudaPolicy:
    policy = object.__new__(BatchedOrtCudaPolicy)
    policy._input_names = inputs
    policy._output_names = outputs
    return policy


def test_session_outputs_may_contain_unconsumed_diagnostics() -> None:
    policy = _policy(outputs=("action",))
    policy._session = _Session(("observation",), ("action", "diagnostic"))

    policy._validate_session_io()


def test_session_inputs_remain_exact() -> None:
    policy = _policy()
    policy._session = _Session(("observation", "hidden"), ("action",))

    with pytest.raises(ValueError, match="input names mismatch"):
        policy._validate_session_io()


def test_configured_outputs_must_be_a_nonempty_subset() -> None:
    policy = _policy(outputs=("missing",))
    policy._session = _Session(("observation",), ("action", "diagnostic"))
    with pytest.raises(ValueError, match="output names mismatch"):
        policy._validate_session_io()

    policy = _policy(outputs=())
    policy._session = _Session(("observation",), ("action",))
    with pytest.raises(ValueError, match="output names mismatch"):
        policy._validate_session_io()


def test_cpu_ep_fallback_is_disabled_unless_bfm_explicitly_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bfm_adapters.onnx_policy as policy_module

    config = load_instance_config(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "bfms"
        / "heft_pmg.yaml"
    )
    assert config.allow_cpu_ep_fallback is False
    captured: list[dict[str, str]] = []
    loading_order: list[str] = []

    class Options:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, name: str, value: str) -> None:
            self.entries[name] = value

    class Session(_Session):
        def get_providers(self):
            return ["CUDAExecutionProvider"]

    def make_session(*args, **kwargs):
        assert loading_order[-1] == "preload"
        loading_order.append("session")
        options = kwargs["sess_options"]
        captured.append(options.entries.copy())
        return Session(
            tuple(spec.name for spec in config.policy_inputs),
            tuple(spec.name for spec in config.policy_outputs),
        )

    monkeypatch.setattr(
        policy_module.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider"],
    )
    monkeypatch.setattr(policy_module.ort, "SessionOptions", Options)
    monkeypatch.setattr(policy_module.ort, "InferenceSession", make_session)
    def preload(*, directory):
        assert directory == ""
        loading_order.append("preload")

    monkeypatch.setattr(policy_module.ort, "preload_dlls", preload)

    BatchedOrtCudaPolicy("unused.onnx", config)
    BatchedOrtCudaPolicy(
        "unused.onnx", replace(config, allow_cpu_ep_fallback=True)
    )

    assert captured == [{"session.disable_cpu_ep_fallback": "1"}, {}]
    assert loading_order == ["preload", "session", "preload", "session"]


def test_policy_reuses_bound_cuda_values_per_batch_and_owns_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bfm_adapters.onnx_policy as policy_module

    class Value:
        def __init__(self, array: np.ndarray) -> None:
            self.array = np.array(array, copy=True)
            self.updates = 0

        def update_inplace(self, array: np.ndarray) -> None:
            np.copyto(self.array, array)
            self.updates += 1

        def numpy(self) -> np.ndarray:
            return self.array

    class Values:
        @staticmethod
        def ortvalue_from_numpy(array, *_args):
            return Value(array)

        @staticmethod
        def ortvalue_from_shape_and_type(shape, dtype, *_args):
            return Value(np.zeros(shape, dtype=dtype))

    class Binding:
        def __init__(self) -> None:
            self.inputs: dict[str, Value] = {}
            self.outputs: dict[str, Value] = {}
            self.synchronizations = 0

        def bind_ortvalue_input(self, name: str, value: Value) -> None:
            self.inputs[name] = value

        def bind_ortvalue_output(self, name: str, value: Value) -> None:
            self.outputs[name] = value

        def synchronize_outputs(self) -> None:
            self.synchronizations += 1

    class Session:
        def __init__(self) -> None:
            self.bindings: list[Binding] = []
            self.runs = 0

        def io_binding(self) -> Binding:
            binding = Binding()
            self.bindings.append(binding)
            return binding

        def run_with_iobinding(self, binding: Binding) -> None:
            self.runs += 1
            binding.outputs["action"].array[:] = binding.inputs["observation"].array

    monkeypatch.setattr(policy_module.ort, "OrtValue", Values)
    policy = _policy()
    policy._input_specs = {
        "observation": TensorSpec("observation", "float32", (2,))
    }
    policy._output_specs = {"action": TensorSpec("action", "float32", (2,))}
    policy._device_id = 0
    policy._batches = {}
    policy._session = Session()

    first = policy.run({"observation": np.ones((3, 2), dtype=np.float32)})
    second = policy.run({"observation": np.full((3, 2), 2.0, dtype=np.float32)})
    policy.run({"observation": np.ones((1, 2), dtype=np.float32)})

    assert len(policy._session.bindings) == 2
    assert policy._session.runs == 3
    assert policy._batches[3].inputs["observation"].updates == 2
    assert policy._batches[3].binding.synchronizations == 2
    np.testing.assert_array_equal(first["action"], 1.0)
    np.testing.assert_array_equal(second["action"], 2.0)
