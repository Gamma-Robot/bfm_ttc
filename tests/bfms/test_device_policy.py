from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import bfm_adapters.device_policy as device_policy
from bfm_adapters.contracts import TensorSpec
from bfm_adapters.device_policy import PersistentOrtCudaPolicy
from bfm_adapters.registry import BFMInstanceConfig


_WARP_FLOAT32 = object()


@dataclass(frozen=True)
class _Device:
    ordinal: int = 3
    is_cuda: bool = True

    def __str__(self) -> str:
        return f"cuda:{self.ordinal}"


@dataclass(frozen=True)
class _Stream:
    device: _Device
    cuda_stream: int = 123456


@dataclass
class _Buffer:
    shape: tuple[int, ...]
    ptr: int
    device: _Device
    dtype: object = _WARP_FLOAT32
    is_contiguous: bool = True


class _Binding:
    def __init__(self) -> None:
        self.inputs: list[tuple[object, ...]] = []
        self.outputs: list[tuple[object, ...]] = []
        self.synchronized = 0
        self.cleared_inputs = 0
        self.cleared_outputs = 0

    def bind_input(self, *args: object) -> None:
        self.inputs.append(args)

    def bind_output(self, *args: object) -> None:
        self.outputs.append(args)

    def synchronize_outputs(self) -> None:
        self.synchronized += 1

    def clear_binding_inputs(self) -> None:
        self.cleared_inputs += 1

    def clear_binding_outputs(self) -> None:
        self.cleared_outputs += 1


class _Options:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, name: str, value: str) -> None:
        self.entries[name] = value


class _RunOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_run_config_entry(self, name: str, value: str) -> None:
        self.entries[name] = value


class _Session:
    def __init__(
        self,
        config: BFMInstanceConfig,
        binding: _Binding,
        *,
        input_shapes: dict[str, tuple[object, ...]] | None = None,
    ) -> None:
        self._binding = binding
        self._inputs = [
            SimpleNamespace(
                name=spec.name,
                shape=("batch", *spec.non_batch_shape)
                if input_shapes is None
                else input_shapes[spec.name],
                type="tensor(float)",
            )
            for spec in config.policy_inputs
        ]
        self._outputs = [
            SimpleNamespace(
                name=spec.name,
                shape=("batch", *spec.non_batch_shape),
                type="tensor(float)",
            )
            for spec in config.policy_outputs
        ]
        self._outputs.append(
            SimpleNamespace(name="diagnostic", shape=("batch", 1), type="tensor(float)")
        )
        self.runs: list[object | None] = []

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def io_binding(self) -> _Binding:
        return self._binding

    def run_with_iobinding(self, binding: _Binding, options: object = None) -> None:
        assert binding is self._binding
        self.runs.append(options)


def _config(*, allow_cpu_ep_fallback: bool = False) -> BFMInstanceConfig:
    return BFMInstanceConfig(
        family="toy",
        instance="toy_policy",
        artifacts={"policy": "toy-policy.batched"},
        control_period=0.02,
        motion_command_query_offsets=(0,),
        policy_inputs=(
            TensorSpec("observation", "float32", (4,)),
            TensorSpec("recurrent", "float32", (5,)),
        ),
        policy_outputs=(
            TensorSpec("joint_target", "float32", (3,)),
            TensorSpec("next_recurrent", "float32", (5,)),
        ),
        policy_joint_names=("joint",),
        command_joint_names=("joint",),
        motion_command_body_names=("body",),
        family_config={},
        allow_cpu_ep_fallback=allow_cpu_ep_fallback,
    )


def _buffers(config: BFMInstanceConfig, device: _Device, *, batch: int = 7):
    pointer = 1000

    def make(specs: tuple[TensorSpec, ...]) -> dict[str, _Buffer]:
        nonlocal pointer
        result: dict[str, _Buffer] = {}
        for spec in specs:
            result[spec.name] = _Buffer(
                shape=(batch, *spec.non_batch_shape), ptr=pointer, device=device
            )
            pointer += 1000
        return result

    return make(config.policy_inputs), make(config.policy_outputs)


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config: BFMInstanceConfig,
    *,
    input_shapes: dict[str, tuple[object, ...]] | None = None,
):
    binding = _Binding()
    session = _Session(config, binding, input_shapes=input_shapes)
    warp = SimpleNamespace(float32=_WARP_FLOAT32, synchronized=[])
    warp.synchronize_stream = warp.synchronized.append
    captured: dict[str, object] = {}

    def make_session(*args, **kwargs):
        assert captured["preload_directory"] == ""
        captured.update(kwargs)
        return session

    monkeypatch.setattr(device_policy, "_warp_runtime", lambda: warp)
    monkeypatch.setattr(
        device_policy.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider"],
    )
    monkeypatch.setattr(device_policy.ort, "SessionOptions", _Options)
    monkeypatch.setattr(device_policy.ort, "RunOptions", _RunOptions)
    monkeypatch.setattr(device_policy.ort, "InferenceSession", make_session)
    monkeypatch.setattr(
        device_policy.ort, "preload_dlls",
        lambda *, directory: captured.update(preload_directory=directory),
    )
    return binding, session, warp, captured


def test_binds_every_configured_tensor_once_and_reuses_the_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    device = _Device()
    stream = _Stream(device)
    inputs, outputs = _buffers(config, device)
    binding, session, warp, captured = _install_runtime(monkeypatch, config)

    policy = PersistentOrtCudaPolicy(
        "unused.onnx",
        config,
        input_buffers=inputs,
        output_buffers=outputs,
        stream=stream,
    )

    assert policy.batch_size == 7
    assert policy.asynchronous is True
    assert policy.input_names == ("observation", "recurrent")
    assert policy.output_names == ("joint_target", "next_recurrent")
    assert [call[0] for call in binding.inputs] == list(policy.input_names)
    assert [call[0] for call in binding.outputs] == list(policy.output_names)
    assert all(call[1:3] == ("cuda", 3) for call in binding.inputs + binding.outputs)
    assert all(call[3] is np.float32 for call in binding.inputs + binding.outputs)
    assert captured["provider_options"][0]["user_compute_stream"] == "123456"
    assert captured["sess_options"].entries == {
        "session.disable_cpu_ep_fallback": "1"
    }

    policy.enqueue()
    policy.enqueue()
    policy.enqueue(synchronize=True)
    assert len(binding.inputs) == 2
    assert len(binding.outputs) == 2
    assert len(session.runs) == 3
    assert all(isinstance(options, _RunOptions) for options in session.runs[:2])
    assert session.runs[-1] is None
    assert policy._pending == 0
    assert binding.synchronized == 1
    assert warp.synchronized == []

    # An interrupted segment still has an explicit failure-path drain.
    policy.enqueue()
    assert policy._pending == 1
    policy.drain()
    assert warp.synchronized == [stream]
    assert binding.synchronized == 2
    assert session.runs[-1] is None
    assert policy._pending == 0

    policy.close()
    policy.close()
    assert binding.cleared_inputs == 1
    assert binding.cleared_outputs == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.enqueue()


def test_cpu_fallback_uses_explicit_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(allow_cpu_ep_fallback=True)
    device = _Device()
    stream = _Stream(device)
    inputs, outputs = _buffers(config, device)
    binding, session, warp, captured = _install_runtime(monkeypatch, config)

    policy = PersistentOrtCudaPolicy(
        "unused.onnx",
        config,
        input_buffers=inputs,
        output_buffers=outputs,
        stream=stream,
    )
    policy.enqueue()
    policy.drain()

    assert policy.asynchronous is False
    assert session.runs == [None]
    assert binding.synchronized == 1
    assert warp.synchronized == [stream, stream]
    assert captured["sess_options"].entries == {}
    policy.close()


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda inputs, outputs: outputs.pop("joint_target"), ValueError, "names mismatch"),
        (
            lambda inputs, outputs: setattr(inputs["observation"], "shape", (7, 3)),
            ValueError,
            "shape",
        ),
        (
            lambda inputs, outputs: setattr(inputs["observation"], "dtype", object()),
            TypeError,
            "float32",
        ),
        (
            lambda inputs, outputs: setattr(
                inputs["observation"], "is_contiguous", False
            ),
            ValueError,
            "contiguous",
        ),
        (
            lambda inputs, outputs: setattr(
                outputs["joint_target"], "device", _Device(ordinal=4)
            ),
            ValueError,
            "must be on",
        ),
    ],
)
def test_rejects_buffers_that_do_not_match_the_configured_abi(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    error: type[Exception],
    match: str,
) -> None:
    config = _config()
    device = _Device()
    inputs, outputs = _buffers(config, device)
    mutation(inputs, outputs)
    _install_runtime(monkeypatch, config)

    with pytest.raises(error, match=match):
        PersistentOrtCudaPolicy(
            "unused.onnx",
            config,
            input_buffers=inputs,
            output_buffers=outputs,
            stream=_Stream(device),
        )


def test_rejects_non_float_config_and_static_onnx_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    bad_config = replace(
        config,
        policy_outputs=(TensorSpec("joint_target", "float64", (3,)),),
    )
    device = _Device()
    inputs, outputs = _buffers(bad_config, device)
    _install_runtime(monkeypatch, bad_config)
    with pytest.raises(TypeError, match="float32 buffers only"):
        PersistentOrtCudaPolicy(
            "unused.onnx",
            bad_config,
            input_buffers=inputs,
            output_buffers=outputs,
            stream=_Stream(device),
        )

    inputs, outputs = _buffers(config, device)
    _install_runtime(
        monkeypatch,
        config,
        input_shapes={"observation": ("batch", 99), "recurrent": ("batch", 5)},
    )
    with pytest.raises(ValueError, match="axis 1 mismatch"):
        PersistentOrtCudaPolicy(
            "unused.onnx",
            config,
            input_buffers=inputs,
            output_buffers=outputs,
            stream=_Stream(device),
        )
