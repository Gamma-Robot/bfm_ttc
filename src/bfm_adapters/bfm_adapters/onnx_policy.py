"""Batched ONNX Runtime CUDA policy session with IO binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from bfm_adapters.contracts import NamedTensors, TensorSpec
from bfm_adapters.registry import BFMInstanceConfig


@dataclass
class _BoundBatch:
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    binding: Any


class BatchedOrtCudaPolicy:
    """One CUDA-primary ORT session consuming/returning named policy tensors."""

    def __init__(
        self,
        model_path: str | Path,
        config: BFMInstanceConfig,
        *,
        device_id: int = 0,
    ) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")

        self._input_specs = {spec.name: spec for spec in config.policy_inputs}
        self._output_specs = {spec.name: spec for spec in config.policy_outputs}
        self._input_names = tuple(spec.name for spec in config.policy_inputs)
        self._output_names = tuple(spec.name for spec in config.policy_outputs)
        self._device_id = int(device_id)
        self._batches: dict[int, _BoundBatch] = {}

        options = ort.SessionOptions()
        if not config.allow_cpu_ep_fallback:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        # CUDA remains the preferred provider. A bfm may explicitly permit
        # ORT's CPU placement for unsupported shape-only graph nodes.
        provider_options = {
            "device_id": str(self._device_id),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": "1",
            "use_tf32": "0",
        }
        # Use the NVIDIA wheels installed by ORT's CUDA/cuDNN extras.
        ort.preload_dlls(directory="")
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
            provider_options=[provider_options],
        )
        providers = self._session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDAExecutionProvider is not active; providers={providers}"
            )
        self._validate_session_io()

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def device_id(self) -> int:
        return self._device_id

    def run(self, inputs: NamedTensors) -> dict[str, np.ndarray]:
        """Run one forward pass using persistent buffers for its batch size."""

        batch = self._validate_inputs(inputs)
        bound = self._batches.get(batch)
        if bound is None:
            bound = self._bind_batch(batch)
            self._batches[batch] = bound
        for name in self._input_names:
            array = np.ascontiguousarray(inputs[name])
            bound.inputs[name].update_inplace(array)
        self._session.run_with_iobinding(bound.binding)
        bound.binding.synchronize_outputs()
        result: dict[str, np.ndarray] = {}
        for name in self._output_names:
            array = np.asarray(bound.outputs[name].numpy()).copy()
            self._validate_output_array(name, array, batch)
            result[name] = array
        return result

    def _bind_batch(self, batch: int) -> _BoundBatch:
        binding = self._session.io_binding()
        inputs: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        for name in self._input_names:
            spec = self._input_specs[name]
            shape = (batch, *spec.non_batch_shape)
            array = np.zeros(shape, dtype=np.dtype(spec.dtype))
            value = ort.OrtValue.ortvalue_from_numpy(
                array, "cuda", self._device_id
            )
            inputs[name] = value
            binding.bind_ortvalue_input(name, value)
        for name in self._output_names:
            spec = self._output_specs[name]
            shape = (batch, *spec.non_batch_shape)
            value = ort.OrtValue.ortvalue_from_shape_and_type(
                shape, np.dtype(spec.dtype), "cuda", self._device_id
            )
            outputs[name] = value
            binding.bind_ortvalue_output(name, value)
        return _BoundBatch(inputs=inputs, outputs=outputs, binding=binding)

    def _validate_session_io(self) -> None:
        session_inputs = {item.name for item in self._session.get_inputs()}
        session_outputs = {item.name for item in self._session.get_outputs()}
        expected_inputs = set(self._input_names)
        expected_outputs = set(self._output_names)
        if session_inputs != expected_inputs:
            raise ValueError(
                "ONNX input names mismatch: "
                f"session={sorted(session_inputs)} config={sorted(expected_inputs)}"
            )
        if not expected_outputs or not expected_outputs <= session_outputs:
            raise ValueError(
                "ONNX output names mismatch: "
                f"session={sorted(session_outputs)} config={sorted(expected_outputs)}"
            )

    def _validate_inputs(self, inputs: NamedTensors) -> int:
        if set(inputs) != set(self._input_names):
            raise ValueError(
                "policy input names mismatch: "
                f"got={sorted(inputs)} expected={sorted(self._input_names)}"
            )
        batch: int | None = None
        for name in self._input_names:
            array = inputs[name]
            spec = self._input_specs[name]
            self._validate_array(name, array, spec)
            if batch is None:
                batch = int(array.shape[0])
            elif int(array.shape[0]) != batch:
                raise ValueError(
                    f"policy inputs disagree on batch size: {name} has "
                    f"{array.shape[0]}, expected {batch}"
                )
        assert batch is not None
        return batch

    def _validate_output_array(
        self, name: str, array: np.ndarray, batch: int
    ) -> None:
        spec = self._output_specs[name]
        self._validate_array(name, array, spec)
        if int(array.shape[0]) != batch:
            raise ValueError(
                f"policy output {name!r} batch {array.shape[0]} != input batch {batch}"
            )

    @staticmethod
    def _validate_array(name: str, array: np.ndarray, spec: TensorSpec) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"tensor {name!r} must be a numpy array")
        expected_dtype = np.dtype(spec.dtype)
        if array.dtype != expected_dtype:
            raise TypeError(
                f"tensor {name!r} dtype {array.dtype} != expected {expected_dtype}"
            )
        expected_shape = (array.shape[0],) + spec.non_batch_shape
        if array.shape != expected_shape:
            raise ValueError(
                f"tensor {name!r} shape {array.shape} != expected {expected_shape}"
            )
        if array.shape[0] < 1:
            raise ValueError(f"tensor {name!r} batch must be non-empty")
