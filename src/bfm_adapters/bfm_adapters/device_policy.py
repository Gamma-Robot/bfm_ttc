"""Persistent zero-copy ONNX Runtime policy execution on a Warp CUDA stream."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnxruntime as ort

from bfm_adapters.contracts import TensorSpec
from bfm_adapters.registry import BFMInstanceConfig


def _warp_runtime() -> ModuleType:
    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - exercised only without runtime extra
        raise RuntimeError("device policy execution requires warp-lang") from exc
    return wp


def _metadata_shape(metadata: Any, actual: tuple[int, ...], *, label: str) -> None:
    """Validate static ONNX dimensions while accepting a symbolic batch axis."""

    expected = tuple(metadata.shape)
    if len(expected) != len(actual):
        raise ValueError(f"{label} rank mismatch: expected {expected}, got {actual}")
    for axis, (declared, value) in enumerate(zip(expected, actual, strict=True)):
        if isinstance(declared, int) and declared != value:
            raise ValueError(
                f"{label} axis {axis} mismatch: expected {declared}, got {value}"
            )


class PersistentOrtCudaPolicy:
    """Bind caller-owned Warp CUDA buffers once and reuse them for inference.

    The bfm instance config is the complete policy I/O ABI; no output name
    or action shape is special. CUDA-only graphs enqueue asynchronously on the
    caller's Warp stream. Instances that explicitly allow ORT CPU placement use
    a synchronized path because host nodes cannot preserve stream-only ordering.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: BFMInstanceConfig,
        *,
        input_buffers: Mapping[str, Any],
        output_buffers: Mapping[str, Any],
        stream: Any,
    ) -> None:
        wp = _warp_runtime()
        device = getattr(stream, "device", None)
        if device is None or not getattr(device, "is_cuda", False):
            raise ValueError("PersistentOrtCudaPolicy requires a CUDA Warp stream")
        try:
            device_id = int(device.ordinal)
            stream_handle = int(stream.cuda_stream)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("stream must expose CUDA device and stream handles") from exc
        if device_id < 0 or stream_handle < 0:
            raise ValueError("CUDA device and stream handles must be non-negative")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")

        self._wp = wp
        self._stream = stream
        self._device_id = device_id
        self._input_specs = _specs(config.policy_inputs, "policy_inputs")
        self._output_specs = _specs(config.policy_outputs, "policy_outputs")
        self._input_buffers = dict(input_buffers)
        self._output_buffers = dict(output_buffers)
        self._asynchronous = not config.allow_cpu_ep_fallback
        self._closed = False
        self._pending = 0

        options = ort.SessionOptions()
        if self._asynchronous:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "device_id": str(device_id),
            "user_compute_stream": str(stream_handle),
            "do_copy_in_default_stream": "1",
            "use_ep_level_unified_stream": "1",
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

        self._batch_size = self._validate_buffers()
        self._validate_session_io()
        self._binding = self._session.io_binding()
        for name, spec in self._input_specs.items():
            self._bind_input(name, spec, self._input_buffers[name])
        for name, spec in self._output_specs.items():
            self._bind_output(name, spec, self._output_buffers[name])

        self._run_options: Any | None = None
        if self._asynchronous:
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry(
                "disable_synchronize_execution_providers", "1"
            )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def asynchronous(self) -> bool:
        """Whether ``enqueue`` returns without a device synchronization."""

        return self._asynchronous

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self._input_specs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._output_specs)

    def enqueue(self, *, synchronize: bool = False) -> None:
        """Run once, optionally completing queued GPU-only work."""

        self._require_open()
        if self._asynchronous:
            if synchronize:
                # A normal bound run is both the final real policy inference
                # and the ORT 1.23 provider-completion boundary.  Reusing that
                # inference avoids a separate drain inference per segment.
                self._session.run_with_iobinding(self._binding)
                self._binding.synchronize_outputs()
                self._pending = 0
                return
            self._session.run_with_iobinding(self._binding, self._run_options)
            self._pending += 1
            return

        # CPU fallback nodes cannot be ordered solely by the CUDA stream.
        self._wp.synchronize_stream(self._stream)
        self._session.run_with_iobinding(self._binding)
        self._binding.synchronize_outputs()
        self._wp.synchronize_stream(self._stream)

    def drain(self) -> None:
        """Complete work and release retained provider-completion state."""

        self._require_open()
        if not self._pending:
            return
        self._wp.synchronize_stream(self._stream)
        self._binding.synchronize_outputs()
        # ORT 1.23 retains asynchronous provider-completion state after a run
        # that disabled synchronization. One final bound synchronous run makes
        # teardown deterministic and releases that state while buffers are live.
        self._session.run_with_iobinding(self._binding)
        self._pending = 0

    def close(self) -> None:
        """Drain and release ORT objects without taking ownership of buffers."""

        if self._closed:
            return
        self.drain()
        self._binding.clear_binding_inputs()
        self._binding.clear_binding_outputs()
        self._binding = None
        self._session = None
        self._run_options = None
        self._closed = True
        gc.collect()

    def __enter__(self) -> PersistentOrtCudaPolicy:
        self._require_open()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def _validate_buffers(self) -> int:
        _exact_names(self._input_buffers, self._input_specs, "input buffers")
        _exact_names(self._output_buffers, self._output_specs, "output buffers")
        batches: set[int] = set()
        for kind, buffers, specs in (
            ("input", self._input_buffers, self._input_specs),
            ("output", self._output_buffers, self._output_specs),
        ):
            for name, spec in specs.items():
                shape = self._validate_buffer(
                    buffers[name], label=f"policy {kind} {name!r}"
                )
                expected = (shape[0], *spec.non_batch_shape)
                if shape != expected:
                    raise ValueError(
                        f"policy {kind} {name!r} shape {shape} != expected {expected}"
                    )
                batches.add(shape[0])
        if len(batches) != 1:
            raise ValueError(f"policy buffers disagree on batch size: {sorted(batches)}")
        return batches.pop()

    def _validate_buffer(self, value: Any, *, label: str) -> tuple[int, ...]:
        if getattr(value, "device", None) != self._stream.device:
            raise ValueError(f"{label} must be on {self._stream.device}")
        if getattr(value, "dtype", None) != self._wp.float32:
            raise TypeError(f"{label} must use Warp float32")
        if not getattr(value, "is_contiguous", False):
            raise ValueError(f"{label} must be contiguous")
        try:
            shape = tuple(int(size) for size in value.shape)
            pointer = int(value.ptr)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be a Warp array") from exc
        if not shape or any(size < 1 for size in shape):
            raise ValueError(f"{label} has invalid shape {shape}")
        if pointer <= 0:
            raise ValueError(f"{label} has invalid CUDA pointer {pointer}")
        return shape

    def _validate_session_io(self) -> None:
        session_inputs = {value.name: value for value in self._session.get_inputs()}
        session_outputs = {value.name: value for value in self._session.get_outputs()}
        _exact_names(session_inputs, self._input_specs, "ONNX inputs")
        missing_outputs = set(self._output_specs) - set(session_outputs)
        if missing_outputs:
            raise ValueError(f"ONNX outputs missing configured names: {sorted(missing_outputs)}")
        for kind, metadata, specs in (
            ("input", session_inputs, self._input_specs),
            ("output", session_outputs, self._output_specs),
        ):
            for name, spec in specs.items():
                _metadata_shape(
                    metadata[name],
                    (self._batch_size, *spec.non_batch_shape),
                    label=f"ONNX {kind} {name!r}",
                )
                declared_type = getattr(metadata[name], "type", None)
                if declared_type is not None and declared_type != "tensor(float)":
                    raise TypeError(
                        f"ONNX {kind} {name!r} type {declared_type!r} is not float32"
                    )

    def _bind_input(self, name: str, spec: TensorSpec, buffer: Any) -> None:
        self._binding.bind_input(
            name,
            "cuda",
            self._device_id,
            np.dtype(spec.dtype).type,
            (self._batch_size, *spec.non_batch_shape),
            int(buffer.ptr),
        )

    def _bind_output(self, name: str, spec: TensorSpec, buffer: Any) -> None:
        self._binding.bind_output(
            name,
            "cuda",
            self._device_id,
            np.dtype(spec.dtype).type,
            (self._batch_size, *spec.non_batch_shape),
            int(buffer.ptr),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PersistentOrtCudaPolicy is closed")


def _specs(values: tuple[TensorSpec, ...], label: str) -> dict[str, TensorSpec]:
    specs = {spec.name: spec for spec in values}
    if not specs:
        raise ValueError(f"{label} must not be empty")
    unsupported = [
        spec.name for spec in values if np.dtype(spec.dtype) != np.dtype(np.float32)
    ]
    if unsupported:
        raise TypeError(
            f"device policy supports float32 buffers only; {label}={unsupported}"
        )
    return specs


def _exact_names(
    actual: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(
            f"{label} names mismatch: got={sorted(actual)} expected={sorted(expected)}"
        )
