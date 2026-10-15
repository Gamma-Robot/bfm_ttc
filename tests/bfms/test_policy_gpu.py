"""GPU tests for batched ORT CUDA policy (no BatchedBFM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
ort = pytest.importorskip("onnxruntime")

from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from bfm_adapters.registry import load_instance_config
from command_optimization.artifacts import ensure_artifact

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "bfms" / "heft_pmg.yaml"
ARTIFACT_ID = "heft-pmg-policy.batched"
TOL = 1e-5


def _require_cuda_and_artifact():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider unavailable")
    try:
        path = ensure_artifact(ARTIFACT_ID, root=REPO_ROOT)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
    config = load_instance_config(CONFIG_PATH)
    assert config.artifacts["policy"] == ARTIFACT_ID
    return path, config


def _random_inputs(config, batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        spec.name: rng.standard_normal(
            (batch, *spec.non_batch_shape), dtype=np.dtype(spec.dtype)
        ).astype(np.dtype(spec.dtype), copy=False)
        for spec in config.policy_inputs
    }


def test_batched_ort_cuda_policy_batch1_and_lane0() -> None:
    model_path, config = _require_cuda_and_artifact()
    policy = BatchedOrtCudaPolicy(model_path, config, device_id=0)

    assert policy.input_names == tuple(spec.name for spec in config.policy_inputs)
    assert policy.output_names == tuple(spec.name for spec in config.policy_outputs)
    assert {item.name for item in policy._session.get_inputs()} == set(
        policy.input_names
    )
    assert {item.name for item in policy._session.get_outputs()} >= set(
        policy.output_names
    )
    assert policy._session.get_providers()[0] == "CUDAExecutionProvider"

    rng = np.random.default_rng(0)
    inputs_b1 = _random_inputs(config, batch=1, rng=rng)
    out_a = policy.run(inputs_b1)
    out_b = policy.run(inputs_b1)
    assert set(out_a) == set(policy.output_names)
    for name, array in out_a.items():
        spec = next(spec for spec in config.policy_outputs if spec.name == name)
        assert array.shape == (1, *spec.non_batch_shape)
        assert array.dtype == np.dtype(spec.dtype)
        assert np.all(np.isfinite(array))
        np.testing.assert_allclose(array, out_b[name], rtol=0.0, atol=0.0)

    batch = 8
    inputs_b = _random_inputs(config, batch=batch, rng=np.random.default_rng(1))
    for name, value in inputs_b1.items():
        inputs_b[name] = inputs_b[name].copy()
        inputs_b[name][0] = value[0]
    out_batched = policy.run(inputs_b)
    for name in policy.output_names:
        assert out_batched[name].shape[0] == batch
        assert np.all(np.isfinite(out_batched[name]))
        np.testing.assert_allclose(
            out_batched[name][0],
            out_a[name][0],
            rtol=TOL,
            atol=TOL,
        )
