from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from bfm_adapters.registry import load_instance_config
from command_optimization.artifacts import ensure_artifact


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "bfms" / "heft_pmg.yaml"
SOURCE = "heft-pmg-policy"
PREPARED = "heft-pmg-policy.batched"


def _raw_session(path: Path, device: int):
    ort.preload_dlls(directory="")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": str(device), "use_tf32": "0"})],
    )


def test_heft_prepared_policy_matches_source_for_batch_one_and_many() -> None:
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    device = int(os.environ.get("BFM_TTC_TEST_DEVICE", "0"))
    config = load_instance_config(CONFIG)
    source = ensure_artifact(SOURCE, root=REPO)
    prepared = ensure_artifact(PREPARED, root=REPO)
    raw = _raw_session(source, device)
    batched = BatchedOrtCudaPolicy(prepared, config, device_id=device)

    rng = np.random.default_rng(47)
    batch = 4
    inputs = {
        spec.name: rng.standard_normal(
            (batch, *spec.non_batch_shape), dtype=np.dtype(spec.dtype)
        ).astype(np.dtype(spec.dtype), copy=False)
        for spec in config.policy_inputs
    }
    actual = batched.run(inputs)["action"]
    expected = np.concatenate(
        [
            raw.run(
                ["action"],
                {name: value[lane : lane + 1] for name, value in inputs.items()},
            )[0]
            for lane in range(batch)
        ],
        axis=0,
    )

    assert actual.shape == (batch, 29)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    assert batched.output_names == ("action",)
    assert len(raw.get_outputs()) > len(batched.output_names)
