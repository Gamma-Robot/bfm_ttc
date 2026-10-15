"""Official TWIST2 checkpoint ABI and CUDA execution facts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from bfm_adapters import load_instance_config
from bfm_adapters.onnx_policy import BatchedOrtCudaPolicy
from command_optimization.artifacts import ensure_artifact

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "bfms" / "twist2_1017_20k.yaml"
POLICY = "twist2-1017-20k-policy"


def test_twist2_official_policy_supports_dynamic_batch_and_static_graph_abi() -> None:
    config = load_instance_config(CONFIG)
    path = ensure_artifact(POLICY, root=REPO)
    policy = BatchedOrtCudaPolicy(
        path,
        config,
        device_id=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")),
    )
    rng = np.random.default_rng(1414)
    values = rng.standard_normal((4, 1432), dtype=np.float32)
    actual = policy.run({"input": values})["output"]
    one = policy.run({"input": values[:1]})["output"]

    assert actual.shape == (4, 29)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(one[0], actual[0], rtol=1.0e-5, atol=1.0e-5)

    # The released graph accepts 35 future-motion values but its future
    # encoder slices off index 1431. Preserve that checkpoint-bound fact.
    perturbed = values.copy()
    perturbed[:, 1431] += np.float32(1000.0)
    np.testing.assert_array_equal(
        policy.run({"input": perturbed})["output"], actual
    )
