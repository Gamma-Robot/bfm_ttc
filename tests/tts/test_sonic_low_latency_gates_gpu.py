from __future__ import annotations

from pathlib import Path

import pytest

from wave_invariance import assert_real_paired_wave_invariance

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "tests" / "fixtures" / "experiments" / "sonic_low_latency_smoke.yaml"


def test_sonic_low_latency_real_paired_run_is_wave_size_invariant(
    tmp_path: Path,
) -> None:
    assert_real_paired_wave_invariance(
        EXPERIMENT, tmp_path, selected_cost_rtol=1.0e-4
    )
