from __future__ import annotations

from pathlib import Path

import pytest

from wave_invariance import assert_real_paired_wave_invariance

pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "tests" / "fixtures" / "experiments" / "twist2_smoke.yaml"


def test_twist2_real_paired_run_is_wave_size_invariant(tmp_path: Path) -> None:
    # Use the same 0.3% repeatability bound as device-evaluation conformance.
    # TWIST2's high arm gains can amplify float32 replay noise to 3.13e-3 in
    # qpos over this clip; candidate identity and executed motion_command stay strict.
    assert_real_paired_wave_invariance(
        EXPERIMENT,
        tmp_path,
        selected_cost_rtol=3.0e-3,
        qpos_atol=5.0e-3,
    )
