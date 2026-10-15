"""Real-family parity for host and device Candidate Rollout evaluation."""

from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from real_evaluation import build_real_evaluation_case
from command_optimization.motion_tracking.command import take_motion_command


pytestmark = [pytest.mark.gpu, pytest.mark.conformance]

HORIZON = 4
CAPACITY = 3
# Two repeated runs on one RTX 4090 observed repeat drift <= 2.38e-6
# (2.29e-5 relative) and host/device drift <= 6.5e-7 (1.21e-5 relative).
COST_RTOL = 3.0e-5
COST_ATOL = 1.0e-8


@pytest.mark.parametrize("family", ("heft", "sonic"))
def test_real_execution_continues_across_snapshot_boundary(family: str) -> None:
    case = build_real_evaluation_case(
        f"{family}_smoke.yaml",
        horizon=8,
        num_knots=3,
        device_index=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")),
    )
    try:
        executor = case.device_pool.executor(1)
        whole = executor.execute(
            case.execution_state, case.reference_motion, right_context=case.right_context
        )
        prefix = executor.execute(
            case.execution_state, case.reference_motion, num_steps=4,
            right_context=case.right_context,
        )
        world = case.device_pool.world(1)
        before_reset = world.robot_state()
        warmstart = world._data.qacc_warmstart.numpy().copy()
        world.reset(prefix.final_execution_state)
        after_reset = world.robot_state()
        np.testing.assert_array_equal(world._data.qacc_warmstart.numpy(), warmstart)
        for name in (
            "body_position_world", "body_orientation_world",
            "body_linear_velocity_world", "body_angular_velocity_world",
        ):
            np.testing.assert_allclose(
                getattr(after_reset, name), getattr(before_reset, name),
                rtol=0.0, atol=2.0e-6,
            )
        tail = take_motion_command(case.reference_motion, np.arange(4, 8))
        resumed = executor.execute(
            prefix.final_execution_state, tail, right_context=case.right_context
        )
        print(f"{family} continuation qpos max delta:",
              float(np.max(np.abs(whole.qpos[:, 4:] - resumed.qpos))))
        np.testing.assert_allclose(
            whole.qpos[:, 4:], resumed.qpos, rtol=0.0, atol=3.0e-5
        )
    finally:
        case.close()


@pytest.mark.parametrize(
    ("experiment_name", "cost_rtol"),
    (
        ("heft_smoke.yaml", COST_RTOL),
        ("sonic_smoke.yaml", COST_RTOL),
        ("sonic_low_latency_smoke.yaml", COST_RTOL),
        # Five repeated TWIST2 host/device runs observed <= 0.201% drift;
        # its training gains amplify MJWarp float32 repeat noise.
        ("twist2_smoke.yaml", 3.0e-3),
    ),
)
def test_real_family_device_rollout_costs_match_host(
    experiment_name: str, cost_rtol: float
) -> None:
    wp.init()
    assert wp.is_cuda_available()
    case = build_real_evaluation_case(
        experiment_name,
        horizon=HORIZON,
        num_knots=3,
        device_index=int(os.environ.get("BFM_TTC_TEST_DEVICE", "0")),
    )
    try:
        rng = np.random.default_rng(1200)
        candidates = np.stack(
            (
                np.zeros(case.space.latent_shape),
                rng.uniform(-0.7, 0.7, case.space.latent_shape),
                rng.uniform(-0.7, 0.7, case.space.latent_shape),
            )
        ).astype(np.float64)
        host_evaluator = case.evaluator("host-reference", max_worlds=CAPACITY)
        device_evaluator = case.evaluator("device", max_worlds=CAPACITY)

        host_first = host_evaluator.evaluate(candidates)
        host_repeat = host_evaluator.evaluate(candidates)
        device_first = device_evaluator.evaluate(candidates)
        device_repeat = device_evaluator.evaluate(candidates)

        for result in (host_first, host_repeat, device_first, device_repeat):
            assert not any(wave.failure for wave in result.ledger.waves)
            assert result.ledger.logical_rollouts == CAPACITY
        for first, repeat in (
            (host_first, host_repeat),
            (device_first, device_repeat),
        ):
            np.testing.assert_allclose(
                repeat.costs, first.costs, rtol=cost_rtol, atol=COST_ATOL
            )
        for host_result, device_result in (
            (host_first, device_first),
            (host_repeat, device_repeat),
        ):
            np.testing.assert_allclose(
                device_result.costs,
                host_result.costs,
                rtol=cost_rtol,
                atol=COST_ATOL,
            )
        best = int(np.argmin(host_first.costs))
        assert {
            int(np.argmin(result.costs))
            for result in (host_repeat, device_first, device_repeat)
        } == {best}
    finally:
        case.close()
