import numpy as np
import pytest

from command_optimization.planning.risk import reduce_scenario_costs
from command_optimization.simulation.randomization import (
    RandomizationRanges,
    RobustnessConfig,
    ScenarioBatch,
    sample_scenarios,
)


def test_nominal_scenarios_are_immutable_and_exact() -> None:
    batch = ScenarioBatch.nominal(count=3, num_joints=2)

    assert batch.count == 3
    assert batch.num_joints == 2
    np.testing.assert_array_equal(batch.contact_friction_scale, np.ones(3))
    np.testing.assert_array_equal(batch.joint_velocity_offset, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        batch.actuation_strength_scale[0] = 2.0


def test_sampling_is_seed_deterministic_and_digest_binds_content() -> None:
    ranges = RandomizationRanges(
        contact_friction_scale=[0.8, 1.2],
        joint_armature_scale=[0.9, 1.1],
        joint_frictionloss_scale=[0.5, 1.5],
        actuation_strength_scale=[0.9, 1.1],
        root_linear_velocity_std=0.02,
        root_angular_velocity_std=0.03,
        joint_velocity_std=0.01,
    )
    first = sample_scenarios(ranges, count=4, seed=17, num_joints=3)
    repeated = sample_scenarios(ranges, count=4, seed=17, num_joints=3)
    different = sample_scenarios(ranges, count=4, seed=18, num_joints=3)

    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(repeated, field))
    assert first.digest == repeated.digest
    assert first.digest != different.digest


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"contact_friction_scale": [1.2, 0.8]}, "ordered"),
        ({"joint_armature_scale": [0.0, 1.0]}, "strictly positive"),
        ({"joint_velocity_std": -0.1}, "non-negative"),
    ],
)
def test_ranges_reject_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RandomizationRanges(**kwargs)


@pytest.mark.parametrize("field, value", [("count", 0), ("num_joints", 0)])
def test_sampling_rejects_nonpositive_shapes(field: str, value: int) -> None:
    kwargs = {"count": 1, "seed": 0, "num_joints": 1, field: value}
    with pytest.raises(ValueError, match=field):
        sample_scenarios(RandomizationRanges(), **kwargs)


def test_scenario_batch_rejects_mismatched_axes() -> None:
    with pytest.raises(ValueError, match="leading axis"):
        ScenarioBatch(
            contact_friction_scale=np.ones(2),
            joint_armature_scale=np.ones(2),
            joint_frictionloss_scale=np.ones(2),
            actuation_strength_scale=np.ones(3),
            root_linear_velocity_offset=np.zeros((2, 3)),
            root_angular_velocity_offset=np.zeros((2, 3)),
            joint_velocity_offset=np.zeros((2, 4)),
        )


def test_risk_reduction_is_trajectory_level_and_lower_is_better() -> None:
    costs = np.array([[1.0, 9.0, 5.0], [4.0, 4.0, 4.0]], dtype=np.float64)

    np.testing.assert_array_equal(
        reduce_scenario_costs(costs, strategy="mean"), np.array([5.0, 4.0])
    )
    np.testing.assert_array_equal(
        reduce_scenario_costs(costs, strategy="worst"), np.array([9.0, 4.0])
    )


def test_risk_reduction_rejects_nonfinite_or_unknown_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        reduce_scenario_costs(
            np.array([[1.0, np.inf]], dtype=np.float64), strategy="mean"
        )
    with pytest.raises(ValueError, match="unknown"):
        reduce_scenario_costs(
            np.array([[1.0]], dtype=np.float64), strategy="median"  # type: ignore[arg-type]
        )


def test_robustness_config_has_strict_disabled_defaults() -> None:
    assert not RobustnessConfig().enabled
    enabled = RobustnessConfig(
        enabled=True,
        num_scenarios=4,
        scenario_seed=7,
        risk="worst",
        ranges=RandomizationRanges(contact_friction_scale=(0.8, 1.2)),
    )
    assert enabled.num_scenarios == 4

    with pytest.raises(ValueError, match="disabled robustness"):
        RobustnessConfig(num_scenarios=2)
    with pytest.raises(ValueError, match="nominal parity"):
        RobustnessConfig(
            enabled=True,
            ranges=RandomizationRanges(contact_friction_scale=(0.8, 1.2)),
        )
