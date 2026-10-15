import numpy as np

from command_optimization.simulation.mjwarp import _scaled_scenario_rows


def test_scaled_scenario_rows_only_changes_selected_entries() -> None:
    reference_motion = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
    actual = _scaled_scenario_rows(
        reference_motion,
        np.asarray([0.5, 2.0]),
        np.asarray([0, 2], dtype=np.int32),
    )

    expected = np.repeat(reference_motion[None], 2, axis=0)
    expected[0, [0, 2]] *= 0.5
    expected[1, [0, 2]] *= 2.0
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
