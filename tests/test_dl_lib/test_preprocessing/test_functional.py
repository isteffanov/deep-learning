
import pytest
import numpy as np

from dl_lib.preprocessing import functional

@pytest.mark.parametrize(
    ("input", "expected_output"),
    [
        ([], []),
        ([1], [0]),
        ([1, 1], [0, 0]),
    ],
)
def test_min_max_normalize_when_all_values_equal_returns_all_zero_array(input, expected_output):
    np.testing.assert_array_equal(functional.min_max_normalize(np.array(input)), np.array(expected_output))

@pytest.mark.parametrize(
    ("input", "expected_output"),
    [
        ([0, 1], [0, 1]),
        ([0, 5, 7.5, 10], [0, 0.5, 0.75, 1]),
        ([10, 20, 30], [0, 0.5, 1]),
        ([-1, 0, 1], [0, 0.5, 1]),
        ([-10, -8, -5], [0, 0.4, 1]),
    ],
)
def test_min_max_normalize_when_differing_values_returns_rescaled_array(input, expected_output):
    np.testing.assert_allclose(functional.min_max_normalize(np.array(input)), np.array(expected_output))
