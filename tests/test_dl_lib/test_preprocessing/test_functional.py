
import pytest
import numpy as np

from dl_lib.preprocessing import functional

@pytest.mark.parametrize(
    ("input", "expected_output"),
    [
        ([], [])
    ],
)
def test_min_max_normalize_when_all_values_equal_returns_all_zero_array(input, expected_output):
    np.testing.assert_array_equal(functional.min_max_normalize(np.array(input)), np.array(expected_output))

