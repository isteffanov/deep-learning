
import pytest
import numpy as np

from dl_lib.preprocessing import functional

# min_max_normalize

@pytest.mark.parametrize(
    ("input", "expected_output"),
    [
        ([], []),
        ([1], [0]),
        ([1, 1], [0, 0]),
        ([7.0, 7.0, 7.0], [0.0, 0.0, 0.0]),
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
        ([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_min_max_normalize_when_differing_values_returns_rescaled_array(input, expected_output):
    np.testing.assert_allclose(functional.min_max_normalize(np.array(input)), np.array(expected_output))

@pytest.mark.parametrize(
    "input",
    [
        5,                                  # 0D: a single scalar
        [[1, 2], [3, 4]],                   # 2D: a matrix
        [[1, 2, 3]],                        # 2D: one row, but still 2D
        [[1], [2], [3]],                    # 2D: one column
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # 3D
        np.empty((0, 3)),                   # 2D and empty
    ],
)
def test_min_max_normalize_when_shape_is_not_1d_throws_exception(input):
    with pytest.raises(ValueError):
        functional.min_max_normalize(np.array(input))

# standardize

@pytest.mark.parametrize(
    ("input", "expected_output"),
    [
        ([], [])
    ],
)
def test_standardize_when_all_values_equal_returns_all_zero_array(input, expected_output):
    np.testing.assert_array_equal(functional.standardize(np.array(input)), np.array(expected_output))