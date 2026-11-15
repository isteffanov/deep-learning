import pytest
import numpy as np

from dl_lib.perceptron import functional

@pytest.mark.parametrize(
    ("low", "high", "size"),
    [
        (0.0, 10.0, 0),
        (0.0, 10.0, 5),
        (0.0, 10.0, 10),
        (-100.0, 100.0, 50),
    ]
)
def test_initialize_weights_when_called_then_returns_size_many_values_in_the_given_range(low, high, size):
    actual = functional.initialize_weights(low, high, size=size)

    # Assert
    assert actual.shape == (size,)
    assert np.all(actual >= low)
    assert np.all(actual < high)

@pytest.mark.parametrize(
    ("low", "high", "seed"),
    [
        (0.0, 10.0, 42),
    ]
)
def test_initialize_weights_when_seed_is_fixed_then_result_is_reproducible(low, high, seed):
    first = functional.initialize_weights(low, high, size=3, seed=seed)
    second = functional.initialize_weights(low, high, size=3, seed=seed)

    # Assert
    np.testing.assert_array_equal(first, second)

# Forward

@pytest.mark.parametrize(
    ("x", "weights", "expected"),
    [
        (np.array([]), np.array([]), 0),
        (np.array([2.0, 3.0]), np.array([1.0, 2.0]), 8)
    ]
)
def test_forward_when_activation_is_none_then_returns_the_weighted_sum(x, weights, expected):
    # Act
    actual = functional.forward(x, weights)

    # Assert
    assert pytest.approx(expected) == actual

@pytest.mark.parametrize(
    ("x", "weights", "bias", "expected"),
    [
        (np.array([]), np.array([]), 1.0, 1.0),
        (np.array([2.0, 3.0]), np.array([1.0, 2.0]), 0.5, 8.5)
    ]
)
def test_forward_when_activation_is_none_has_bias_then_returns_the_weighted_sum_plus_bias(x, weights, bias, expected):
    # Act
    actual = functional.forward(x, weights, bias=bias)

    # Assert
    assert pytest.approx(expected) == actual
