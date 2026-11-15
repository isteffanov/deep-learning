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
