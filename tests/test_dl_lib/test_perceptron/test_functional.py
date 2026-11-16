import pytest
import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
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

@pytest.mark.parametrize(
    ("x", "weights", "bias", "expected"),
    [
        (np.array([1.0]), np.array([0.0]), 0.0, 0.5),
        (np.array([10000.0]), np.array([10000.0]), 10000.0, 1.0),
    ]
)
def test_forward_when_an_activation_is_given_then_applies_it_to_the_weighted_sum(x, weights, bias, expected):
    # Act
    actual = functional.forward(x, weights, bias=bias, activation=sigmoid)

    # Assert
    assert pytest.approx(actual) == expected

# Numerical Gradient

def test_numerical_gradient_when_called_on_a_quadratic_loss_then_matches_the_analytical_derivative():
    # Arrange
    def loss_fn(params: np.ndarray) -> np.float64:
        return np.sum(params ** 2)

    params = np.array([3.0, -2.0])
    expected = np.array([6.0, -4.0])

    # Act
    actual = functional.numerical_gradient(loss_fn, params)

    # Assert
    np.testing.assert_allclose(actual, expected, atol=1e-4)

def test_when_params_is_two_dimensional_then_gradient_matches_the_analytical_derivative():
    # Arrange
    def loss_fn(params: np.ndarray) -> np.float64:
        return np.sum(params**2)

    params = np.array([[3.0, -2.0], [1.0, 0.5]])
    expected = np.array([[6.0, -4.0], [2.0, 1.0]])

    # Act
    actual = functional.numerical_gradient(loss_fn, params)

    # Assert
    np.testing.assert_allclose(actual, expected, atol=1e-4)

def test_when_called_then_leaves_params_unchanged():
    # Arrange
    def loss_fn(params: np.ndarray) -> np.float64:
        return np.sum(params ** 2)

    params = np.array([3.0, -2.0])
    expected = np.array([3.0, -2.0])

    # Act
    functional.numerical_gradient(loss_fn, params)

    # Assert
    np.testing.assert_array_equal(params, expected)

# Gradient Descent Step

def test_when_called_then_moves_params_against_the_gradient():
    # Arrange
    params = np.array([1.0, 1.0])
    gradient = np.array([2.0, -2.0])
    learning_rate = 0.1
    expected = np.array([0.8, 1.2])

    # Act
    actual = functional.gradient_descent_step(params, gradient, learning_rate)

    # Assert
    np.testing.assert_allclose(actual, expected)


# Analytical Gradient
def test_analytical_gradient_when_no_activation_then_matches_the_closed_form_gradient():
    # Arrange
    dataset = [
        (np.array([0.0]), 0.0),
        (np.array([1.0]), 2.0),
        (np.array([2.0]), 4.0),
    ]
    weights = np.array([1.0])
    expected_weight_gradient = np.array([-10.0 / 3.0])
    expected_bias_gradient = -2.0

    # Act
    actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
        dataset, weights
    )

    # Assert
    np.testing.assert_allclose(actual_weight_gradient, expected_weight_gradient)
    assert actual_bias_gradient == pytest.approx(expected_bias_gradient)

def test_analytical_gradient_when_activation_given_then_matches_the_numerical_gradient():
    # Arrange
    dataset = [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0]), 0.0),
        (np.array([1.0, 1.0]), 1.0),
    ]
    features = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    weights = np.array([0.5, -0.1])
    bias = 0.7
    activation = lambda z: 1.0 / (1.0 + np.exp(-z))
    activation_derivative = lambda z: activation(z) * (1.0 - activation(z))

    weights_loss = lambda w: functional.dataset_loss(features, targets, w, bias, activation=activation)
    bias_loss = lambda b: functional.dataset_loss_given_bias(features, targets, weights, b, activation=activation)

    expected_weight_gradient = functional.numerical_gradient(weights_loss, weights)
    expected_bias_gradient = functional.numerical_gradient(
        bias_loss, np.array([bias])
    )[0]

    # Act
    actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
        dataset,
        weights,
        bias,
        activation=activation,
        activation_derivative=activation_derivative,
    )

    # Assert
    np.testing.assert_allclose(
        actual_weight_gradient, expected_weight_gradient, atol=1e-4
    )
    assert actual_bias_gradient == pytest.approx(expected_bias_gradient, abs=1e-4)

def test_analytical_gradient_when_bce_loss_then_matches_the_numerical_gradient_without_activation_derivative():
    # Arrange
    dataset = [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0]), 0.0),
        (np.array([1.0, 1.0]), 1.0),
    ]
    features = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    weights = np.array([0.5, -0.1])
    bias = 0.7
    activation = lambda z: 1.0 / (1.0 + np.exp(-z))

    weights_loss = lambda w: functional.dataset_loss(
        features,
        targets,
        w,
        bias,
        activation=activation,
        loss=binary_cross_entropy,
    )
    bias_loss = lambda b: functional.dataset_loss_given_bias(
        features,
        targets,
        weights,
        b,
        activation,
        loss=binary_cross_entropy,
    )
    expected_weight_gradient = functional.numerical_gradient(weights_loss, weights)
    expected_bias_gradient = functional.numerical_gradient(
        bias_loss, np.array([bias])
    )[0]

    # Act
    actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
        dataset,
        weights,
        bias,
        activation=activation,
        loss="bce",
    )

    # Assert
    np.testing.assert_allclose(
        actual_weight_gradient, expected_weight_gradient, atol=1e-4
    )
    assert actual_bias_gradient == pytest.approx(expected_bias_gradient, abs=1e-4)
