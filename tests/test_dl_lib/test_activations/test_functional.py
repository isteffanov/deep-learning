import pytest
import torch

import numpy as np

from dl_lib.activations import functional

# Sigmoid

def test_sigmoid_when_input_is_zero_then_returns_one_half():
    # Arrange
    x = np.array([0.0])
    expected = 0.5

    # Act
    actual = functional.sigmoid(x)[0]

    # Assert
    assert pytest.approx(expected) == actual

def test_sigmoid_when_input_is_very_negative_then_approaches_zero():
    # Arrange
    x = np.array([-100.0])
    expected = 0.0

    # Act
    actual = functional.sigmoid(x)[0]

    # Assert
    assert pytest.approx(expected) == actual

def test_sigmoid_when_input_is_very_positive_then_approaches_one():
    # Arrange
    x = np.array([100.0])
    expected = 1.0

    # Act
    actual = functional.sigmoid(x)[0]

    # Assert
    assert pytest.approx(expected) == actual

def test_sigmoid_when_called_then_matches_torch_sigmoid():
    # Arrange
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = torch.sigmoid(torch.from_numpy(x)).numpy()

    # Act
    actual = functional.sigmoid(x)

    # Assert
    np.testing.assert_allclose(actual, expected)


# ReLU

def test_relu_when_input_is_negative_then_returns_zero():
    # Arrange
    x = np.array([-5.0, -1.0])
    expected = np.array([0.0, 0.0])

    # Act
    actual = functional.relu(x)

    # Assert
    np.testing.assert_allclose(actual, expected)

def test_relu_when_input_is_positive_then_returns_input_unchanged():
    # Arrange
    x = np.array([2.0, 7.5])

    # Act
    actual = functional.relu(x)

    # Assert
    np.testing.assert_allclose(actual, x)

def test_relu_when_called_then_matches_torch_relu():
    # Arrange
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = torch.nn.functional.relu(torch.from_numpy(x)).numpy()

    # Act
    actual = functional.relu(x)

    # Assert
    np.testing.assert_allclose(actual, expected)


# Tanh

def test_when_input_is_zero_then_returns_zero():
    # Arrange
    x = np.array([0.0])
    expected = 0.0

    # Act
    actual = functional.tanh(x)[0]

    # Assert
    pytest.approx(expected) == actual

def test_when_input_is_very_negative_then_approaches_negative_one():
    # Arrange
    x = np.array([-100.0])

    # Act
    actual = functional.tanh(x)[0]

    # Assert
    pytest.approx(actual) == -1.0

def test_when_called_then_matches_torch_tanh():
    # Arrange
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = torch.tanh(torch.from_numpy(x)).numpy()

    # Act
    actual = functional.tanh(x)

    # Assert
    np.testing.assert_allclose(actual, expected)


# Softmax
def test_softmax_when_called_then_probabilities_sum_to_one():
    # Arrange
    x = np.array([1.0, 2.0, 3.0])
    expected = 1.0

    # Act
    actual = functional.softmax(x).sum()

    # Assert
    assert pytest.approx(expected) == actual

def test_softmax_when_input_has_large_values_then_does_not_overflow():
    # Arrange
    x = np.array([1000.0, 1001.0, 1002.0])

    # Act
    actual = functional.softmax(x)

    # Assert
    assert np.all(np.isfinite(actual))
    assert pytest.approx(actual.sum()) == 1.0

def test_softmax_when_called_then_matches_torch_softmax():
    # Arrange
    x = np.array([1.0, 2.0, 3.0])
    expected = torch.softmax(torch.from_numpy(x), dim=-1).numpy()

    # Act
    actual = functional.softmax(x)

    # Assert
    np.testing.assert_allclose(actual, expected)

def test_softmax_when_input_is_batched_then_normalizes_each_row_independently():
    # Arrange
    x = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    expected_row_sums = np.array([1.0, 1.0])

    # Act
    actual = functional.softmax(x)

    # Assert
    np.testing.assert_allclose(actual.sum(axis=-1), expected_row_sums)