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