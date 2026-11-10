import pytest
import torch
import numpy as np

from dl_lib.losses import functional

# Mean Absolute Error

@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_output"),
    [
        ([1], [1], 0),
        ([1, 1], [1, 1], 0),
        ([1, 2], [1, 2], 0),
        ([7.0, 7.0, 7.0], [7.0, 7.0, 7.0], 0),
    ],
)
def test_mae_when_all_values_equal_returns_zero(y_true, y_pred, expected_output):
    assert np.isclose(functional.mae(np.array(y_true), np.array(y_pred)), expected_output)

@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_output"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0], 1.25),
    ],
)
def test_mae_when_called_matches_hand_calculation(y_true, y_pred, expected_output):
    assert np.isclose(functional.mae(np.array(y_true), np.array(y_pred)), expected_output)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0]),
    ],
)
def test_mae_when_called_matches_l1_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    expected_output = torch.nn.functional.l1_loss(torch.from_numpy(y_pred),
                                                torch.from_numpy(y_true)).item()
    
    assert np.isclose(functional.mae(y_true, y_pred), expected_output)

def test_mae_when_empty_arrays_raises_value_error():
    with pytest.raises(ValueError):
        functional.mae(np.array([]), np.array([]))

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([1], [1, 1]),
        ([1, 1], [1]),
        ([1, 2], [[1], [2]]),
        ([], [7.0, 7.0, 7.0]),
    ],
)
def test_mae_when_arrays_are_different_shape_raises_value_error(y_true, y_pred):
     with pytest.raises(ValueError):
        functional.mae(np.array(y_true), np.array(y_pred))

