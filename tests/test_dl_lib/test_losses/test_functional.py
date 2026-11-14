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


# Mean Squared Error

@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_output"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0], 1.75),
    ],
)
def test_mse_when_called_then_matches_hand_calculation(y_true, y_pred, expected_output):
    assert np.isclose(functional.mse(np.array(y_true), np.array(y_pred)), expected_output)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0]),
    ],
)
def test_mse_when_called_then_matches_torch_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    expected_output = torch.nn.functional.mse_loss(
                                    torch.from_numpy(y_pred),
                                    torch.from_numpy(y_true)).item()

    assert np.isclose(functional.mse(y_pred, y_true), expected_output)

@pytest.mark.parametrize(
    ("y_true", "y_pred_close", "y_pred_far"),
    [
        ([1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 1.0], [10.0, 1.0, 1.0, 1.0]),
    ],
)
def test_when_one_prediction_is_far_off_then_penalizes_it_more_than_mae_does(y_true, y_pred_close, y_pred_far):
    y_true = np.array(y_true)
    y_pred_close = np.array(y_pred_close)
    y_pred_far = np.array(y_pred_far)

    mae_ratio = functional.mae(y_true, y_pred_far) / functional.mae(y_true, y_pred_close)
    mse_ratio = functional.mse(y_true, y_pred_far) / functional.mse(y_true, y_pred_close)

    assert np.greater(mse_ratio, mae_ratio)

# Root Mean Squared Error

@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_output"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0], 1.3229),
    ],
)
def test_rmse_when_called_then_matches_hand_calculation(y_true, y_pred, expected_output):
    np.testing.assert_almost_equal(functional.rmse(np.array(y_true), np.array(y_pred)), expected_output, decimal=4)

@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([3.0, 5.0, 7.0, 9.0], [2.0, 6.0, 8.0, 7.0]),
    ],
)
def test_when_called_then_equals_square_root_of_mse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        expected = np.sqrt(functional.mse(y_true, y_pred))

        actual = functional.rmse(y_true, y_pred)

        np.testing.assert_almost_equal(actual, expected)


# Binary Cross Entropy

def test_binary_cross_entropy_when_called_then_matches_hand_calculation():
    # Arrange
    y_true = np.array([1.0, 0.0])
    p = np.array([0.9, 0.1])
    expected = -np.log(0.9)

    # Act
    actual = functional.binary_cross_entropy(y_true, p)

    # Assert
    assert pytest.approx(expected) == actual

def test_binary_cross_entropy_when_called_then_matches_torch_binary_cross_entropy():
    # Arrange
    y_true = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.2, 0.6])
    expected = torch.nn.functional.binary_cross_entropy(torch.from_numpy(p),
                                                        torch.from_numpy(y_true)).item()

    # Act
    actual = functional.binary_cross_entropy(y_true, p)

    # Assert
    assert pytest.approx(expected) == actual

def test_binary_cross_entropy_when_confidently_wrong_then_penalizes_more_than_confidently_right():
    # Arrange
    y_true = np.array([1.0])

    # Act
    confidently_right = functional.binary_cross_entropy(y_true, np.array([0.999]))
    confidently_wrong = functional.binary_cross_entropy(y_true, np.array([0.001]))

    # Assert
    assert confidently_wrong > confidently_right


# Categorical Cross Entropy

# def test_categorical_cross_entropy_when_called_then_matches_hand_calculation():
#     # Arrange
#     y_true = np.array([0, 1, 2])
#     probabilities = np.array([
#         [0.7, 0.2, 0.1],
#         [0.1, 0.8, 0.1],
#         [0.2, 0.2, 0.6],
#     ])
#     expected = -np.mean(np.log([0.7, 0.8, 0.6]))

#     # Act
#     actual = functional.categorical_cross_entropy(y_true, probabilities)

#     # Assert
#     assert pytest.approx(expected) == actual

# def test_categorical_cross_entropy_when_called_then_matches_torch_nll_loss_on_log_probabilities():
#     # Arrange
#     y_true = np.array([0, 1, 2])
#     probabilities = np.array([
#         [0.7, 0.2, 0.1],
#         [0.1, 0.8, 0.1],
#         [0.2, 0.2, 0.6],
#     ])
#     expected = torch.nn.functional.nll_loss(torch.log(torch.from_numpy(probabilities)),
#                                             torch.from_numpy(y_true)).item()

#     # Act
#     actual = functional.categorical_cross_entropy(y_true, probabilities)

#     # Assert
#     assert pytest.approx(expected) == actual

# def test_categorical_cross_entropy_when_confidently_wrong_then_penalizes_more_than_confidently_right():
#     # Arrange
#     y_true = np.array([0])

#     # Act
#     confidently_right = functional.categorical_cross_entropy(
#         y_true, np.array([[0.999, 0.0005, 0.0005]]))
#     confidently_wrong = functional.categorical_cross_entropy(
#         y_true, np.array([[0.001, 0.4995, 0.4995]]))

#     # Assert
#     assert confidently_wrong > confidently_right
