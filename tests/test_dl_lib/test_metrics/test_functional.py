import pytest
import numpy as np

from dl_lib.metrics import functional

@pytest.mark.parametrize(
    ("probabilities", "threshold", "expected"),
    [
        (np.array([0.1, 0.8, 0.4, 0.7, 0.9]), 0.5, np.array([0, 1, 0, 1, 1])),
    ],
)
def test_apply_threshold_when_probabilities_straddle_threshold_then_splits_them(probabilities, threshold, expected):
        # Act
        actual = functional.apply_threshold(probabilities, threshold)

        # Assert
        np.testing.assert_array_equal(actual, expected)

def test_apply_threshold_when_threshold_rises_then_fewer_predictions_are_positive():
        # Arrange
        probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])

        # Act
        low_threshold_positives = functional.apply_threshold(probabilities, 0.1).sum()
        high_threshold_positives = functional.apply_threshold(probabilities, 0.9).sum()

        # Assert
        assert low_threshold_positives > high_threshold_positives