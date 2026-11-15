import pytest
from dl_lib.perceptron import datasets


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, []),
        (1, [(0, 0)]),
        (2, [(0, 0), (1, 2)]),
        (3, [(0, 0), (1, 2), (2, 4)]),
        (4, [(0, 0), (1, 2), (2, 4), (3, 6)]),
    ],
)
def test_when_called_then_returns_n_consecutive_x_and_2x_pairs(n, expected):
	actual = datasets.create_dataset(n)

	assert expected == actual