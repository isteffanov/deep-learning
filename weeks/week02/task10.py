"""
Create a script weeks/week2/task10.py that prints sigmoid at -10, 0, and 10, 
prints the output range over [-10, 10], and shows a plot of sigmoid(x) over 
that range.

Lastly, show how PyTorch can also be used to calculate sigmoid via torch.sigmoid.

Acceptance criteria:

* Calculations are done via numpy.
* PyTorch is used in the script to show the built-in way of calculating sigmoid.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch

from dl_lib.activations.functional import sigmoid

def main() -> int:

	print_points = [-10, 0, 10]

	for x in print_points:
		print(f"sigmoid({x})={sigmoid(x):4f}")

	print(f"\noutput range = [{min(sigmoid(x) for x in print_points):4f}, {max(sigmoid(x) for x in print_points):4f}]")


	points = np.linspace(-10, 10, 1_000)
	values = np.array([sigmoid(x) for x in points])

	plt.plot(points, values)
	plt.grid()
	plt.legend(["sigmoid(x)"])

	plt.xlabel("x")
	plt.ylabel("sigmoid(x)")
	plt.title("sigmoid(x) graph from -10 to 10")

	plt.savefig("plots/week02/task10.png")

	torch.nn.functional.sigmoid(torch.from_numpy(points))

	return 0