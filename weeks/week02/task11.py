"""
Create a script weeks/week2/task11.py that prints the output range of sigmoid, 
relu, and tanh over [-10, 10], prints all three at x = -5, 0, 5, and shows a 
plot of all three on the same axes.

Show how torch.relu and torch.tanh can be used to obtain the same results.

Acceptance criteria:

* Calculations are done via numpy.
* PyTorch is used to show the built-in way of calculating the same functions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from dl_lib.activations.functional import sigmoid, relu, tanh

def main() -> int:

	wire = np.linspace(-10, 10, 10_000) 
	sigmoid_wire = [sigmoid(x) for x in wire]
	relu_wire = [relu(x) for x in wire]
	tanh_wire = [tanh(x) for x in wire]

	print(f"output range of sigmoid = [{min(sigmoid_wire):4f}, {max(sigmoid_wire):4f}]")
	print(f"output range of relu = [{min(relu_wire):4f}, {max(relu_wire):4f}]")
	print(f"output range of tanh = [{min(tanh_wire):4f}, {max(tanh_wire):4f}]")
	print()

	print_points = [-5, 0, 5]

	for x in print_points:
		print(f"sigmoid({x})={sigmoid(x):4f}")
		print(f"relu({x})=   {relu(x):4f}")
		print(f"tanh({x})=   {tanh(x):4f}")
		print()


	plt.plot(wire, sigmoid_wire)
	plt.plot(wire, relu_wire)
	plt.plot(wire, tanh_wire)

	plt.grid()
	plt.legend(["sigmoid", "relu", "tanh"])
	plt.xlabel("x")
	plt.ylabel("activation value")
	plt.title("Sigmoid, ReLU and Tanh between -10 and 10")

	plt.savefig("plots/week02/task11.png")

	torch.nn.functional.sigmoid(torch.from_numpy(wire))
	torch.nn.functional.relu(torch.from_numpy(wire))
	torch.nn.functional.tanh(torch.from_numpy(wire))

	return 0