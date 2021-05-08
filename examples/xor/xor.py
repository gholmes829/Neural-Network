#!/usr/bin/env python3
"""
Learning non-linear xor function.
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
for i in range(2):
	parentdir = os.path.dirname(currentdir)
	sys.path.append(parentdir)
	currentdir = parentdir

import numpy as np
from matplotlib import pyplot as plt

from neural_net import NeuralNetwork


def plotCosts(costs):
	t = np.linspace(0, len(costs)-1, len(costs))
	plt.plot(t, costs)
	plt.xlabel("Iterations")
	plt.ylabel("Cost")
	plt.title("Neural Network Training Performance")
	plt.show()


def main():
	inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	labels = np.array([[0], [1], [1], [0]])

	architecture = (2, 3, 1)
	nn = NeuralNetwork(architecture, activation="sigmoid", cost="mse")
	
	costs = nn.train(inputs, labels, alpha=1, iterations=5000)
	print("\nCost always decreases:", all([costs[i+1] < costs[i] for i in range(len(costs)-1)]))
	print("Lowest cost:", min(costs))
	print("\nResults:", "\n", np.round(nn.evaluate(inputs), 0))
	plotCosts(costs)

	# GRAPHING
	plt.style.use(["dark_background"])
	#plt.rc("grid", alpha=0.25)

	start, end = -0.5, 1.5

	fidelity = 0.01
	n = int((end-start)/fidelity)
	points = np.meshgrid(np.linspace(start, end, n+1), np.linspace(start, end, n+1))
	values = np.zeros((n+1, n+1))
	
	for i in range(n+1):
		for j in range(n+1):
			values[i, j] = nn.evaluate(np.array([points[0][i, j], points[1][i, j]]))

	x, y = points
	# RdYlGn
	plt.contourf(x, y, values, np.linspace(0, 1, 51), cmap="jet_r")
	plt.colorbar()
	plt.contour(x, y, values, 0.5, linewidths=2, linestyles="dashed", colors="black")
		
	plt.grid(color="black", alpha=0.25)
	plt.axhline(y=0, color="k", lw=2)
	plt.axvline(x=0, color="k", lw=2)
	plt.title("Neural Network XOR Boundary")
	plt.xlabel("X Axis")
	plt.ylabel("Y Axis")
	plt.show()

if __name__ == "__main__":
	main()

