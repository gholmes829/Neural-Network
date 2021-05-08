#!/usr/bin/env python3
"""
Shows that neural networks can approximate arbitrary functions.
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


def f(x):
	return x**2 * np.cos(0.25*x)

def plotCosts(costs):
	t = np.linspace(0, len(costs)-1, len(costs))
	plt.plot(t, costs, color="green")
	plt.xlabel("Iterations")
	plt.ylabel("Cost")
	plt.title("Neural Network Training Performance")
	plt.show()

def rmse(labels, predictions) -> any:
		"""Mean squared error."""
		m = labels.shape[0]
		return (1/(m)) * np.sqrt(np.sum(np.square(labels - predictions)))

def main():
	print("Neural Network Function Approximation!\n")
	n = 100

	trainingData = []
	trainingLabels = []

	rawTrainingData = [i/4 for i in range(n)]
	trainingMean, trainingStd = np.mean(rawTrainingData), np.std(rawTrainingData)

	trainingData = (np.array([[i] for i in rawTrainingData]) - trainingMean) / trainingStd
	trainingLabels = np.array([[f(i)] for i in rawTrainingData])

	architecture = (1, 10, 10, 1)
	nn = NeuralNetwork(architecture, activation="tanh", outputActivation="linear", cost="mse")

	initialGuess = nn.evaluate(trainingData)

	costs = nn.train(trainingData, trainingLabels, alpha=0.0005, iterations=25000)

	finalGuess = nn.evaluate(trainingData)
	print("\nCost always decreases:", all([costs[i+1] < costs[i] for i in range(len(costs)-1)]))
	print("Initial error:", rmse(trainingLabels, initialGuess))
	print("Final error:", rmse(trainingLabels, finalGuess))

	plt.style.use(["dark_background"])
	plt.plot(rawTrainingData, trainingLabels, lw=4, label="Truth", color="cyan")
	plt.plot(rawTrainingData, finalGuess, lw=3, label="Trained Approx", color="red")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("f(x)=x^2 * cos(0.25*x)")
	plt.legend()
	plt.show()	
	plotCosts(costs)

	print("\nDone!")

if __name__ == "__main__":
	main()
