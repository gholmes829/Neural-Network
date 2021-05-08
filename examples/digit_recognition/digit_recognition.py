#!/usr/bin/env python3

"""
MNIST handwritten digit classification. 
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
for i in range(2):
	parentdir = os.path.dirname(currentdir)
	sys.path.append(parentdir)
	currentdir = parentdir

import numpy as np
from matplotlib import pyplot as plt
from time import time

import neural_net

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


def showImage(image):
	plt.imshow(image, cmap="gray")
	plt.show()

def plotCosts(costs):
	t = np.linspace(0, len(costs)-1, len(costs))
	plt.plot(t, costs)
	plt.xlabel("Iterations")
	plt.ylabel("Cost")
	plt.title("Neural Network Training Performance")
	plt.show()

def main():
	print("DIGIT RECOGNITION")
	
	print("\nLoading data...")
	with np.load("mnist.npz") as data: 
		trainingSet = np.squeeze(data["training_images"])
		trainingLabels = np.squeeze(data["training_labels"])
		testingSet = np.squeeze(data["test_images"])
		testingLabels = np.squeeze(data["test_labels"])
		validationSet = np.squeeze(data["validation_images"])
		validationLabels = np.squeeze(data["validation_labels"])

	#showImage(testingSet[0].reshape(28, 28))
	
	trainingSize = min(1000, trainingSet.shape[0])  
	architecture = (784, 24, 10)
	nn = neural_net.NeuralNetwork(architecture, activation="sigmoid", cost="crossEntropy")

	initialTestingAccuracy = nn.classificationAccuracy(testingSet, testingLabels)
	initialGuess = np.argmax(nn.evaluate(testingSet[0]))

	timer = time()
	costs = nn.train(trainingSet[:trainingSize], trainingLabels[:trainingSize], alpha=3, iterations=3000, threshold=0.00001)
	elapsed = round(time()-timer, 3)
	trainingAccuracy = nn.classificationAccuracy(trainingSet[:trainingSize], trainingLabels[:trainingSize])
	finalTestingAccuracy = nn.classificationAccuracy(testingSet, testingLabels)

	print("\nTime elapsed:", elapsed, "sec")	

	print("\nInitial guess:", initialGuess)
	print("Final guess:", np.argmax(nn.evaluate(testingSet[0])))
	print("Correct guess:", np.argmax(testingLabels[0]))

	print("\nCost always decreases:", all([costs[i+1] < costs[i] for i in range(len(costs)-1)]))

	print("\nInitial testing accuracy:", round(initialTestingAccuracy, 3), "%")	
	print("Training accuracy:", round(trainingAccuracy, 3), "%")
	print("Final testing accuracy:", round(finalTestingAccuracy, 3), "%")	

	t = np.linspace(0, len(costs)-1, len(costs))
	plt.plot(t, costs)
	plt.show()

	save = input("\nOverwrite saved model? (y/n): ")
	
	if save in {"y", "Y"}:
		path = os.path.join(os.getcwd(), "saved_model.npz")
		np.savez(path, weights=np.array(nn.weights, dtype=object), biases=np.array(nn.biases, dtype=object))

if __name__ == "__main__":
	main()


