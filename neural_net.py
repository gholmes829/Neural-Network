"""
Contains neural networks.

Classes
-------
NeuralNetwork
	Neural network
"""

import numpy as np

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class NeuralNetwork:
	"""
	Classic deep neural network.

	Attributes
	----------
	layerSizes -> list
		Info about sizes of each layer
	weightShapes -> list
		Info about shapes of weights for each layer
	weights -> np.ndarray
		Array of weights for each layer
	biases -> np.ndarray
		Array of biases for each layer
	activation -> callable
		Non-linear activation function for neurons

	Public Methods
	--------------
	evaluate(inputs) -> np.ndarray:
		Feeds inputs through neural net to obtain output.
	"""
	def __init__(self, layerSizes: tuple, activation: str = "sigmoid", outputActivation: str = "sigmoid", cost: str = "mse", weights: list = None, biases: list = None) -> None:
		"""
		Initializes network.

		Parameters
		----------
		layerSizes: tuple
			Layer architecture
		activation: str, default="sigmoid"
			String denoting activation function to use
		weights: list, optional
			List of arrays of weights for each layer, randomized if not passed in
		biases: list, optional
			List of arrays of biases for each layer, randomized if not passed in
		"""
		activations = {
			"linear": NeuralNetwork.linear,
			"sigmoid": NeuralNetwork.sigmoid,
			"tanh": NeuralNetwork.tanh,
			"reLu": NeuralNetwork.reLu
		}
	
		activationGradients = {
			"linear": NeuralNetwork.linearGradient,
			"sigmoid": NeuralNetwork.sigmoidGradient,
			"tanh": NeuralNetwork.tanhGradient,
			"reLu": NeuralNetwork.reLuGradient
		}
		
		costs = {
			"mse": NeuralNetwork.mse,
			"crossEntropy": NeuralNetwork.crossEntropy
		}

		costGradients = {
			"mse": NeuralNetwork.mseGradient,
			"crossEntropy": NeuralNetwork.crossEntropyGradient
		}

		self.layerSizes = layerSizes
		weightShapes = [(i, j) for i, j in zip(layerSizes[1:], layerSizes[:-1])]
		self.weights = [np.random.randn(*s) for s in weightShapes] if weights is None else weights
		self.biases = [np.random.standard_normal(s) for s in self.layerSizes[1:]] if biases is None else biases

		self.activation = activations[activation]
		self.activationGradient = activationGradients[activation]

		self.outputActivation = activations[outputActivation]
		self.outputActivationGradient = activationGradients[outputActivation]

		self.cost = costs[cost]
		self.costGradient = costGradients[cost]

	def evaluate(self, a: np.ndarray) -> np.ndarray:
		"""
		Feeds inputs through neural net to obtain output.

		Parameters
		----------
		a: np.ndarray
			Inputs to neural network

		Returns
		-------
			np.ndarray: input after fed through neural network
		"""
		for w, b in zip(self.weights[:-1], self.biases[:-1]):
			a = self.activation(a @ w.T + b)
		output = self.outputActivation(a @ self.weights[-1].T + self.biases[-1])
		return output

	def train(self, trainingSet: np.ndarray, labels: np.ndarray, alpha: float, iterations: int = 0, threshold: float = 0., output: str = True) -> list:
		costs = []
		costs.append(self.cost(labels, self.evaluate(trainingSet)))

		cost = self.cost(labels, self.evaluate(trainingSet))
		dc = threshold + 1

		if output and iterations:
			print("Progress: 0.0%", end=" ", flush=True)

		i = 0
		while (not iterations or i < iterations) and (not threshold or dc > threshold):
			if output and (iterations and (i + 1) % int(iterations / 10) == 0):
				print(round((100 * (i + 1)) / iterations, 2), end="% ", flush=True)
			
			weightGradient, biasGradient = self.backpropogate(trainingSet, labels)
			self.weights = [self.weights[i] - alpha * weightGradient[i] for i in range(len(self.weights))]
			self.biases = [self.biases[i] - alpha * biasGradient[i] for i in range(len(self.biases))]
			cost = self.cost(labels, self.evaluate(trainingSet))
			costs.append(cost)
			dc = costs[-2] - costs[-1]
			i += 1

		if output and iterations:
			print()		
	
		if output and (threshold and not dc > threshold):
			print("Terminating bc threshold passed at iteration " + str(i+1) + ":", round(dc, 6), "<", threshold)

		return costs

	def classificationAccuracy(self, testingSet: np.ndarray, labels: np.ndarray) -> float:
		predictions = self.evaluate(testingSet)
		resultLabels = predictions.copy()
		for i, result in enumerate(predictions):
			resultLabels[i] = np.zeros(10)
			resultLabels[i][np.argmax(result)] = 1
		
		correct = 0
		for i, result in enumerate(resultLabels):
			if np.all(result==labels[i]):
				correct += 1
		
		return (100 * correct) / testingSet.shape[0]

	def backpropogate(self, trainingSet: np.ndarray, labels: np.ndarray) -> None:
		"""
		Uses backpropogation algorithm to train network.
		"""
		m = len(trainingSet)

		activations, rawNeurons = [], []
		activations.append(trainingSet)
		deltas = []
		
		for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
			z = activations[i] @ w.T + b
			rawNeurons.append(z)
			a = self.activation(z)
			activations.append(a)

		z = activations[-1] @ self.weights[-1].T + self.biases[-1]
		rawNeurons.append(z)
		a = self.outputActivation(z)
		activations.append(a)
			
		outputDelta = self.costGradient(labels, activations[-1]) * self.outputActivationGradient(rawNeurons[-1])
		deltas.append(outputDelta)
		for i in range(len(self.layerSizes) - 2):
			delta = (deltas[i] @ self.weights[-i - 1]) * self.activationGradient(rawNeurons[-i - 2])
			deltas.append(delta)

		deltas.reverse()

		weightGradient = []
		biasGradient = []

		for i in range(len(self.layerSizes)-1):
			weightGradient.append((1 / m) * deltas[i].T @ activations[i])
			biasGradient.append((1 / m) * deltas[i].T @ np.ones(m))

		# UNCOMMENT TO CHECK GRADIENT OF NETWORK USED IN DIGIT RECOGNITION  WHEN RUNNING DIGIT RECOGNITION
		#dW1, dB1 = weightGradient[-1][5, 3], biasGradient[-1][2]
		#dW2, dB2 = self.numCheck(trainingSet, labels)
		#print("Backprop:", float(dW1), float(dB1))
		#print("Numerical:", float(dW2), float(dB2))
		#print()

		return weightGradient, biasGradient

	def numCheck(self, trainingSet, labels):
		e = 0.00001

		savedWeights = self.weights[-1].copy()
		savedBias = self.biases[-1].copy()
		
		weight = self.weights[-1][5, 3]
		neg, pos = weight - e, weight + e
	
		self.weights[-1][5, 3] = neg
		predictions = self.evaluate(trainingSet)
		backLoss = self.cost(labels, predictions)

		self.weights[-1][5, 3] = pos
		predictions = self.evaluate(trainingSet)
		fwdLoss = self.cost(labels, predictions)
		
		dW = (fwdLoss - backLoss) / (2 * e)

		self.weights[-1] = savedWeights

		bias = self.biases[-1][2]
		neg, pos = bias - e, bias + e

		self.biases[-1][2] = neg
		predictions = self.evaluate(trainingSet)
		backLoss = self.cost(labels, predictions)

		self.biases[-1][2] = pos
		predictions = self.evaluate(trainingSet)
		fwdLoss = self.cost(labels, predictions)
		
		dB = (fwdLoss - backLoss) / (2 * e)

		self.biases[-1] = savedBias
		
		return dW, dB	

	# COSTS
	@staticmethod
	def mse(labels: np.ndarray, predictions: np.ndarray) -> any:
		"""Mean squared error."""
		m = labels.shape[0]
		return (1 / (2 * m)) * np.sum(np.square(labels - predictions))

	@staticmethod
	def mseGradient(labels: np.ndarray, predictions: np.ndarray) -> any:
		"""Mean squared error gradient."""
		m = labels.shape[0]
		return predictions - labels

	@staticmethod
	def crossEntropy(labels: np.ndarray, predictions: np.ndarray) -> any:
		"""Cross entropy for classification error."""
		m = labels.shape[0]
		return -(1 / m) * np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

	@staticmethod
	def crossEntropyGradient(labels: np.ndarray, predictions: np.ndarray) -> any:
		"""Cross entropy gradient."""		
		m = labels.shape[0]
		return -((labels / predictions) - ((1 - labels) / (1 - predictions)))

	# ACTIVATIONS
	@staticmethod
	def sigmoid(z: any) -> any:
		"""Sigmoid."""
		return 1 / (1 + np.exp(-z))

	@staticmethod
	def sigmoidGradient(z: any) -> any:
		"""Sigmoid derivative."""
		return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))

	@staticmethod
	def reLu(x: any) -> any:
		"""Rectified linear unit."""
		return np.maximum(0, x)
	
	@staticmethod
	def reLuGradient(x: any) -> any:
		"""reLu derivative."""
		return np.where(x > 0, x, x * 0.01)

	@staticmethod
	def tanh(x: any) -> any:
		"""Hyperbolic tangent."""
		return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	@staticmethod
	def tanhGradient(x):
		"""Hyperbolic tangent gradient."""
		return 1 - (NeuralNetwork.tanh(x) ** 2)

	@staticmethod
	def linear(x: any) -> any:
		"""Linear."""
		return x

	@staticmethod
	def linearGradient(x):
		"""Gradient of linear."""
		return 1

