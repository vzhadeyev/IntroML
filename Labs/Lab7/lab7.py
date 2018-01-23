# Partially based on http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
from sklearn.metrics import accuracy_score

class NN:
	"""A class representing a single layer neural network for classification."""

	def __init__(self,
				 n_input,
				 n_hidden,
				 n_output,
				 learning_rate=0.01,
				 reg_lambda=0.01):
		"""Initializes the neural network.

		Arguments:
			n_input(int): The number of units in the input layer.
			n_hidden(int): The number of units in the hidden layer.
			n_output(int): The number of units in the output layer.
			learning_rate(float): The learning rate to use when performing
								  gradient descent.
			reg_lambda(float): The lambda regularization value.
		"""

		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda

		self.initialize_weights()

	def initialize_weights(self):
		"""Initializes the weights of the network.

		The following variables should be initialized:
			self.W1 - weights connecting input to hidden layer
			self.b1 - bias for hidden layer
			self.W2 - weights connecting hidden to output layer
			self.b2 - bias for output layer
		"""
		self.W1 = np.random.randn(self.n_input, self.n_hidden)
		self.b1 = np.random.randn(self.n_hidden)
		self.W2 = np.random.randn(self.n_hidden, self.n_output)
		self.b2 = np.random.randn(self.n_output)
		return


	def softmax(self, o):
		"""Computes the softmax function for the array o.

		Arguments:
			o(ndarray): A length n_output numpy array representing
						the input to the output layer.

		Returns:
			A length n_output numpy array with the result of applying
			the softmax function to o.
		"""

		#return np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
		return np.exp(o) / np.sum(np.exp(o))

	def feed_forward(self, x):
		"""Runs the network on a data point and outputs probabilities.

		Arguments:
			x(ndarray): An length n_input numpy array where n_input is
						the dimensionality of a data point.

		Returns:
			A length n_output numpy array containing the probabilities
			of each class according to the neural network.
		"""
		return self.softmax(np.dot(self.W2.T, np.tanh(np.dot(self.W1.T, x) + self.b1)) + self.b2)
		#return self.softmax((np.dot(np.tanh(np.dot(np.array(x), self.W1) + self.b1), self.W2) + self.b2))

	def predict(self, x):
		"""Predicts the class of a data point.

		Arguments:
			x(ndarray): An length n_input numpy array where n_input is
						the dimensionality of a data point.

		Returns:
			A length n numpy array containing the predicted class
			for each data point.
		"""
		return np.argmax(self.feed_forward(x))
		#return np.argmax(self.feed_forward(x), axis=1)

	def compute_accuracy(self, X, y):
		"""Computes the accuracy of the network on data.

		Arguments:
			X(ndarray): An n by n_input ndarray where n is the number
						of data points and n_input is the size of each
						data point.
			y(ndarray): A length n numpy array containing the correct
						classes for the data points.

		Returns:
			A float with the accuracy of the neural network.
		"""

		#return np.sum([self.predict(X) == y]) / len(y)
		return np.sum([self.predict(x) == p for x, p in zip(X, y)]) / len(y)
