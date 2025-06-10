"""

nielsen_net.py

The goal of this exercise is to modify the code of network.py so that it uses a fully
matrix-based approach to work across all batches in conjunction.

"""

import random
import time
import numpy as np
import pdb

class Network(object):

    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, A):
        pass

    def backprop(self, x, y):
        pass

    def gradient_descent(self, eta):
        pass

    def SGD(self, training_data, mini_batches, eta, test_data=None):
        pass

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations assuming a quadratic cost function"""
        return (output_activations - y)


# Helper functions

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))