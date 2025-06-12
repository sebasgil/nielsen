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
        for b,W in zip(self.biases, self.weights):
            # This retains the same shape, but now b is broadcasted everywhere
            A = sigmoid(np.dot(W, A) + b)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # Get the individual x's and y's from mini batches:
                xs = [x for x, y in mini_batch]
                ys = [y for x, y in mini_batch]
                # Now concatenate them into matrices for batch processing
                X = np.concatenate(xs, axis=1)
                Y = np.concatenate(ys, axis=1)
                # Now pass on to update
                self.update_mini_batch(X, Y, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2} took {3:.2f} seconds to complete.".format(j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds.".format(j, time2-time1))

    def update_mini_batch(self, X, Y, eta):        
        # Backpropagate the entire batch at once, making sure the returned deltas are averaged
        nabla_b, nabla_w = self.backprop(X, Y)
        # Update the weights and biases without dividing by the batch size because we already averaged them in backprop
        self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]         

    def backprop(self, X, Y):
        # Initialize the activation for the first layer of the mini-batch
        activation = X
        activations = [X] # store the first layer activations
        zs = [] # will store all the z vectors in layer by layer
        # Forward pass
        for b,W in zip(self.biases, self.weights):
            Z = np.dot(W,activation) + b
            zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        # Get the batch size
        batch_size = X.shape[1]
        # Backward pass
        delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(zs[-1])
        # Initialize nabla_b and nabla_w for the last layer
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Calculate the nabla_b and nabla_w for the last layer
        nabla_b[-1] = (1/batch_size) * np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = (1/batch_size) * np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            Z = zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = (1/batch_size) * np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = (1/batch_size) * np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
 
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, Y):
        """Return the vector of partial derivatives for the output activations assuming a quadratic cost function"""
        return (output_activations - Y)


# Helper functions

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))