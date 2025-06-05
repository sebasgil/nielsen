"""

nielsen_net.py

A module to implement the stochastic gradient descent learning algorithm for a feedforward
neural network in python 3 based on Michael Nielsen's "Neural Networks and Deep Learning" book.
The backpropagation algorithm calculates the gradient of the cost function with respect to the
weights and biases of the network. Readability and ease of understanding of the code are the main
design priority and take precedence over optimization.

"""

import random
import time
import numpy as np
import pdb


class Network(object):

    def __init__(self, sizes):
        """
        The list 'sizes' contains the number of neurons in the respective layers of the network such that a
        [2,3,1] network would have three layers with 2, 3, and 1 neuron respectively. Biases and weights are
        randomly initialized from a zero mean and unity variance Gaussian distribution.  Since the first layer
        is conventionally the input layer, it lacks a bias.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Returns the output of the network from an input a."""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """This method trains the neural network using mini-batch stochastic gradient descent.
        It presumes that the 'training_data' is a list of tuples '(x, y)' where 'x' is an input
        and 'y' is an output, epochs is the number of passes through the training data, mini_batch_size
        is the number of training examples in each mini-batch, and eta is the learning rate . Finally,
        the optional parameter test_data can help track the accuracy of the training process at the
        expense of computational time."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2} took {3:.2f} seconds to complete.".format(j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds.".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent through backpropagation
        on a single mini batch. The inputs are of the same shape as in SGD."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]    
    
    def backprop(self, x, y):
        """Returns a tuple of the form (nabla_b, nabla_w) representing the gradient of the cost function
        C_x. nabla_b and nabla_w are layer-by-layer lists of numpy arrays not unlike self.biases and self.weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # store all activations layer by layer in this list
        zs = [] # store all the z vectors in this list layer by layer
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # pdb.set_trace()
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for wich the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        "Return the vector of partial derivatives for the output activations."
        return (output_activations - y)


# Helper functions

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))