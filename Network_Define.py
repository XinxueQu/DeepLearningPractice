import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        params = zip(self.biases, self.weights)
        for b, w in params[:-1]:
            w = w.astype(float)
            a = a.astype(float)
            a = ReLU(np.dot(w, a) + b)
        b, w = params[-1]
        w = w.astype(float)
        a = a.astype(float)
        a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        The training_data is a list of tuples (x, y) representing the training inputs and the desired outputs."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate."""
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)
            gradient_b = [nb + dgb for nb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw + dgw for nw, dgw in zip(gradient_w, delta_gradient_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, gradient_b)]

    def backprop(self, x, y):
        """Return gradient for weight and bias.
        ``gradient_b`` and ``gradient_w`` are layer-by-layer lists of numpy arrays."""
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer\
        params=zip(self.biases, self.weights)
        layer=1
        for b, w in params:
            w = w.astype(float)
            activation = activation.astype(float)
            z = np.dot(w, activation) + b
            zs.append(z)
            if layer==len(params):
                activation = sigmoid(z)
            else:
                activation = ReLU(z)
            layer += 1
            activations.append(activation)
        # backward pass
        delta = np.nan_to_num(activations[-1]-y)
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = ReLU_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradient_b[-l] = delta
            delta = delta.astype(float)
            activations[-l - 1] = activations[-l - 1].astype(float)
            gradient_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (gradient_b, gradient_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(1.0 * (self.feedforward(x) > 0.5), y)
                        for (x, y) in test_data]
        return sum(int(int(x) == int(y)) for (x, y) in test_results)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def ReLU (x,epsilon=0.001):
    result = np.maximum(x*epsilon,np.zeros((len(x),1)))
    return result

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU_prime(z,epsilon=0.001):
    """Derivative of the ReLU function."""
    derivative=1.0 * (z > 0)
    derivative[derivative == 0] = epsilon
    #derivative=np.exp(z)/(1+np.exp(z))
    return derivative