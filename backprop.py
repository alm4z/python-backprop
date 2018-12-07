import numpy as np

np.random.seed(0)


#
# Back propagation algorithm realization
#
#
# From
# https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b
#
# The back propagation algorithm begins by comparing the actual value output
# by the forward propagation process to the expected value and then moves
# backward through the network, slightly adjusting each of the weights in
# a direction that reduces the size of the error by a small degree.
#
# Both forward and back propagation are re-run thousands of times on each
# input combination until the network can accurately predict the expected
# output of the possible inputs using forward propagation.
#
# For the XOR problem, 100% of possible data examples are available to use
# in the training process. We can therefore expect the trained network to be
# 100% accurate in its predictions and there is no need to be concerned with
# issues such as bias and variance in the resulting model.
#

class OneHiddenLayerNetwork:

    #
    # activation functions
    #

    @staticmethod
    def tang(y):
        return np.tanh(y)

    @staticmethod
    def derivative_tang(y):
        return 1.0 - y ** 2

    @staticmethod
    def sigmoid(y):
        return 1 / (1 + np.exp(-y))

    @staticmethod
    def derivative_sigmoid(y):
        return y * (1 - y)

    def __init__(self, learning_rate=0.1):

        #
        # neural network architecture
        # simple 2 x 2 x 1 that is enough for XOR example
        # input x hidden x output
        self.learning_rate = learning_rate
        self.output = None

        # weights with random values
        self.weights = [
            np.random.uniform(low=-0.2, high=0.2, size=(2, 2)),  # input layer
            np.random.uniform(low=-2, high=2, size=(2, 1))  # hidden layer
        ]

    def activation(self, activation_type, y):
        if activation_type == 'sigmoid':
            return self.sigmoid(y)
        if activation_type == 'tang':
            return self.tang(y)

        raise ValueError('Undefined activation function: {}'.format(activation_type))

    def derivative_activation(self, activation_type, y):
        if activation_type == 'sigmoid':
            return self.derivative_sigmoid(y)
        if activation_type == 'tang':
            return self.derivative_tang(y)

        raise ValueError('Undefined derivative activation function: {}'.format(activation_type))

    #
    # forward pass
    # layer by layer
    #
    def feed_forward_pass(self, x_values):

        # forward
        input_layer = x_values
        hidden_layer = self.activation('tang', np.dot(input_layer, self.weights[0]))
        output_layer = self.activation('tang', np.dot(hidden_layer, self.weights[1]))

        self.layers = [
            input_layer,
            hidden_layer,
            output_layer
        ]

        # last layer is an output
        return self.layers[2]

    #
    # back propagation error through the network layers
    #
    def backward_pass(self, target_output, actual_output):

        # divergence of network output
        err = (target_output - actual_output)

        # backward from output to input layer
        # propagate gradients using chain rule
        for backward in range(2, 0, -1):
            err_delta = err * self.derivative_activation('tang', self.layers[backward])

            # update weights using computed gradient
            self.weights[backward - 1] += self.learning_rate * np.dot(self.layers[backward - 1].T, err_delta)

            # propagate error using updated weights of previous layer
            err = np.dot(err_delta, self.weights[backward - 1].T)

    def train(self, x_values, target):
        self.output = self.feed_forward_pass(x_values)
        self.backward_pass(target, self.output)

    def predict(self, x_values):
        return self.feed_forward_pass(x_values)


#
# X - XOR dataset data
# y = target
#
X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)

network = OneHiddenLayerNetwork(learning_rate=0.1)
iterations = 5000

# training
for i in range(iterations):
    network.train(X, y)

    ten = iterations // 10
    if i % ten == 0:
        print('-' * 10)
        print("Iteration number: " + str(i) + ' / ' +
              "Squared loss: " + str(np.mean(np.square(y - network.output))))

# predict
for i in range(len(X)):
    print('-' * 10)
    print('Input value: ' + str(X[i]))
    print('Predicted target: ' + str(network.predict(X[i])))
    print('Actual target: ' + str(y[i]))
