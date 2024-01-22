import numpy as np
import matplotlib.pyplot as plt

# create static class filled with actiation functions
class ActivationFunctions:
    # sigmoid
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    # tanh
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    # relu
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # softmax
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    # linear
    @staticmethod
    def linear(x):
        return x

class ENN:
    def __init__(self, dims, weights_in) -> None:
        # example dims = [2, [3, 4, 1]]
        # scalar
        self.i_units = dims[0]
        # 1x2 vector
        self.h_units = dims[1]
        self.layers = []
        self.biases = []
        self.genome_size = ENN.get_genome_size(dims)
        self.weights_flat = weights_in
        self.set_weights(weights_in)

    # static function to get genome size
    @staticmethod
    def get_genome_size(dims):
        genome_size = 0
        prev_units = dims[0]
        for units in dims[1]:
            # add weights
            genome_size += prev_units*units
            # add biases
            genome_size += units
            # update prev_units
            prev_units = units
        return genome_size

    def set_weights(self, weights_in):
        self.layers = []
        self.biases = []
        flat_weights = weights_in
        prev_units = self.i_units

        for units in self.h_units:
            # create weights
            weights_len = prev_units*units
            weights = np.array(
                flat_weights[:weights_len]
            ).reshape((prev_units, units))

            # create biases
            biases_len = units
            biases = np.array(
                flat_weights[weights_len:weights_len+biases_len]
            ).reshape((units))

            # add to layers
            self.layers.append(weights)
            self.biases.append(biases)

            flat_weights = flat_weights[weights_len+biases_len:]
            
            # update prev_units
            prev_units = units
        
        return self.layers, self.biases

    def forward(self, input):
        input = np.array(input)
        # check if input is the right size
        if len(input) != self.i_units:
            raise Exception("input is not the right size, expected " + str(self.i_units) + " but got " + str(len(input)))
        
        # set input as prev_layer
        prev_layer = input

        # loop through layers
        for layer, bias in zip(self.layers, self.biases):
            prev_layer = np.dot(layer.T, prev_layer) + bias
            prev_layer = np.array(list(map(self.activation, prev_layer)))
            
        return prev_layer

    def get_weights(self):
        return self.weights_flat

    def activation(self, x):
        return ActivationFunctions.tanh(x)


# write a test case to check the output
if __name__ == "__main__":
    # write a simple feed forward test
    # create a network with 2 inputs, 1 hidden layer with 3 units, and 1 output
    # create a random genome
    genome = np.random.rand(ENN.get_genome_size([2, [3, 1]]))
    # create a network
    network = ENN([2, [3, 1]], genome)
    # create an input
    input = [32, 5]
    # run the network
    output = network.forward(input)
    # print the output
    print("output: ", output)
    # print the weights
    print(network.layers)
    # print the biases
    print(network.biases)