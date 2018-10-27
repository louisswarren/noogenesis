import numpy as np

class Layer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def copy(self):
        return Layer(np.copy(self.weights), np.copy(self.biases))

    def crossover(self, other):
        weights = (self.weights + other.weights) / 2
        biases = (self.biases + other.biases) / 2
        return Layer(weights, biases)

    def mutate(self, epsilon):
        self.weights += epsilon * (2 * np.random.rand(*self.weights.shape) + 1)
        self.biases += epsilon * (2 * np.random.rand(*self.biases.shape) + 1)

    def think(self, x):
        return x @ self.weights + self.biases

def RandomLayer(insize, outsize):
    weights = 2 * np.random.rand(insize, outsize) - 1
    biases = 2 * np.random.rand(outsize) - 1
    return Layer(weights, biases)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def copy(self):
        return Network([x.copy() for x in self.layers])

    def crossover(self, other):
        # Dumb strategy: take means
        layers = []
        for sl, ol in zip(self.layers, other.layers):
            layers.append(sl.crossover(ol))
        return Network(layers)

    def mutate_weights(self, epsilon):
        for layer in self.layers:
            layer.mutate(epsilon)

    def mutate_dimension(self, prob):
        pass

    def think(self, x):
        for layer in self.layers:
            x = layer.think(x)
        return x

def RandomNetwork(*sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(RandomLayer(sizes[i], sizes[i + 1]))
    return Network(layers)


class Stem:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, value):
        self.value = value


class TreeNetwork:
    '''Recursively inputs a binary tree into a neural network.'''
    def __init__(self, treenet, endnet):
        self.treenet = treenet
        self.endnet = endnet

    def copy(self):
        return TreeNetwork(self.treenet.copy(), self.endnet.copy())

    def crossover(self, other):
        return TreeNetwork(self.treenet.crossover(other.treenet),
                           self.endnet.crossover(other.endnet))

    def mutate_weights(self, epsilon):
        self.treenet.mutate_weights(epsilon)
        self.endnet.mutate_weights(epsilon)

    def _treeinput(self, t):
        if isinstance(t, Leaf):
            return t.value
        elif isinstance(t, Stem):
            left = self._treeinput(t.left)
            right = self._treeinput(t.right)
            return self.treenet.think([*t.value, *left, *right])
        else:
            raise NotImplementedError

    def think(self, t):
        return self.endnet.think(self._treeinput(t))

def RandomTreeNetwork(root_size, leaf_size, tree_hidden_sizes, end_hidden_sizes, end_outsize):
    treenet = RandomNetwork(root_size + 2 * leaf_size, *tree_hidden_sizes, leaf_size)
    endnet = RandomNetwork(leaf_size, *end_hidden_sizes, end_outsize)
    return TreeNetwork(treenet, endnet)
