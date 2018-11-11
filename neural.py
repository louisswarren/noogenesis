import numpy as np
import random

def _rescale(r, epsilon):
    return epsilon * (2 * r - 1)

class Layer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def crossover_mean(self, other):
        # Dumb strategy: take means
        weights = (self.weights + other.weights) / 2
        biases = (self.biases + other.biases) / 2
        return Layer(weights, biases)

    def crossover_neurons(self, other):
        # Higher level: mix neurons
        sel = [random.choice((True, False)) for _ in range(len(self.weights))]
        weights = np.array([u if s else v
                            for s, u, v in zip(sel, self.weights, other.weights)])
        biases = np.array([x if s else y
                           for s, x, y in zip(sel, self.biases, other.biases)])
        return Layer(weights, biases)

    def crossover(self, other):
        return self.crossover_neurons(other)

    def mutate(self, epsilon):
        weights = self.weights + epsilon * (2 * np.random.rand(*self.weights.shape) - 1)
        biases = self.biases + epsilon * (2 * np.random.rand(*self.biases.shape) - 1)
        return Layer(weights, biases)

    def mutate_one(self, epsilon):
        wm = np.zeros(self.weights.shape)
        h = random.randrange(self.weights.shape[0])
        w = random.randrange(self.weights.shape[1] + 1)
        if w == self.weights.shape[1]:
            bm = np.zeros(self.biases.shape)
            bm[h][0] = _rescale(random.random(), epsilon)
            return Layer(self.weights, self.biases + bm)
        else:
            wm = np.zeros(self.weights.shape)
            wm[h][w] = _rescale(random.random(), epsilon)
            return Layer(self.weights + wm, self.biases)


    def think(self, x):
        return self.weights @ x + self.biases

def RandomLayer(insize, outsize):
    weights = 2 * np.random.rand(outsize, insize) - 1
    biases = 2 * np.random.rand(outsize, 1) - 1
    return Layer(weights, biases)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def crossover(self, other):
        layers = []
        for sl, ol in zip(self.layers, other.layers):
            layers.append(sl.crossover(ol))
        return Network(layers)

    def mutate_weights(self, epsilon):
        layers = [layer.mutate_one(epsilon) for layer in self.layers]
        return Network(layers)

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

    def crossover(self, other):
        return TreeNetwork(self.treenet.crossover(other.treenet),
                           self.endnet.crossover(other.endnet))

    def mutate_weights(self, epsilon):
        treenet = self.treenet.mutate_weights(epsilon)
        endnet = self.endnet.mutate_weights(epsilon)
        return TreeNetwork(treenet, endnet)

    def _treeinput(self, t):
        if isinstance(t, Leaf):
            return t.value
        elif isinstance(t, Stem):
            left = self._treeinput(t.left)
            right = self._treeinput(t.right)
            return self.treenet.think([[x] for x in [*t.value, *left, *right]])
        else:
            raise NotImplementedError

    def think(self, t):
        return self.endnet.think(self._treeinput(t))

def RandomTreeNetwork(root_size, leaf_size, tree_hidden_sizes, end_hidden_sizes, end_outsize):
    treenet = RandomNetwork(root_size + 2 * leaf_size, *tree_hidden_sizes, leaf_size)
    endnet = RandomNetwork(leaf_size, *end_hidden_sizes, end_outsize)
    return TreeNetwork(treenet, endnet)
