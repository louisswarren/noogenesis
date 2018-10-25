import time
import numpy as np
import random

class Layer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def breed(self, other):
        weights = (self.weights + other.weights) / 2
        biases = (self.biases + other.biases) / 2
        return Layer(weights, biases)


    def think(self, x):
        return x @ self.weights + self.biases

class RandomLayer(Layer):
    def __init__(self, insize, outsize):
        self.weights = 2 * np.random.rand(insize, outsize) - 1
        self.biases = 2 * np.random.rand(outsize) - 1


class Network:
    def __init__(self, layers):
        self.layers = layers

    def mutate_weights(epsilon):
        pass

    def mutate_dimension(prob):
        pass

    def breed(self, other):
        # Dumb strategy: take means
        layers = []
        for sl, ol in zip(self.layers, other.layers):
            layers.append(sl.breed(ol))
        return Network(layers)

    def think(self, x):
        for layer in self.layers:
            x = layer.think(x)
        return x

class RandomNetwork(Network):
    def __init__(self, *sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(RandomLayer(sizes[i], sizes[i + 1]))

# We want to feed formule into a recursive network

from prover import *

class Branch:
    def __init__(self, vec, left, right):
        self.rootvec = vec
        self.left = left
        self.right = right

class Node:
    def __init__(self, vec):
        self.vec = vec

class TreeNetwork:
    def __init__(self, treenet, endnet):
        self.treenet = treenet
        self.endnet = endnet

    def breed(self, other):
        return TreeNetwork(self.treenet.breed(other.treenet), self.endnet.breed(other.endnet))

    def rthink(self, t):
        if isinstance(t, Node):
            return t.vec
        elif isinstance(t, Branch):
            left = self.rthink(t.left)
            right = self.rthink(t.right)
            return self.treenet.think([*t.rootvec, *left, *right])
        else:
            print(t)
            raise NotImplementedError

    def think(self, t):
        x = self.rthink(t)
        return self.endnet.think(x)

class RandomTreeNetwork(TreeNetwork):
    def __init__(self, root_size, leaf_size, tree_hidden_sizes, end_hidden_sizes, end_outsize):
        self.treenet = RandomNetwork(root_size + 2 * leaf_size, *tree_hidden_sizes, leaf_size)
        self.endnet = RandomNetwork(leaf_size, *end_hidden_sizes, end_outsize)


P, Q, R = (Atom(x) for x in "PQR")

def encode_formula(x):
    '''Atoms must be encoded to a vector with the same size as the network leaf size.'''
    if isinstance(x, Impl):
        return Branch(np.array([1]), encode_formula(x.prem), encode_formula(x.conc))
    elif x == P:
        return Node(np.array([1, 0, 0]))
    elif x == Q:
        return Node(np.array([0, 1, 0]))
    elif x == R:
        return Node(np.array([0, 0, 1]))
    else:
        raise NotImplementedError

tree_hidden_sizes = 8,
fnet = RandomTreeNetwork(1, 3, tree_hidden_sizes, tree_hidden_sizes, 1)
print(fnet.think(encode_formula(P)))
print(fnet.think(encode_formula((P >> (Q >> R)) >> (Q >> (P >> R)))))

# Let's try, using no mutation or breeding, to recognise trivialities

def random_formula(p):
    '''p is a parameter to manipulate depth. I didn't really think about how it
    would work.'''
    if random.random() < p:
        left = random_formula(p * p)
        right = random_formula(p * p)
        return left >> right
    else:
        return random.choice((P, Q, R))

def random_nontriv_impl(p):
    left = random_formula(p)
    right = random_formula(p)
    while left == right:
        right = random_formula(p)
    return left >> right

def triv(x):
    return x >> x

p = 0.9
train_size = 1000
nontriv_training = [encode_formula(random_nontriv_impl(p)) for _ in range(train_size // 2)]
triv_training = [encode_formula(triv(random_formula(p*p))) for _ in range(train_size // 2)]
test_size = 100
nontriv_testing = [encode_formula(random_nontriv_impl(p)) for _ in range(test_size // 2)]
triv_testing = [encode_formula(triv(random_formula(p*p))) for _ in range(test_size // 2)]

def score(network, nontriv_data, triv_data):
    c = 0
    for nontriv in nontriv_data:
        if network.think(nontriv)[0] >= 1:
            c += 1
    for triv in triv_data:
        if network.think(triv)[0] < 1:
            c += 1
    return c

# Create a lot of networks and find the best
pool = [RandomTreeNetwork(1, 3, [20], [20], 1) for _ in range(100)]
top20 = sorted(pool, key=lambda n: score(n, nontriv_training, triv_training), reverse=True)[:20]

for net in top20:
    print(score(net, nontriv_testing, triv_testing) / test_size)

print()
newpool = []
for (i, x) in enumerate(top20):
    for y in top20[i+1:]:
        newpool.append(x.breed(y))

nontriv_testing = [encode_formula(random_nontriv_impl(p)) for _ in range(test_size // 2)]
triv_testing = [encode_formula(triv(random_formula(p*p))) for _ in range(test_size // 2)]
newtop20 = sorted(newpool, key=lambda n: score(n, nontriv_training, triv_training), reverse=True)[:20]

for net in newtop20:
    print(score(net, nontriv_testing, triv_testing) / test_size)

