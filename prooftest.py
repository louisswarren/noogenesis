import random

from neural import *
from formula import *
from genetics import *

enc_size = 8 # Need at least 3 for P, Q, R
P, Q, R = (Atom(x) for x in "PQR")
assignments = list(truth_assignments([P, Q, R]))

def encode_formula(x):
    '''Atoms must be encoded to a vector with the same size as the network leaf size.'''
    if isinstance(x, Impl):
        return Stem(np.array([1]), encode_formula(x.prem), encode_formula(x.conc))
    elif x == P:
        return Leaf(np.eye(enc_size, 1, 0))
    elif x == Q:
        return Leaf(np.eye(enc_size, 1, -1))
    elif x == R:
        return Leaf(np.eye(enc_size, 1, -2))
    else:
        raise NotImplementedError

def random_formula(p):
    '''p is a parameter to manipulate depth. I didn't really think about how it
    would work.'''
    if random.random() < p:
        left = random_formula(p * p)
        right = random_formula(p * p)
        return left >> right
    else:
        return random.choice((P, Q, R))

def random_true_formula(p):
    f = random_formula(p)
    while not is_forced_by_all(f, assignments):
        f = random_formula(p)
    return f


# Idea: each candidate network takes a pair (goal, contextelement), gives a
# weighting. Highest weighted context element is used. For assumption, use the
# zero vector.

def strategy_works(network, proposition):
    

def score(network, propositions):
    c = 0
    for nontriv in nontrivs:
        if abs(network.think(nontriv)[0]) >= 1:
            c += 1
    for triv in trivs:
        if abs(network.think(triv)[0]) < 1:
            c += 1
    return c / (len(nontrivs) + len(trivs))

depthparam = 0.6
train_size = 200
test_size = 100

root_size = 1 # Just have implication

hiddens = [20]

pool_init_size = 200
pool_max_size = 100

mutparam = 1.0

training_data = [random_true_formula(depthparam) for _ in range(train_size)]

if __name__ == '__main__':
    pool = (RandomTreeNetwork(root_size, enc_size, hiddens, hiddens, 1)
            for _ in range(pool_init_size))
    print("First generation created")
    fitness = lambda x: score(x, *training_data)
    mutator = lambda x: x.mutate(mutparam)
    crosser = lambda x, y: x.crossover(y)
    run_pool(pool, fitness, pool_max_size, mutator, crosser, 5)
