import random

from neural import *
from formula import *

enc_size = 8 # Need at least 3 for P, Q, R
P, Q, R = (Atom(x) for x in "PQR")

def encode_formula(x):
    '''Atoms must be encoded to a vector with the same size as the network leaf size.'''
    if isinstance(x, Impl):
        return Stem(np.array([1]), encode_formula(x.prem), encode_formula(x.conc))
    elif x == P:
        return Leaf(np.eye(enc_size, 1, 0))
    elif x == Q:
        return Leaf(np.eye(enc_size, 1, 1))
    elif x == R:
        return Leaf(np.eye(enc_size, 1, 2))
    else:
        raise NotImplementedError


# Let's try, using no mutation or crossover, to recognise trivialities

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

def gen_nontriv(p, n):
    return [encode_formula(random_nontriv_impl(p)) for _ in range(n)]

def gen_triv(p, n):
    return [encode_formula(triv(random_formula(p*p))) for _ in range(n)]

def score(network, nontrivs, trivs):
    c = 0
    for nontriv in nontrivs:
        if network.think(nontriv)[0] >= 1:
            c += 1
    for triv in trivs:
        if network.think(triv)[0] < 1:
            c += 1
    return c / (len(nontrivs) + len(trivs))

depthparam = 0.9
train_size = 1000
test_size = 100

root_size = 1 # Just have implication

hiddens = [20]

pool_init_size = 100
pool_max_parents_size = 50

mutparam = 0.03

training_data = (gen_nontriv(depthparam, train_size // 2),
                    gen_triv(depthparam, train_size // 2))
testing_data = (gen_nontriv(depthparam, train_size // 2),
                    gen_triv(depthparam, train_size // 2))

from heapq import nlargest

def run():
    pool = [RandomTreeNetwork(root_size, enc_size, hiddens, hiddens, 1)
            for _ in range(pool_init_size)]
    print("First generation created")
    while True:
        scored = ((net, score(net, *training_data)) for net in pool)
        print("Scored")
        fittest = nlargest(pool_max_parents_size, scored, key=lambda x: x[1])
        print("Best scores:", fittest[0][1], fittest[1][1], fittest[2][1])
        print("VS testing:", score(fittest[0][0], *testing_data),
                             score(fittest[1][0], *testing_data),
                             score(fittest[2][0], *testing_data))
        pool = [x[0] for x in fittest]
        pool += [x[0].crossover(y[0])
                 for (i, x) in enumerate(fittest) for y in fittest[i+1:]]
        for net in pool:
            net.mutate_weights(mutparam)
        print("Breeding complete")

if __name__ == '__main__':
    run()
