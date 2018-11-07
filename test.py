import random

from neural import *
from formula import *
from genetics import *

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

def gen_nontriv(p, n):
    return [encode_formula(random.choice([
        P >> Q,
        Q >> P,
        P >> R,
        R >> P,
        Q >> R,
        R >> Q,
        ])) for _ in range(n)]

def gen_triv(p, n):
    return [encode_formula(random.choice(
            [P >> P, Q >> Q, R >> R])) for _ in range(n)]

def score(network, nontrivs, trivs):
    c = 0
    for nontriv in nontrivs:
        if abs(network.think(nontriv)[0]) >= 1:
            c += 1
    for triv in trivs:
        if abs(network.think(triv)[0]) < 1:
            c += 1
    return c / (len(nontrivs) + len(trivs))

depthparam = 0.8
train_size = 1000
test_size = 100

root_size = 1 # Just have implication

hiddens = [20]

pool_init_size = 200
pool_max_size = 1000

mutparam = 0.3

training_data = (gen_nontriv(depthparam, train_size // 2),
                    gen_triv(depthparam, train_size // 2))
testing_data = (gen_nontriv(depthparam, train_size // 2),
                    gen_triv(depthparam, train_size // 2))

if __name__ == '__main__':
    pool = (RandomTreeNetwork(root_size, enc_size, hiddens, hiddens, 1)
            for _ in range(pool_init_size))
    print("First generation created")
    fitness = lambda x: score(x, *training_data)
    mutator = lambda x: x.mutate_weights(mutparam)
    crosser = lambda x, y: x.crossover(y)
    run_pool(pool, fitness, pool_max_size, mutator, crosser, 5)
