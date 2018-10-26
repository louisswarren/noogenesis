import random

from neural import *
from formula import *

P, Q, R = (Atom(x) for x in "PQR")

def encode_formula(x):
    '''Atoms must be encoded to a vector with the same size as the network leaf size.'''
    if isinstance(x, Impl):
        return Stem(np.array([1]), encode_formula(x.prem), encode_formula(x.conc))
    elif x == P:
        return Leaf(np.array([1, 0, 0]))
    elif x == Q:
        return Leaf(np.array([0, 1, 0]))
    elif x == R:
        return Leaf(np.array([0, 0, 1]))
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
        newpool.append(x.crossover(y))

nontriv_testing = [encode_formula(random_nontriv_impl(p)) for _ in range(test_size // 2)]
triv_testing = [encode_formula(triv(random_formula(p*p))) for _ in range(test_size // 2)]
newtop20 = sorted(newpool, key=lambda n: score(n, nontriv_training, triv_training), reverse=True)[:20]

for net in newtop20:
    print(score(net, nontriv_testing, triv_testing) / test_size)

