import random

from neural import *
from genetics import *
import numpy as np
import random

dim = 3

def random_congruent():
    # Don't even multiply by a scalar for now
    v = [random.random() for _ in range(dim)]
    return np.array([[x] for x in v + v])

def random_incongruent():
    # Yes, these could theoretically be congruent
    v = [random.random() for _ in range(2 * dim)]
    return np.array([[x] for x in v])

def gen_congruent(n):
    return [random_congruent() for _ in range(n)]

def gen_incongruent(n):
    return [random_congruent() for _ in range(n)]

def score(network, incongs, congs):
    c = 0
    for incong in incongs:
        if abs(network.think(incong)[0]) < 1:
            c += 1
    for cong in congs:
        if abs(network.think(cong)[0]) >= 1:
            c += 1
    return c / (len(incongs) + len(congs))

train_size = 1000
test_size = 100

hiddens = [20]

pool_init_size = 200
pool_max_size = 50

mutparam = 0.3

training_data = (gen_incongruent(train_size // 2),
                    gen_congruent(train_size // 2))
testing_data = (gen_incongruent(test_size // 2),
                    gen_congruent(test_size // 2))

if __name__ == '__main__':
    pool = (RandomNetwork(dim * 2, *hiddens, 1) for _ in range(pool_init_size))
    print("First generation created")
    fitness = lambda x: score(x, *training_data)
    mutator = lambda x: x.mutate(mutparam)
    crosser = lambda x, y: x.crossover(y)
    run_pool(pool, fitness, pool_max_size, mutator, crosser, 5)
