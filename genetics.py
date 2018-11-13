from heapq import nlargest
import datetime
import pickle

THRESHOLD = 0.95

def save(specimen):
    with open('saved_' + datetime.datetime.now().isoformat(), 'wb') as f:
        pickle.dump(specimen, f)


def pool_iteration(pool, fitness, max_size, mutator, crosser, trace=0):
    scored = []
    pool = list(pool)
    print("Pool iteration. Pool size:", len(pool))
    for x in pool:
        score = fitness(x)
        if score > THRESHOLD:
            save(x)
            return
        if score > 0.5:
            scored.append((x, score))
    print("Only", len(scored), "survived.")
    fittest = nlargest(max_size, scored, key=lambda x: x[1])
    if trace:
        for x, score in fittest[:trace]:
            print(score, end='\t')
        print()
    print("Mutating ... ", end='')
    if mutator:
        for x, _ in fittest:
            yield mutator(x)
    print("Done")
    print("Crossing ... ", end='')
    if crosser:
        for i, x_score in enumerate(fittest):
            x, _ = x_score
            for y, _ in fittest[i + 1:]:
                yield crosser(x, y)
    print("Done")
    for x, _ in fittest:
        yield x

def run_pool(pool, fitness, max_size, mutator, crosser, trace=0):
    pool = list(pool)
    while pool:
        pool = list(pool_iteration(pool, fitness, max_size, mutator, crosser, trace))
