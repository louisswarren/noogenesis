from heapq import nlargest

def pool_iteration(pool, fitness, max_size, mutator, crosser, trace=0):
    scored = []
    for x in pool:
        score = fitness(x)
        if score > 0.5:
            scored.append((x, score))
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
    while True:
        print("Pool iteration")
        pool = list(pool_iteration(pool, fitness, max_size, mutator, crosser, trace))
