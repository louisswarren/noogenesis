import itertools

class Formula:
    def __rshift__(self, other):
        return Impl(self, other)

class Atom(Formula):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Atom('{self.name}')"

    def __str__(self):
        return self.name

    def valuation(self, c):
        return c[self]

class Impl(Formula):
    def __init__(self, prem, conc):
        self.prem = prem
        self.conc = conc

    def __repr__(self):
        return f"({self.prem} >> {self.conc})"

    def valuation(self, c):
        return self.conc.valuation(c) or not self.prem.valuation(c)

def truth_assignments(atoms):
    for tvs in itertools.product([False, True], repeat=len(atoms)):
        yield {a: b for a, b in zip(atoms, tvs)}

def is_forced_by_all(f, assignments):
    return all(f.valuation(assignment) for assignment in assignments)
