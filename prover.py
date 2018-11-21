from formula import *

_compose = lambda f: lambda g: lambda *a, **k: f(g(*a, **k))

class DeductionError(Exception):
    pass

class Prover:
    def __init__(self, context, goal):
        self.context = context
        self.goal = goal
        self.proved = False
        self.subproofs = []

    def do_assume(self):
        if not isinstance(self.goal, Impl):
            raise DeductionError("Assumption is not possible")
        self.context.append(self.goal.prem)
        self.goal = self.goal.conc

    def do_use(self, n):
        if not 0 <= n < len(self.context):
            raise DeductionError("Context formula out of range")
        f = self.context[n]
        premises = []
        while f != self.goal:
            if not isinstance(f, Impl):
                raise DeductionError("Formula is not usable for goal")
            premises.append(f.prem)
            f = f.conc
        for premise in premises:
            self.subproofs.append(Prover(self.context, premise))
        self.proved = True

    def open_problems(self):
        if not self.proved:
            yield self
        for subproof in self.subproofs:
            yield from subproof.open_problems()

    def next_problem(self):
        return next(self.open_problems())

    def assume(self):
        self.next_problem().do_assume()

    def use(self, n):
        self.next_problem().do_use(n)

    @_compose('\n'.join)
    def __str__(self):
        for i, f in enumerate(self.context):
            yield f"{i:>2}: {f}"
        yield "-" * 40
        yield f"  {self.goal}"
        yield ""

    def is_complete(self):
        return not bool(list(self.open_problems()))

    @_compose('\n'.join)
    def status(self):
        problems = [str(p) for p in self.open_problems()]
        if problems:
            yield from reversed(problems)
        else:
            yield "Done."

if __name__ == '__main__':
    P, Q, R = (Atom(x) for x in "PQR")
    p = Prover([], (P >> (Q >> R)) >> (Q >> (P >> R)))
    p.assume()
    p.assume()
    p.assume()
    p.use(0)
    p.use(2)
    p.use(1)

    print(p.status())
