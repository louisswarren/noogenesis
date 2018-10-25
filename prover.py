compose = lambda f: lambda g: lambda *a, **k: f(g(*a, **k))

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

class Impl(Formula):
    def __init__(self, prem, conc):
        self.prem = prem
        self.conc = conc

    def __repr__(self):
        return f"({self.prem} >> {self.conc})"

class Proof:
    def __init__(self, context, goal):
        self.context = context
        self.goal = goal
        self.proved = False
        self.subproofs = []

    def do_assume(self):
        assert("Assumption is possible" and isinstance(self.goal, Impl))
        self.context.append(self.goal.prem)
        self.goal = self.goal.conc

    def do_use(self, n):
        assert("Context formula in range" and 0 <= n < len(self.context))
        f = self.context[n]
        premises = []
        while f != self.goal:
            assert("Formula is usable for goal" and isinstance(f, Impl))
            premises.append(f.prem)
            f = f.conc
        for premise in premises:
            self.subproofs.append(Proof(self.context, premise))
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

    @compose('\n'.join)
    def __str__(self):
        for i, f in enumerate(self.context):
            yield f"{i:>2}: {f}"
        yield "-" * 40
        yield f"  {self.goal}"
        yield ""

    @compose('\n'.join)
    def status(self):
        problems = [str(p) for p in self.open_problems()]
        if problems:
            yield from reversed(problems)
        else:
            yield "Done."

if __name__ == '__main__':
    P, Q, R = (Atom(x) for x in "PQR")
    p = Proof([], (P >> (Q >> R)) >> (Q >> (P >> R)))
    p.assume()
    p.assume()
    p.assume()
    p.use(0)
    p.use(2)
    p.use(1)

    print(p.status())
