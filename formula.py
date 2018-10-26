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

