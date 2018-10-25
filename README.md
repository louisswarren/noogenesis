# noogenesis
A proof-assistant-using neural network formed using a genetic algorithm

## Goals
The primary goal is to train a network to use
[MINLOG](www.mathematik.uni-muenchen.de/~logik/minlog/). More achievable
subgoals are:

* Recognise trivial implications (those of the form `P => P`).
* Use a python-implemented implicational logic prover, supporting only the
  `use` and `assume` commands of MINLOG.
* As above, but interacting directly with MINLOG.

## Formulae into neural networks
We need to be able to enter formulae into a neural network. Since formulae have
a tree-like structure, and have arbitrary size, we do this using a second
neural network for encoding formulae as vectors.

Fix some number `n`, at least as large as the number of atomic formulae we want
to use. The `k`th atomic formula is then the vector `e_k` in `R^n`. We also fix
a number `m`, equal to the number of logical operations, and similarly the
`k`th logical operation is `e_k` in `R^m`. Now form a neural network from
`R^(m+2n)` to `R^n`, and recursively feed formula trees into this network to
obtain a corresponding vector in `R^n`.

If atoms are encoded as above by a function `A`, and operations by a function
`L`, and the neural network is `N`, then the recursive definition for the
function `E` which encodes arbitrary formulae is

    E(Atom p) = A(p)
    E(x op y) = N(L(op) | E(x) | E(y)).

Now any neural network operating on formulae can operate on `R^n` using this
encoding. An appropriate encoding network will be learned at the same time as
the main network.

## Learning
Gradient descent does not seem appropriate for the goals above. Instead, a
genetic algorithm will be used to learn a proof strategy.
