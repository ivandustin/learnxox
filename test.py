#!/usr/bin/env python
from orbax.checkpoint import StandardCheckpointer
from flax.nnx import softmax
from learn.load import load
from encode import encode
from xox.init import init
from xox import O


def ask(model, x):
    y = softmax(model(encode(x)))
    return y, y.argmax()


with StandardCheckpointer() as checkpointer:
    model = load(checkpointer)
    state = init()
    state[range(8)] = O
    y, i = ask(model, state)
    assert i == 8

    from xox.exceptions import Over, Win, Lose, Draw
    from xox.string import string
    from xox.loop import loop

    def show(state):
        print(string(state) + "\n")

    def o(state):
        show(state)
        return int(input("Your move: ")) - 1

    def x(state):
        show(state)
        y, i = ask(model, (state))
        print(y)
        print(i + 1, y[i])
        return i

    try:
        loop(x, o)
    except Over as e:
        (state,) = e.args
        show(state)
        if isinstance(e, Win):
            print("You win!")
        elif isinstance(e, Lose):
            print("You lose!")
        elif isinstance(e, Draw):
            print("Draw!")
