#!/usr/bin/env python
from collections import deque
from pathlib import Path
from numpy import any
from xox.exceptions import Over, Win, Lose
from xox.loop import loop
from xox.rand import rand
from xox.put import put
from xox import O
from encode import encode

stack = []
maxlen = 10000
win = deque(maxlen=maxlen)
block = deque(maxlen=maxlen)


def x(state):
    index = rand(state)
    stack.append((state.copy(), index))
    if is_block(state, index):
        block.append((state.copy(), index))
    return index


def o(state):
    return rand(state)


def string(encoded):
    assert any(encoded == 4)
    return "\n".join([" ".join(map(str, row)) for row in encoded.T + 1])


def out(file, state, index):
    print(string(encode(state)), file=file)
    print(index + 1, file=file)
    print(file=file)


def is_full(queue):
    return queue.maxlen == len(queue)


def is_block(state, index):
    try:
        state = state.copy()
        put(state, index, O)
    except Over as e:
        if isinstance(e, Lose):
            return True
    return False


def write(path, queue):
    with open(path, "w") as file:
        for state, index in queue:
            out(file, state, index)


while not is_full(win) or not is_full(block):
    try:
        stack = []
        loop(x, o)
    except Over as e:
        if isinstance(e, Win):
            win.append(stack.pop())

winfile = Path("win") / "train.txt"
blockfile = Path("block") / "train.txt"

write(winfile, win)
write(blockfile, block)
