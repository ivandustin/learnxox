from collections import deque
from numpy import any
from xox.exceptions import Over, Win, Lose
from xox.loop import loop
from xox.rand import rand
from xox.put import put
from xox import O
from encode import encode

stack = []
queue = deque(maxlen=100)
block = deque(maxlen=900)


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


def out(state, index):
    print(string(encode(state)))
    print(index + 1)
    print()


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


while not is_full(queue) or not is_full(block):
    try:
        stack = []
        loop(x, o)
    except Over as e:
        if isinstance(e, Win):
            queue.append(stack.pop())

for items in [queue, block]:
    for item in items:
        out(*item)
