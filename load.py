from os import chdir
from learn.load import load as fn


def load(checkpointer):
    chdir("win")
    win = fn(checkpointer)
    chdir("../block")
    block = fn(checkpointer)
    chdir("..")
    return win, block
