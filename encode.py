from numpy import array, arange


def encode(state):
    return array([state, arange(state.shape[0])])
