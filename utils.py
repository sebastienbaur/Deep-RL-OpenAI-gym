import random
from random import sample
import numpy as np


class Buffer:
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size

    def add(self, x):
        if len(self.memory) <= self.max_size:
            self.memory.append(x)
        else:
            self.memory[random.randint(0, len(self.memory)-1)] = x

    def sample(self, n):
        return sample(self.memory, n)

    @property
    def n(self):
        return len(self.memory)


def moving_average(a, n=25) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return n*[None] + (ret[n - 1:] / n).tolist()


def color(actions):
    colors = []
    for a in actions:
        if a == 0:
            colors.append('red')
        elif a == 1:
            colors.append('yellow')
        else:
            colors.append('green')
    return colors
