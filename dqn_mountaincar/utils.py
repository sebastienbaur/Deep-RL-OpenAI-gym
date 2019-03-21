import os
import random
from random import sample
import numpy as np
from collections import deque
from scipy.special import softmax


class PrioritizedBuffer:
    def __init__(self, max_size, temp=30.):
        self.memory = deque()
        self.priorities = deque()
        self.max_size = max_size
        self.temp = temp

    def add(self, x, priority):
        self.memory.append(x)
        self.priorities.append(priority)

    def sample(self, n):
        return np.random.choice(self.memory, n, replace=False, p=softmax(np.array(self.priorities)*self.temp))

    @property
    def n(self):
        return len(self.memory)


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


def create_exp_dir():
    i = 1
    while True:
        if os.path.isdir('./tensorboard/exp%d'%i):
            i += 1
        else:
            break
    os.mkdir('./tensorboard/exp%d'%i)
    return 'tensorboard/exp%d'%i

