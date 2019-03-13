import torch as t
import torch.nn.functional as F
import numpy as np
from collections import deque
from scipy.special import softmax
import os


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


class DQN(t.nn.Module):
    """
    3 layered fully-connected neural network with batch norm
    It takes the state as inputs and outputs Q(s,0), Q(s,1), Q(s,2)
    """
    def __init__(self, hdim=100):
        super(DQN, self).__init__()

        # sequence of blocks FC + BN + ReLU
        self.fc1 = t.nn.Linear(2, hdim, bias=False)  # 2d state space
        self.bn1 = t.nn.BatchNorm1d(hdim)
        self.fc2 = t.nn.Linear(hdim, hdim, bias=False)
        self.bn2 = t.nn.BatchNorm1d(hdim)
        self.fc3 = t.nn.Linear(hdim, 3, bias=True)  # one output per action
        # self.bn3 = t.nn.BatchNorm1d(3)
        # self.sign = sign

    def forward(self, x):
        # reshape if necessary
        xx = x*1. if len(x.shape) == 2 else x.view(1, -1)

        # forward pass
        xx = self.bn1(F.relu(self.fc1(xx.float())))
        # xx = F.relu(self.fc1(xx.float()))
        xx = self.bn2(F.relu(self.fc2(xx)))
        # xx = F.relu(self.fc2(xx))
        # xx = F.relu(self.bn3(self.fc3(xx)))
        xx = F.relu(self.fc3(xx))

        # reshape if necessary
        xx = xx if len(x.shape) == 2 else xx.view(-1)
        return xx

    def action(self, x, return_score=False, eps=.1):
        """
        Choose action in epsilon greedy fashion
        :param x:
        :param eps: the probaility of selecting a random action
        :return: an action, or one per element of the batch
        """
        values = self.forward(x)
        u = np.random.random()
        if u < eps:
            values = t.rand_like(values)
            return t.argmax(values, dim=1 if len(x.shape) == 2 else 0)
        else:
            return t.argmax(values, dim=1 if len(x.shape) == 2 else 0) if not return_score else (t.argmax(values, dim=1 if len(x.shape) == 2 else 0), values)

    def q_values(self, s, a):
        return self.forward(s).gather(1, a).view(-1)


def update_eval_network(dqn_eval, dqn, i, tau):
    if i % tau == 0:
        dqn_eval.load_state_dict(dqn.state_dict())


def update_min_max_pos(pos, min_pos, max_pos):
    if pos > max_pos:
        max_pos = pos*1.
    if pos < min_pos:
        min_pos = pos*1.
    return min_pos, max_pos


def parse_batch(batch):
    a = t.from_numpy(np.stack([np.array([transition['a']]) for transition in batch])).long()
    s = t.from_numpy(np.stack([transition['s'] for transition in batch]))
    r = t.from_numpy(np.stack([np.array([transition['r']]) for transition in batch]))
    s_ = t.from_numpy(np.stack([transition["s'"] for transition in batch]))
    d = t.from_numpy(np.stack([np.array([transition['d']])*1. for transition in batch])).byte().view(-1)
    return a,s,r,s_,d


def build_target(dqn, dqn_eval, r, s_, d, gamma):
    dqn.eval()
    dqn_eval.eval()
    target = r.view(-1).float()
    greedy_actions = t.max(dqn.forward(s_), 1)[1].view(-1, 1)
    update_target = dqn_eval.forward(s_).gather(1, greedy_actions).view(-1)
    update_target = (gamma * update_target[1 - d]).float()
    target[1 - d] += update_target
    # target[1 - d] += (gamma * t.max(dqn.forward(s_), 1)[0][1 - d]).float()
    target.detach_()  # don't propagate gradients through this
    return target


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


def huber(x):
    mask = (t.abs(x) < 1.).float()
    return mask*.5*t.pow(x, 2) + (1.-mask)*(t.abs(x) - .5)
