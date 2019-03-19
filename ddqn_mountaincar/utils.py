import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from scipy.special import softmax
import os


###########################################
# NOT TESTED
###########################################


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


class DQN(torch.nn.Module):
    """
    3 layered fully-connected neural network with batch norm
    It takes the state as inputs and outputs Q(s,0), Q(s,1), Q(s,2)
    """
    def __init__(self, hdim=100):
        super(DQN, self).__init__()

        # sequence of blocks FC + BN + ReLU
        self.fc1 = torch.nn.Linear(2, hdim, bias=False)  # 2d state space
        self.bn1 = torch.nn.BatchNorm1d(hdim)
        self.fc2 = torch.nn.Linear(hdim, hdim, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(hdim)
        self.fc3 = torch.nn.Linear(hdim, 3, bias=True)  # one output per action
        # self.bn3 = torch.nn.BatchNorm1d(3)
        # self.sign = sign

    def forward(self, x):
        # reshape if necessary
        xx = x*1. if len(x.shape) == 2 else x.view(1, -1)

        # forward pass
        xx = self.bn1(F.relu(self.fc1(xx.float())))
        xx = self.bn2(F.relu(self.fc2(xx)))
        xx = self.fc3(xx)

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
            values = torch.rand_like(values)
            return torch.argmax(values, dim=1 if len(x.shape) == 2 else 0)
        else:
            return torch.argmax(values, dim=1 if len(x.shape) == 2 else 0) if not return_score else (torch.argmax(values, dim=1 if len(x.shape) == 2 else 0), values)

    def q_values(self, s, a):
        return self.forward(s).gather(1, a).view(-1)


def parse_batch(batch):
    a = torch.from_numpy(np.stack([np.array([transition['a']]) for transition in batch])).long()
    s = torch.from_numpy(np.stack([transition['s'] for transition in batch]))
    r = torch.from_numpy(np.stack([np.array([transition['r']]) for transition in batch]))
    s_ = torch.from_numpy(np.stack([transition["s'"] for transition in batch]))
    d = torch.from_numpy(np.stack([np.array([transition['d']])*1. for transition in batch])).byte().view(-1)
    return a,s,r,s_,d


def build_target(dqn, dqn_eval, r, s_, d, gamma):
    dqn.eval()
    dqn_eval.eval()

    target = r.view(-1).float()

    q = dqn.forward(s_)
    greedy_actions = torch.max(q + torch.rand_like(q) / 1e6, 1)[1].view(-1, 1)   # actions that maximizes `Q(S_t+1, a)` w.r.t. `a`
    update_target = dqn_eval.forward(s_).gather(1, greedy_actions).view(-1)  # values of these actions `a` with the evaluation network `Q'(S_t+1, a)`
    update_target = (gamma * update_target[1 - d]).float()

    target[1 - d] += update_target  # update only those transitions that are not done
    target.detach_()  # don't propagate gradients through this

    return target


###########################################
# TESTED
###########################################

def update_eval_network(dqn_eval, dqn, i, tau):
    if i % tau == 0:
        dqn_eval.load_state_dict(dqn.state_dict())


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
    mask = (torch.abs(x) < 1.).float()
    return mask*.5*torch.pow(x, 2) + (1.-mask)*(torch.abs(x) - .5)


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


def _reward(pos):
    return np.exp(.1/(.6-pos))


def update_reward(pos, done, successes):
    reward = _reward(pos) - _reward(-.3)
    if done:
        if pos >= .5:
            reward += 3
            successes += 1
    return reward, successes


def update_min_max_pos(pos, min_pos, max_pos):
    min_pos = min(min_pos, pos)
    max_pos = max(max_pos, pos)
    return min_pos, max_pos


def tensorboard(dqn, pos_speed_grid, writer, t, cum_reward, successes, t_ep, loss, wloss, r):
    dqn.eval()
    actions, q_scores = dqn.action(torch.from_numpy(pos_speed_grid).float(), eps=0, return_score=True)
    actions = actions.numpy().reshape(-1)
    q_scores = q_scores.detach().numpy()

    # policy
    fig = plt.figure(1)
    plt.clf()
    plt.scatter(pos_speed_grid[:, 0], pos_speed_grid[:, 1], c=color(actions))
    writer.add_figure('policy', fig, global_step=t)

    # histograms of the scores, per action
    fig = plt.figure(2)
    plt.clf()
    plt.hist(q_scores, bins=int(np.sqrt(q_scores.shape[0]))+1, color=('red', 'yellow', 'green'))
    writer.add_figure('q_scores', fig, global_step=t)

    # weights of the neural network
    writer.add_histogram('fc1', dqn.fc1.weight.detach().numpy(), global_step=t)
    writer.add_histogram('fc2', dqn.fc2.weight.detach().numpy(), global_step=t)
    writer.add_histogram('fc3', dqn.fc3.weight.detach().numpy(), global_step=t)

    # a few scalars to monitor the performance of the agent (cumulative rewards and loss essentially)
    writer.add_scalar('num_successes_until_now', successes, global_step=t)
    writer.add_scalar('cum_reward_in_episode', cum_reward, global_step=t)
    writer.add_scalar('num_steps_in_episode', t_ep, global_step=t)
    writer.add_scalar('loss', loss.mean().detach().numpy(), global_step=t) if loss is not None else None
    writer.add_scalar('wloss', wloss.detach().numpy(), global_step=t) if wloss is not None else None
    writer.add_scalar('nonzero_rewards_in_batch', (r.detach() > 0).float().sum().numpy(), global_step=t) if r is not None else None
