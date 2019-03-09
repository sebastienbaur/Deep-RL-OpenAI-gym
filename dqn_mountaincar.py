"""
DQN on OpenAI's MountainCar problem

- Implementation of DQN : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Works with the OpenAI gym environment
- Modified reward function


TODO:
- plotter un scatter plot (position, vitesse) dont les points ont la couleur de l'action choisie (rouge, jaune, vert)
- plotter les valeurs

"""
from random import sample

import gym
import numpy as np
import torch as t
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils import moving_average, color
from itertools import product
from tensorboardX import SummaryWriter
from collections import deque


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
        self.bn3 = t.nn.BatchNorm1d(3)
        # self.sign = sign

    def forward(self, x):
        # reshape if necessary
        xx = x*1. if len(x.shape) == 2 else x.view(1, -1)

        # forward pass
        xx = self.bn1(F.relu(self.fc1(xx.float())))
        # xx = F.relu(self.fc1(xx.float()))
        xx = self.bn2(F.relu(self.fc2(xx)))
        # xx = F.relu(self.fc2(xx))
        xx = self.bn3(self.fc3(xx))
        # xx = self.fc3(xx)

        # reshape if necessary
        xx = xx if len(x.shape) == 2 else xx.view(-1)
        return F.relu(xx)

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


pos_grid = np.linspace(-1.2, .6, 16)
speed_grid = np.linspace(-0.07, 0.07, 16)
pos_speed_grid = np.array(list(product(pos_grid, speed_grid)))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    # INIT
    dqn = DQN(hdim=100)
    writer = SummaryWriter(log_dir='tensorboard')
    lr = 1e-3
    optim = t.optim.RMSprop(dqn.parameters(), lr=lr)
    batch_size = 128

    gamma = .99  # discount factor

    i = 0  # step counts

    N_RANDOM = 40000.  # the number of steps it takes to go from eps=0.95 to eps=0.1
    N_EPISODES = 1500
    MAX_SIZE_BUFFER = 10000  # the maximum number of transitions in the buffer
    N_MIN_TRANSITIONS = batch_size * 8  # the minimum number of transitions to be seen in the buffer before starting training

    replay_memory = deque(maxlen=MAX_SIZE_BUFFER)  # Buffer(MAX_SIZE_BUFFER)
    cum_rewards = []
    i_eps = []
    observation = None

    for i_episode in range(N_EPISODES):
        print('episode %d/%d' % (i_episode, N_EPISODES))
        print('Last final pos: %.2f\n' % observation[0]) if observation is not None else print()
        observation = env.reset()
        cum_reward = 0
        done = False
        i_ep = 0
        while not done:
            env.render()

            i += 1
            i_ep += 1
            eps = float(np.clip(1 - i/N_RANDOM, 0.05, .95))

            # TAKE ACTION
            x = t.from_numpy(observation)
            dqn.eval()
            action = dqn.action(x, eps=eps).numpy()  # eps-greedy action selection
            transition = {'s': observation*1.}
            observation, reward, done, info = env.step(action)

            reward = 0  # observation[0] + .5
            if done and (observation[0] >= .5):
                reward += 1
            cum_reward += (gamma**i_ep)*reward

            # EXPERIENCE REPLAY
            transition['a'] = action
            transition["s'"] = observation*1.
            transition['d'] = done
            transition['r'] = reward

            replay_memory.append(transition)

            # TRAIN DQN IF ENOUGH SAVED TRANSITIONS
            if len(replay_memory) > N_MIN_TRANSITIONS:  # replay_memory.n > N_MIN_TRANSITIONS:

                batch = sample(replay_memory, batch_size)  # replay_memory.sample(batch_size)
                a = t.from_numpy(np.stack([np.array([transition['a']]) for transition in batch])).long()
                s = t.from_numpy(np.stack([transition['s'] for transition in batch]))
                r = t.from_numpy(np.stack([np.array([transition['r']]) for transition in batch]))
                s_ = t.from_numpy(np.stack([transition["s'"] for transition in batch]))
                d = t.from_numpy(np.stack([np.array([transition['d']])*1. for transition in batch])).byte().view(-1)

                dqn.eval()
                target = r.view(-1).float()
                target[1 - d] += (gamma * t.max(dqn.forward(s_), 1)[0][1 - d]).float()
                target.detach_()  # don't propagate gradients through this

                dqn.train()
                q = dqn.forward(s).gather(1, a).view(-1)
                loss = t.pow(q - target, 2)
                loss = t.mean(loss)

                optim.zero_grad()
                loss.backward()
                optim.step()

            if done:
                dqn.eval()
                actions, q_scores = dqn.action(t.from_numpy(pos_speed_grid).float(), eps=0, return_score=True)
                actions = actions.numpy().reshape(-1)
                q_scores = q_scores.detach().numpy()

                fig = plt.figure(1)
                plt.clf()
                plt.scatter(pos_speed_grid[:, 0], pos_speed_grid[:, 1], c=color(actions))
                writer.add_figure('policy', fig)

                fig = plt.figure(2)
                plt.clf()
                plt.hist(q_scores, bins=int(np.sqrt(q_scores.shape[0]))+1, color=('red', 'yellow', 'green'))
                writer.add_figure('q_scores', fig)

                fig = plt.figure(3)
                plt.plot(cum_rewards, label='cum_r', c='g')
                plt.plot(moving_average(cum_rewards, 25), label='smoothed cum_r', c='r')
                writer.add_figure('cum_r', fig)

                writer.add_histogram('fc1', dqn.fc1.weight.detach().numpy(), global_step=i)
                writer.add_histogram('fc2', dqn.fc2.weight.detach().numpy(), global_step=i)
                writer.add_histogram('fc3', dqn.fc3.weight.detach().numpy(), global_step=i)

                cum_rewards.append(cum_reward)
                i_eps.append(i_ep)
                break

    env.close()

    plt.plot(i_eps, label='#steps')
    plt.plot(moving_average(i_eps, 25), label='smoothed #steps')
    plt.ylabel('Number of steps before the end of the episode')
    plt.xlabel('# episodes')
    plt.title('Training of DQN on the MountainCar task')
    plt.legend()
    plt.show()
