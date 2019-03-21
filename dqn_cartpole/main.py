"""
DQN on OpenAI's cart-pole balancing problem

- Implementation of DQN : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Works with the OpenAI gym environment
"""
import gym
import numpy as np
import torch as t
import torch.nn.functional as F
from matplotlib import pyplot as plt
from dqn_mountaincar.utils import moving_average, Buffer


class DQN(t.nn.Module):
    """
    3 layered fully-connected neural network with batch norm
    It takes the state as inputs and outputs Q(s,0), Q(s,1)
    """
    def __init__(self, hdim=100):
        super(DQN, self).__init__()

        # sequence of blocks FC + BN + ReLU
        self.fc1 = t.nn.Linear(4, hdim)  # 4d state space
        self.bn1 = t.nn.BatchNorm1d(hdim)
        self.fc2 = t.nn.Linear(hdim, hdim)
        self.bn2 = t.nn.BatchNorm1d(hdim)
        self.fc3 = t.nn.Linear(hdim, 2)  # one output per action
        self.bn3 = t.nn.BatchNorm1d(2)

    def forward(self, x):
        # reshape if necessary
        xx = x*1. if len(x.shape) == 2 else x.view(1, -1)

        # forward pass
        xx = F.relu(self.bn1(self.fc1(xx.float())))
        xx = F.relu(self.bn2(self.fc2(xx)))
        xx = F.relu(self.bn3(self.fc3(xx)))

        # reshape if necessary
        xx = xx if len(x.shape) == 2 else xx.view(-1)
        return xx

    def action(self, x, eps=.1):
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
            return t.argmax(values, dim=1 if len(x.shape) == 2 else 0)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # INIT
    dqn = DQN()
    lr = 1e-2
    optim = t.optim.RMSprop(dqn.parameters(), lr=lr)
    batch_size = 128

    gamma = 1.  # discount factor

    i = 0  # step counts

    N_RANDOM = 1000.  # the number of steps it takes to go from eps=0.95 to eps=0.05
    N_EPISODES = 500
    N_MIN_TRANSITIONS = batch_size*3  # the minimum number of transitions to be seen in the buffer before starting training
    MAX_SIZE_BUFFER = 10000  # the maximum number of transitions in the buffer

    replay_memory = Buffer(MAX_SIZE_BUFFER)
    cum_rewards = []

    for i_episode in range(N_EPISODES):
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
            action = dqn.action(x, eps=eps).numpy()  # eps-greedy action selction
            transition = {'s': observation*1.}
            observation, reward, done, info = env.step(action)
            cum_reward += reward

            # STOP IF 200 SUCCESSFUL STEPS
            if i_ep > 200:
                done = True

            # EXPERIENCE REPLAY
            transition['a'] = action
            transition["s'"] = observation*1.
            transition['d'] = done
            transition['r'] = reward

            replay_memory.add(transition)

            # TRAIN DQN IF ENOUGH SAVED TRANSITIONS
            if replay_memory.n > N_MIN_TRANSITIONS:
                optim.zero_grad()

                batch = replay_memory.sample(batch_size)
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

                loss.backward()
                optim.step()

            if done:
                cum_rewards.append(cum_reward)
                cum_reward = 0
                break

    env.close()

    plt.plot(cum_rewards, label='cum_r')
    plt.plot(moving_average(cum_rewards, 25), label='smoothed cum_r')
    plt.ylabel('Cumulated reward')
    plt.xlabel('# episodes')
    plt.title('Training of DQN on the cart-pole balancing task')
    plt.legend()
    plt.show()
