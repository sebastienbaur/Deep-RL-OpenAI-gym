"""
DDQN on OpenAI's MountainCar problem

- Implementation of DDQN : https://arxiv.org/abs/1509.06461
- Works with the OpenAI gym environment (MountainCar-v0)
- Modified reward function (see utils.update_reward)
- Use OpenAI PER code (https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)



TODO:
Essayer de comprendre pourquoi ca n'apprend pas

---> remplir la memoire avec des samples qui reussissent
---> pre-train Q sur la reward cumulee
---> visualiser l'erreur V(s) - Q(s,a) a intervalles reguliers
---> Essayer d'autres reward schemes
---> essayer le meme code sur l'environnement Cartpole balancing
--->

"""


from copy import deepcopy
from itertools import product
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from ddqn_mountaincar.per import PrioritizedReplayBuffer
from ddqn_mountaincar.utils import DQN, update_eval_network, build_target, huber, update_reward, tensorboard
from utils import create_exp_dir


# Hyperparams
BATCH_SIZE = 64
TAU = 200  # the update frequency of the target network
GAMMA = .999  # discount factor
N_RANDOM = 40000.  # the number of steps it takes to go from eps=1. to eps=0.05 (exponentially decreasing, loses 99% of the .95 in 40000 steps)
N_EPISODES = 1500
MAX_SIZE_BUFFER = 20000  # the maximum number of transitions in the buffer
T_START_LEARNING = BATCH_SIZE * 10  # the minimum number of transitions to be seen in the buffer before training
ALPHA = .6  # default OpenAI
BETA = .4  # default OpenAI
PER_EPS = 1e-6


# To visualize the policy, create a grid of the state space
pos_grid = np.linspace(-1.2, .6, 16)
speed_grid = np.linspace(-0.07, 0.07, 16)
pos_speed_grid = np.array(list(product(pos_grid, speed_grid)))


# OpenAI's environment
env = gym.make('MountainCar-v0')


# Neural networks and optimizer
dqn = DQN(hdim=100)
dqn_eval = deepcopy(dqn)
dqn_eval.eval()
lr = 2e-4
momentum = .05
optim = torch.optim.Adam(dqn.parameters(), lr=lr)  # RMSprop(dqn.parameters(), lr=lr, momentum=momentum)


# Monitoring
t = 0  # step counts
successes = 0
writer = SummaryWriter(log_dir=create_exp_dir())


# Memory
replay_memory = PrioritizedReplayBuffer(MAX_SIZE_BUFFER, ALPHA)


# Misc
observation_t = None
loss = None
wloss = None
r = None


# LEARN
print('Learning')
for n_episode in range(N_EPISODES):
    print('episode %d/%d' % (n_episode, N_EPISODES))
    print('Last final pos: %.2f\n' % observation_t[0]) if observation_t is not None else print()

    # Init beginning of episode
    observation_t = env.reset()
    observation_tp1 = None
    cum_reward = 0
    done = False
    t_ep = 0

    while not done:
        dqn.eval()

        # Copy weights of eval network if necessary
        update_eval_network(dqn_eval, dqn, t, TAU) if t > T_START_LEARNING else None

        env.render()
        t += 1
        t_ep += 1
        eps = .05 + .95*np.exp(-t*5 / N_RANDOM)

        # OBSERVE
        x = torch.from_numpy(observation_t)
        observation_t = observation_tp1*1. if observation_tp1 is not None else observation_t

        # TAKE ACTION
        action = dqn.action(x, eps=eps).numpy()  # eps-greedy action selection
        observation_tp1, _, done, _ = env.step(action)
        pos = observation_tp1[0]*1.

        # UPDATE REWARD (the reward provided by the environment is too sparse and does not help the algorithm to learn)
        reward, successes = update_reward(pos, done, successes)
        cum_reward += (GAMMA ** t_ep) * reward

        # STORE IN MEMORY
        replay_memory.add(observation_t, action, reward, observation_tp1, done)

        # TRAIN
        if len(replay_memory) > T_START_LEARNING:  # if enough saved transitions
            if t % 3 == 0:  # TRAIN NOT TOO FREQUENTLY
                dqn.train()

                # Sample batch
                s, a, r, s_, d, w, idx = replay_memory.sample(BATCH_SIZE, BETA, to_tensor=True)

                # Compute loss
                q = dqn.q_values(s, a)  # q(s_i, a_i) for all elements of the batch
                target = build_target(dqn, dqn_eval, r, s_, d, GAMMA)
                td_error = q - target
                loss = huber(td_error)
                wloss = torch.mean(w*loss)

                # Update weights
                optim.zero_grad()
                wloss.backward()
                optim.step()

                # Update priorities
                replay_memory.update_priorities(idx, torch.abs(td_error).detach().numpy() + PER_EPS)

        # MONITORING
        if done:
            tensorboard(dqn, pos_speed_grid, writer, t, cum_reward, successes, t_ep, loss, wloss, r)
            break

env.close()

