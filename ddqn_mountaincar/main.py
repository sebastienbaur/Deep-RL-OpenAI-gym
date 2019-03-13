"""
DQN on OpenAI's MountainCar problem

- Implementation of DQN : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Works with the OpenAI gym environment
- Modified reward function


TODO:
- plotter un scatter plot (position, vitesse) dont les points ont la couleur de l'action choisie (rouge, jaune, vert)
- plotter les valeurs

"""
from itertools import product
from copy import deepcopy
import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from utils import color, create_exp_dir
from ddqn_mountaincar.utils import DQN, update_eval_network, update_min_max_pos, build_target, huber
from tqdm import tqdm
from ddqn_mountaincar.per import PrioritizedReplayBuffer


pos_grid = np.linspace(-1.2, .6, 16)
speed_grid = np.linspace(-0.07, 0.07, 16)
pos_speed_grid = np.array(list(product(pos_grid, speed_grid)))


env = gym.make('MountainCar-v0')

# INIT
# Neural networks
dqn = DQN(hdim=100)
dqn_eval = deepcopy(dqn)
lr = 5e-3
momentum = .5
optim = torch.optim.RMSprop(dqn.parameters(), lr=lr, momentum=momentum)
batch_size = 128

# Hyperparams
tau = 10000
gamma = .995  # discount factor
N_RANDOM = 40000.  # the number of steps it takes to go from eps=1. to eps=0.1
N_EPISODES = 1500
MAX_SIZE_BUFFER = 100000  # the maximum number of transitions in the buffer
T_START_LEARNING = batch_size * 3  # the minimum number of transitions to be seen in the buffer before training
alpha = .6
beta = .4
per_eps = 1e-6

# Monitoring
t = 0  # step counts
successes = 0
writer = SummaryWriter(log_dir=create_exp_dir())

# Memory
replay_memory = PrioritizedReplayBuffer(MAX_SIZE_BUFFER, alpha)
observation_t = None
loss = None
wloss = None
nonzero_rewards_in_batch = None

# # FILL MEMORY WITH RANDOM SAMPLES
# policies = np.array([
#     [.75, 0, .25],
#     [.25, 0, .75],
#     [.375, .25, .375]
# ])
# print('policies')
# print(policies)
#
# for policy in policies:
#     print(policy)
#     for n_episode in tqdm(range(100)):
#         observation_t = env.reset()
#         observation_tp1 = None
#         done = False
#         max_pos = -np.inf
#         min_pos = np.inf
#         while not done:
#             env.render()
#
#             # TAKE ACTION
#             action = np.random.choice(list(range(3)), 1, p=policy)[0]  # eps-greedy action selection
#             observation_t = observation_tp1*1. if observation_tp1 is not None else observation_t
#             observation_tp1, reward, done, _ = env.step(action)
#             pos = observation_tp1[0]*1.
#
#             # UPDATE REWARD
#             min_pos, max_pos = update_min_max_pos(pos, min_pos, max_pos)
#             reward = 0  # pos + .5  # modify reward to encourage getting closer to the flag
#             if done:
#                 reward += max_pos - min_pos  # 10  # if you reached the goal, adjust reward
#                 if pos >= .5:
#                     successes += 1
#                     reward = 10
#
#             # EXPERIENCE REPLAY
#             replay_memory.add(observation_t, action, reward, observation_tp1, done)

print('Learning')

# LEARN
for n_episode in range(N_EPISODES):
    print('episode %d/%d' % (n_episode, N_EPISODES))
    print('Last final pos: %.2f\n' % observation_t[0]) if observation_t is not None else print()
    observation_t = env.reset()
    observation_tp1 = None
    cum_reward = 0
    done = False
    t_ep = 0
    max_pos = -np.inf
    min_pos = np.inf
    while not done:
        update_eval_network(dqn_eval, dqn, t, tau)

        env.render()

        t += 1
        t_ep += 1
        eps = float(np.clip(1 - t / N_RANDOM, .1, 1.))

        # TAKE ACTION
        x = torch.from_numpy(observation_t)
        dqn.eval()
        action = dqn.action(x, eps=eps).numpy()  # eps-greedy action selection
        observation_t = observation_tp1*1. if observation_tp1 is not None else observation_t
        observation_tp1, reward, done, _ = env.step(action)
        pos = observation_tp1[0]*1.

        # UPDATE REWARD
        min_pos, max_pos = update_min_max_pos(pos, min_pos, max_pos)
        reward = 0
        if done:
            if pos >= .5:
                successes += 1
                reward = 10
            else:
                reward += max_pos - min_pos
        cum_reward += (gamma ** t_ep) * reward

        # STORE IN MEMORY
        replay_memory.add(observation_t, action, reward, observation_tp1, done)

        # TRAIN DQN IF ENOUGH SAVED TRANSITIONS
        if (len(replay_memory) > T_START_LEARNING) and (t % 3 == 0):

            # Sample batch
            s, a, r, s_, d, w, idx = replay_memory.sample(batch_size, beta)
            s = torch.from_numpy(s)
            a = torch.from_numpy(a).view(-1, 1)
            r = torch.from_numpy(r)
            s_ = torch.from_numpy(s_)
            d = torch.from_numpy(d.astype(int)).byte().view(-1)
            w = torch.from_numpy(w).float().view(-1)

            nonzero_rewards_in_batch = (r.detach() > 0).float().sum().numpy()
            target = build_target(dqn, dqn_eval, r, s_, d, gamma)

            # Train
            # Compute loss
            dqn.train()
            q = dqn.q_values(s, a)  # q(s_i, a_i) for all elements of the batch
            td_error = q - target
            loss = huber(td_error)
            wloss = torch.mean(w*loss)
            # Backprop
            optim.zero_grad()
            wloss.backward()
            optim.step()

            # Update priorirites
            replay_memory.update_priorities(idx, torch.abs(td_error).detach().numpy() + per_eps)

        # MONITORING
        if done:

            dqn.eval()
            actions, q_scores = dqn.action(torch.from_numpy(pos_speed_grid).float(), eps=0, return_score=True)
            actions = actions.numpy().reshape(-1)
            q_scores = q_scores.detach().numpy()

            fig = plt.figure(1)
            plt.clf()
            plt.scatter(pos_speed_grid[:, 0], pos_speed_grid[:, 1], c=color(actions))
            writer.add_figure('policy', fig, global_step=t)

            fig = plt.figure(2)
            plt.clf()
            plt.hist(q_scores, bins=int(np.sqrt(q_scores.shape[0]))+1, color=('red', 'yellow', 'green'))
            writer.add_figure('q_scores', fig, global_step=t)

            writer.add_histogram('fc1', dqn.fc1.weight.detach().numpy(), global_step=t)
            writer.add_histogram('fc2', dqn.fc2.weight.detach().numpy(), global_step=t)
            writer.add_histogram('fc3', dqn.fc3.weight.detach().numpy(), global_step=t)

            writer.add_scalar('num_successes_until_now', successes, global_step=t)
            writer.add_scalar('cum_reward_in_episode', cum_reward, global_step=t)
            writer.add_scalar('num_steps_in_episode', t_ep, global_step=t)
            writer.add_scalar('loss', loss.mean().detach().numpy(), global_step=t) if loss is not None else None
            writer.add_scalar('wloss', wloss.detach().numpy(), global_step=t) if wloss is not None else None
            writer.add_scalar('nonzero_rewards_in_batch', nonzero_rewards_in_batch, global_step=t) if nonzero_rewards_in_batch is not None else None

            break

env.close()

