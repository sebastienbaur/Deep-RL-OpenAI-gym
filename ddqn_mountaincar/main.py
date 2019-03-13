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
from collections import Counter
from copy import deepcopy
import gym
import numpy as np
import torch as t
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from utils import color, create_exp_dir, PrioritizedBuffer
from ddqn_mountaincar.utils import DQN, update_eval_network, update_min_max_pos, parse_batch, build_target
from tqdm import tqdm
import torch.nn.functional as F


pos_grid = np.linspace(-1.2, .6, 16)
speed_grid = np.linspace(-0.07, 0.07, 16)
pos_speed_grid = np.array(list(product(pos_grid, speed_grid)))


env = gym.make('MountainCar-v0')

# INIT
# Neural networks
dqn = DQN(hdim=200)
dqn_eval = deepcopy(dqn)
lr = 1e-3
optim = t.optim.RMSprop(dqn.parameters(), lr=lr, momentum=.95)
batch_size = 128

# Hyperparams
tau = 10000
gamma = .99  # discount factor
N_RANDOM = 40000.  # the number of steps it takes to go from eps=1. to eps=0.1
N_EPISODES = 1500
MAX_SIZE_BUFFER = 100000  # the maximum number of transitions in the buffer
N_MIN_TRANSITIONS = batch_size * 10  # the minimum number of transitions to be seen in the buffer before training
TEMP = 25.

# Monitoring
i = 0  # step counts
successes = 0
writer = SummaryWriter(log_dir=create_exp_dir())

# Memory
replay_memory = PrioritizedBuffer(MAX_SIZE_BUFFER, temp=TEMP)
observation = None
loss = None
nonzero_rewards_in_batch = None

# FILL MEMORY WITH RANDOM SAMPLES
policies = np.random.dirichlet((1, .75, 1), size=5)
print('policies')
print(policies)

for policy in policies:
    print(policy)
    for i_episode in tqdm(range(MAX_SIZE_BUFFER//(200*5))):
        observation = env.reset()
        done = False
        max_pos = -np.inf
        min_pos = np.inf
        while not done:
            env.render()

            # TAKE ACTION
            action = np.random.choice(list(range(3)), 1, p=policy)[0]  # eps-greedy action selection
            transition = {'s': observation*1.}
            observation, reward, done, info = env.step(action)
            pos = observation[0]*1.

            # UPDATE REWARD
            min_pos, max_pos = update_min_max_pos(pos, min_pos, max_pos)
            reward = 0  # pos + .5  # modify reward to encourage getting closer to the flag
            if done:
                reward += max_pos - min_pos  # 10  # if you reached the goal, adjust reward
                if pos >= .5:
                    successes += 1
                    reward += 10

            # EXPERIENCE REPLAY
            transition['a'] = action
            transition["s'"] = observation*1.
            transition['d'] = done
            transition['r'] = reward
            replay_memory.add(transition, reward)

print('Proportion of nonzero rewards in the random exploration')
print(Counter([x > 0 for x in replay_memory.priorities]))

print('Learning')

# LEARN
for i_episode in range(N_EPISODES):
    print('episode %d/%d' % (i_episode, N_EPISODES))
    print('Last final pos: %.2f\n' % observation[0]) if observation is not None else print()
    observation = env.reset()
    cum_reward = 0
    done = False
    i_ep = 0
    max_pos = -np.inf
    min_pos = np.inf
    while not done:
        update_eval_network(dqn_eval, dqn, i, tau)

        env.render()

        i += 1
        i_ep += 1
        eps = float(np.clip(1 - i/N_RANDOM, .1, 1.))

        # TAKE ACTION
        x = t.from_numpy(observation)
        dqn.eval()
        action = dqn.action(x, eps=eps).numpy()  # eps-greedy action selection
        transition = {'s': observation*1.}
        observation, reward, done, info = env.step(action)
        pos = observation[0]*1.

        # UPDATE REWARD
        min_pos, max_pos = update_min_max_pos(pos, min_pos, max_pos)
        reward = 0  # pos + .5  # modify reward to encourage getting closer to the flag
        if done:
            reward += max_pos - min_pos  # 10  # if you reached the goal, adjust reward
            if pos >= .5:
                successes += 1
                reward += 10
        cum_reward += (gamma**i_ep)*reward

        # STORE IN MEMORY
        transition['a'] = action
        transition["s'"] = observation*1.
        transition['d'] = done
        transition['r'] = reward
        replay_memory.add(transition, reward)

        # TRAIN DQN IF ENOUGH SAVED TRANSITIONS
        if (replay_memory.n > N_MIN_TRANSITIONS) and (i % 3 == 0):

            # Sample batch
            batch = replay_memory.sample(batch_size)
            a, s, r, s_, d = parse_batch(batch)
            nonzero_rewards_in_batch = (r.detach() > 0).float().sum().numpy()
            target = build_target(dqn, dqn_eval, r, s_, d, gamma)

            # Train
            # Compute loss
            dqn.train()
            q = dqn.q_values(s, a)  # q(s_i, a_i) for all elements of the batch
            loss = F.smooth_l1_loss(q, target, size_average=True)
            # loss = t.pow(q - target, 2)
            # loss = t.mean(loss)
            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()

        # MONITORING
        if done:

            dqn.eval()
            actions, q_scores = dqn.action(t.from_numpy(pos_speed_grid).float(), eps=0, return_score=True)
            actions = actions.numpy().reshape(-1)
            q_scores = q_scores.detach().numpy()

            fig = plt.figure(1)
            plt.clf()
            plt.scatter(pos_speed_grid[:, 0], pos_speed_grid[:, 1], c=color(actions))
            writer.add_figure('policy', fig, global_step=i)

            fig = plt.figure(2)
            plt.clf()
            plt.hist(q_scores, bins=int(np.sqrt(q_scores.shape[0]))+1, color=('red', 'yellow', 'green'))
            writer.add_figure('q_scores', fig, global_step=i)

            writer.add_histogram('fc1', dqn.fc1.weight.detach().numpy(), global_step=i)
            writer.add_histogram('fc2', dqn.fc2.weight.detach().numpy(), global_step=i)
            writer.add_histogram('fc3', dqn.fc3.weight.detach().numpy(), global_step=i)

            writer.add_scalar('num_successes_until_now', successes, global_step=i)
            writer.add_scalar('cum_reward_in_episode', cum_reward, global_step=i)
            writer.add_scalar('num_steps_in_episode', i_ep, global_step=i)
            writer.add_scalar('loss', loss.detach().numpy(), global_step=i) if loss is not None else None
            writer.add_scalar('nonzero_rewards_in_batch', nonzero_rewards_in_batch, global_step=i) if nonzero_rewards_in_batch is not None else None

            break

env.close()

