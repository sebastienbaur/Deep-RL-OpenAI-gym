import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os


###########################################
# NOT TESTED
###########################################


class DQN(tf.keras.Model):
    """
    3 layered fully-connected neural network with batch norm
    It takes the state as inputs and outputs Q(s,0), Q(s,1), Q(s,2)
    """
    def __init__(self, hdim=100):
        super(DQN, self).__init__()

        # sequence of blocks FC + BN + ReLU
        self.fc1 = tf.keras.layers.Dense(hdim, use_bias=False, activation='relu')  # 2d state space
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(hdim, use_bias=False, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(3, use_bias=True, activation=None)  # one output per action

    def call(self, x):
        # reshape if necessary
        xx = x*1. if len(x.shape) == 2 else tf.keras.layers.Reshape((-1, 1))(x)

        # forward pass
        xx = self.bn1(self.fc1(xx.float()))
        xx = self.bn2(self.fc2(xx))
        xx = self.fc3(xx)

        # reshape if necessary
        xx = xx if len(x.shape) == 2 else tf.keras.layers.Reshape(-1)(xx)
        return xx

    def action(self, x, return_score=False, eps=.1):
        """
        Choose action in epsilon greedy fashion
        :param x:
        :param eps: the probaility of selecting a random action
        :return: an action, or one per element of the batch
        """
        values = self.call(x)
        u = np.random.random()
        if u < eps:
            values = np.random.random(values.shape)
            return tf.argmax(values, axis=1 if len(x.shape) == 2 else 0)
        else:
            return tf.argmax(values, axis=1 if len(x.shape) == 2 else 0) if not return_score else (tf.argmax(values, axis=1 if len(x.shape) == 2 else 0), values)

    def q_values(self, s, a):
        q = self.call(s)
        q = tf.gather(q, tf.stack([a]*3, 1))
        return tf.reshape(q, -1)


def build_target(dqn, dqn_eval, r, s_, d, gamma):
    """Returns r + gamma * max_a { Q_eval(s_t+1,a) } """
    dqn.eval()
    dqn_eval.eval()

    target = tf.reshape(r, -1).float()

    q = dqn.call(s_)
    greedy_actions = tf.reshape(tf.argmax(q + tf.random_normal(q.shape) / 1e6, axis=1), (-1, 1))   # actions that maximizes `Q(S_t+1, a)` w.r.t. `a`
    update_target = tf.reshape(dqn_eval.q_values(s_, greedy_actions), -1)  # values of these actions `a` with the evaluation network `Q'(S_t+1, a)`
    update_target = (gamma * update_target[1 - d]).float()

    target[1 - d] += update_target  # update only those transitions that are not done
    target = tf.stop_gradient(target)  # don't propagate gradients through this

    return target


###########################################
# TESTED
###########################################

def update_eval_network(dqn_eval, dqn, i, tau):
    """Copy the weights of the policy network into the target network"""
    if i % tau == 0:
        dqn_eval.set_weights(dqn.get_weights())


def create_exp_dir():
    """Create a subdirectory within ./tensorboard to store the experiments with TensorBoard"""
    i = 1
    while True:
        if os.path.isdir('./tensorboard/exp%d'%i):
            i += 1
        else:
            break
    os.mkdir('./tensorboard/exp%d'%i)
    return 'tensorboard/exp%d'%i


def huber(x):
    """Huber function (~ smooth L1 norm of x)"""
    mask = (tf.abs(x) < 1.).float()
    return mask*.5*tf.pow(x, 2) + (1.-mask)*(tf.abs(x) - .5)


def color(actions):
    """
    Color 0 in red (push left), 1 in yellow (do nothing), and 2 in green (push right)
    The object of this function is to visualize the policy in the (position, speed) state space (to actually see whatthe policy looks like)
    """
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
    """
    Instead of using the sparse reward signal given by the environment Mountaincar-v0 (the episodes' rewards are always the same until you reach the flag once),
     I use this signal that rewards getting closer to the flag, and penalize being to far, in particular staying in the middle
    """
    # reward = _reward(pos) - _reward(-.3)
    reward = -1
    if done:
        if pos >= .5:
            reward += 1  #50  # you want something big enough so that it is not worth chilling around on the rest of the hill to reap more rewards
            successes += 1
    return reward, successes


def tensorboard(dqn, pos_speed_grid, writer, t, cum_reward, successes, t_ep, loss, wloss, r):
    """Monitoring using TensorBoard"""
    dqn.eval()
    actions, q_scores = dqn.action(tf.Tensor(pos_speed_grid).float(), eps=0, return_score=True)
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
