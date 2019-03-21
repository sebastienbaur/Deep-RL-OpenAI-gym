# Deep-RL-OpenAI-gym
Deep Reinforcement learning on the OpenAI gym environment

* Implementation of DDQN : https://arxiv.org/abs/1509.06461
* Works with the OpenAI gym environment (MountainCar-v0)
* Use OpenAI PER code (https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)
* Trying different choices of hyperparameters


# Hyperparameters that work well:
* `BATCH_SIZE = 64`
* `TAU = 100  # the update frequency of the target network`
* `GAMMA = .999  # discount factor`
* `N_RANDOM = 40000.  # the number of steps it takes to go from eps=1. to eps=0.05 (exponentially decreasing, loses 99% of the .95 in 40000 steps)`
* `N_EPISODES = 1500`
* `MAX_SIZE_BUFFER = 20000  # the maximum number of transitions in the buffer`
* `T_START_LEARNING = BATCH_SIZE * 10  # the minimum number of transitions to be seen in the buffer before training`
* `ALPHA = .6  # default OpenAI`
* `BETA = .4  # default OpenAI`
* `PER_EPS = 1e-6`
* `RMSProp, lr 2e-4, momentum 5e-2`
* original reward scheme


# Hyperparameters whose influence should be tested:
* OPTIMIZER: RMSProp with small momentum >>> Adam. Note that dimishing one of Adam's betas may have the same effect. This is because the objective is non-stationnary, so momentum hurts training
* UPDATE FREQUENCY: Reducing from 10000 to 200 made A HUGE DIFFERENCE (with 10000, there was no learning).
                    Reducing it to 100 makes the performance a bit better still (faster learning)
                    Reducing it to 10 makes it about the same
* GAMMA: .999 is better than .99.
         With the former, training is way faster. It is probably because the positive rewards are delayed in time, so having a higher GAMMA gives them more importance
         It is essential to learn a successful policy
* THE REWARD FUNCTION: it is better to have a dense reward signal, as it helps the agent towards the objective. Note that maximizing the cumulative discounted reward should still be
                       aligned with the objective (i.e. there shouldn't be any way of maximizing it without achieving the goal)
                       NOTE: IS IT REALLT THIS THAT MAKES THE AGENT LEARN SOMETHING, OR RATHER THE UPDATE FREQUENCY OF THE TARGET NETWORK
* REWARD SCHEME: The original reward scheme isnt actually sparse (-1 at each time step), but there is not any truly useful signal before the car reaches the flag.
                 I thought that this signal didn't help and modified it to add a positive signal depending on how close the car gets to the flag.
                 It turns out that the original reward scheme is better (way faster training). It is probably because when tryng an action that doesnt help reaching the flag,
                 its values is decreased (the reward was -1), so another one will be tried next time, which may lead to find the optimal solution.


In conclusion, 3 parameters are REALLY USEFUL:
- using a target network to have a more stationary training objective (DDQN vs DQN)
- the update frequency of the target (updating not too frequently (more than 1 which is the DQN case), more like every episode, like 100 or 200)
- using an optimizer with little momentum


# Other remarks
As the rewards are always negative in the normal reward scheme, it may sound sensible to constrain the output of the Q network to be always negative (by outputting -F.relu(...)) for example
As it turns out, doing this makes learning much harder. Most values tend to be zero (Q(s,a) = 0 \forall s,a), and nothing happens. The reason is rather obscure to me
