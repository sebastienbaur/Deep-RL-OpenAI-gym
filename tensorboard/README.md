# Instructions to read the files

You need to:
* install TensorBoard,
* in a command line, go to this folder, and run `tensorboard --logdir=tensorboard`
* Finally, open your browser to the URL `http://localhost:6006` to see the experiments

# Descriptions of experiments:

* exp 1: tau = 200 instead of 10000, dense rewards
* exp 2: RMSProp with low momentum instead of Adam
* exp 3: gamma .99 instead of .999
* exp 4: tau = 10
* exp 5: ?
* exp 6: more reward on reaching the flag (+50 instead of +3)
* exp 7: original reward scheme

# Conclusion

The introduction of DDQN with a reasonable update frequency (once or twice every episode), with an optimizer (little momentum) is the most important factor to
find an optimal policy quickly.

The GAMMA is also quite important (you want the long term rewards to be very important, so GAMMA ~ 1)


# Final hyperparameters

```
BATCH_SIZE = 64
TAU = 100  # the update frequency of the target network
GAMMA = .999  # discount factor
N_RANDOM = 40000.  # the number of steps it takes to go from eps=1. to eps=0.05 (exponentially decreasing, loses 99% of the .95 in 40000 steps)
N_EPISODES = 1500
MAX_SIZE_BUFFER = 20000  # the maximum number of transitions in the buffer
T_START_LEARNING = BATCH_SIZE * 10  # the minimum number of transitions to be seen in the buffer before training
ALPHA = .6  # default OpenAI
BETA = .4  # default OpenAI
PER_EPS = 1e-6
RMSProp, lr 2e-4, momentum 5e-2
original reward scheme
```