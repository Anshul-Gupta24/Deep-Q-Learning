# Deep-Q-Learning

## Tensorflow implementation of the paper Playing Atari with Deep Reinforcement Learning (Mnih et al, 2013) by Deepmind.

### The architecture consists of 2 CNN layers and a dense layer.

### The procedure is as follows:

### We first preprocess the video frame to convert it to Greyscale, and then crop it to a resolution of 84 X 84
### We store histories of state, action, and reward in a replay buffer, and randomly sample trajectories to perform SGD to train our network.  
### Actions are chosen via an epsilon greedy policy.

### To run:
### >> python DQN.py

### To change game:
### Replace game name in first line,
### env = gym.make('\<game name\>')
