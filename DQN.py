import tensorflow as tf
import random
import math
import numpy as np
import gym
import scipy as sc


env = gym.make('Breakout-v0')
NUM_ACTIONS = env.action_space.n


# Hyperparameters

num_iter = 1400000
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 50000
gamma = 0.99
#learning_rate =  0.0002


# Convert to Grayscale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



# Deep Q Network


# Forward Pass

state_inp = tf.placeholder(shape=[1,336,84,1], dtype=tf.float32)
action_taken = tf.placeholder(shape=[1,NUM_ACTIONS], dtype=tf.float32)

conv1 = tf.layers.conv2d(state_inp, 16, [8,8], [4,4], activation=tf.nn.relu) 
conv2 = tf.layers.conv2d(conv1, 32, [4,4], [2,2], activation=tf.nn.relu) 
conv2_flat = tf.reshape(conv2, [-1, 40*9*32])
dense = tf.layers.dense(conv2_flat, 256, activation=tf.nn.relu)
output = tf.layers.dense(dense, NUM_ACTIONS)
Q_pred = tf.reduce_sum(tf.multiply(output, action_taken), axis=1)


# Gradient Descent

Y_j = tf.placeholder(shape=None, dtype=tf.float32)

loss = tf.square(Y_j - Q_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, epsilon=0.0000008).minimize(loss)


class ReplayBuffer:

	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []

	
	def push(self, old_state, action, reward, next_state, done):
	
		self.buffer.append((old_state, action, reward, next_state, done))
		if(len(self.buffer)>self.capacity):
			self.buffer.pop(0)


	def sample(self):
		return random.sample(self.buffer, 1)[0]





def preprocess(state):

	state_tmp = rgb2gray(state)
	state_tmp = sc.misc.imresize(state_tmp, [110,84])
	state_tmp = state_tmp[13:97,:]
	state_tmp = np.reshape(state_tmp, [-1, 84, 84, 1])

	return state_tmp




#global NUM_ACTIONS
#print(env.action_space.n)

# maintain queue of last 4 frames
frame_queue = []
old_state = np.zeros((1,336,84,1))

episode_reward = 0
reward_file = open('rewards.txt', 'w')


def initialize():
	
	env.reset()
	frame_queue = []
	
	for i in range(4):
		a = env.action_space.sample()
		state, reward, done, _ = env.step(a)
		state_tmp = preprocess(state)

		frame_queue.append(state_tmp)


	return frame_queue
	

done = True

replay_buffer = ReplayBuffer(100000)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())


	for i in range(num_iter):


		if(done==True):
			print("DONE")
			print("DONE")
			print("DONE")
			frame_queue = initialize()
			reward_file.write(str(episode_reward))
			reward_file.write("\n")
			episode_reward = 0


		#
		# Stack 4 frames 
		#

		old_state = np.concatenate(frame_queue, axis=1)


		# epsilon greedy policy for action selection

		epsilon_i = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

		epsilon = epsilon_i(i)	

		rand = random.random()
		if(rand>epsilon):

			_output = sess.run(output, feed_dict = {state_inp:old_state})

			#print(_output.shape)
			#print()
			a = np.argmax(_output)
		else:
			a = env.action_space.sample()	


		print(a)
		print()

		state, reward, done, _ = env.step(a)

		episode_reward += reward


		#
		# Preprocess: convert to grayscale, downsample and crop
		# Final input shape of size 84 X 84
		#

		state_tmp = preprocess(state)


		frame_queue.append(state_tmp)
		frame_queue.pop(0)

		final_state = np.concatenate(frame_queue, axis=1)


		replay_buffer.push(old_state, a, reward, final_state, done)


		# Update Q Function
		
		old_state_buff, action_buff, reward_buff, state_buff, done_buff = replay_buffer.sample()
		_output_state_buff = sess.run(output, feed_dict = {state_inp:state_buff})

		target = reward_buff + (gamma * np.max(_output_state_buff) * (1 - done))
		action_buff_vec = np.zeros((1,NUM_ACTIONS))
		action_buff_vec[0][action_buff] = 1

		sess.run(output,feed_dict = {state_inp:old_state_buff, action_taken:action_buff_vec, Y_j:target})


		#print(final_state.shape)
