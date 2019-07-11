from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.learn as learn

import gym
import random
import numpy as np

######################################################
# STEP ONE: CREATE YOUR TRAINING DATA
######################################################

env = gym.make('MountainCar-v0')
env.reset()

goal_steps = 200
score_requirement = -198
initial_games = 20

training_data = []
accepted_scores = []
for game_index in range(initial_games):
    score = 0
    game_memory = []
    previous_observation = []

    print("Game Index: {}".format(game_index))

    for step_index in range(goal_steps):

        action = random.randrange(0, 3)
        observation, reward, done, info = env.step(action)

        if len(previous_observation) > 0:
            game_memory.append([previous_observation, action])

        previous_observation = observation
        if observation[0] > -0.5:
            reward = 1
            # print("Positive result achieved!")

        score += reward
        if done:
            break

    if score >= score_requirement:
        accepted_scores.append(score)
        for data in game_memory:
            if data[1] == 1:
                output = [0, 1, 0]
            elif data[1] == 0:
                output = [1, 0, 0]
            elif data[1] == 2:
                output = [0, 0, 1]
            training_data.append([data[0], output])

    env.reset()

print("######Accepted Scores########")
print(accepted_scores)

# print("######Training_Data########")
# print(training_data)

########################################
# STEP 2: MANIPULATING THE TRAINING DATA
########################################

# SHuFFLE THE TRAINING DATA
random.shuffle(training_data)
# print("Shuffled Training Data: ")
# print(training_data)

y_labels = []
pos_list = []
vel_list = []

for items in training_data:
    y_labels.append(items[1])
    pos_list.append(items[0][0])
    vel_list.append(items[0][1])

# print('=' * 100)
#
# print(y_labels)
# print(pos_list)
# print(vel_list)

min_pos = min(pos_list)
max_pos = max(pos_list)

min_vel = min(vel_list)
max_vel = max(vel_list)

# print('=' * 100)

normailized_pos = []
for items in pos_list:
    temp_list = []
    items = (items - min_pos)/(max_pos - min_pos)
    temp_list.append(items)
    normailized_pos.append(temp_list)

normailized_vel = []
for items in vel_list:
    temp_list = []
    items = (items - min_vel)/(max_vel - min_vel)
    temp_list.append(items)
    normailized_vel.append(temp_list)

# print(normailized_pos)
# print(normailized_vel)

x_vals = []
for i in range(0, len(normailized_pos)):
    x_vals.append(normailized_pos[i] + normailized_vel[i])

# print('=' * 100)
# print(x_vals)
# print(y_labels)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


print('=' * 100)
########################################
# STEP 3: BUILD THE NEURAL NETWORK
########################################


# PLACEHOLDERS
X = tf.placeholder(tf.float32, shape=[None, 2])
y_true = tf.placeholder(tf.float32, [None, 3])

# VARIABLES
W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))

# CREATE GRAPH OPERATIONS
z = tf.matmul(X, W) + b

# using a cross entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=z))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # Play with Learning Rate

# TRAIN
train = optimizer.minimize(cross_entropy)

# INITIALIZE AND SET SESSION VALUES
init = tf.global_variables_initializer()

"""IDEALLY MORE EPOCHS AND BATCHES; HOWEVER, RUNNING ON CPU NOT GPU, SO MORE
EPOCHS AND BATCHES WOULD TAKE A VERY LONG TIME. """
num_epochs = 10
num_batches = 200
# num_batches = len(y_labels)

# RUN SESSION
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
        for batch in range(num_batches):

            sess.run(train, feed_dict={X:x_vals, y_true:y_labels})

    W = sess.run(W)
    b = sess.run(b)

    print('W shape =', W.shape)  # (2, 3)
    print('b shape =', b.shape)  # (3,)

    z_val = np.matmul(x_vals[4], W) + b  # y = tf.matmul(X, W) + b

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    z_probabilities = softmax(z_val)

    print('z_probabilities: {}'.format(z_probabilities))
    print('Sum of probabilities: {}'.format(sum(z_probabilities)))
    print('y_labels[#]: {}'.format(y_labels[4]))
    print('=' * 40)
