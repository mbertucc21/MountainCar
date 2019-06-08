"""
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up
the mountain on the right; however, the car's engine is not strong enough to scale the mountain
in a single pass. Therefore, the only way to succeed is to drive back and forth to build up
momentum (see http://gym.openai.com/envs/MountainCar-v0/).

We will solve the above problem using a SIMPLE hard-coded policy:
If the car is moving forward, then have car's engine push forward (action space: 2)
If the car is moving backwards, then have the car's engine push backwards (action space:0)

As mentioned earlier, the car's engine is not strong enough to scale the mountain in a single
pass.  The engine will push the car backwards (action space 0) to get maximum momentum. Once the
momentum is at a peak, the car will start moving forward and at this point the engine will switch
so that it pushes the car forward, thus using momentum + the car's engine to reach the peak.
If for some reason the car does not reach the peak and starts moving backwards, then the car's
engine will again switch to help it move backwards to once again build momentum and try to reach
the peak again.

ADDITIONAL NOTE:  This hard coded policy ensures that the car is neutral for the first time step.
As such, the car may first move forward (to build momentum on the right) before going backwards
(to build momentum on the left).  It all depends on the randomly generated initial state.

When tested, the car should scale the right mountain in 1 or 2 attempts.

New and previous car positions and velocities, state (moving forward or backwards), observation,
reward, done and info provided for each time step.  Once Donne, the number of time steps needed
to complete the task is provided, as well as the average car position and velocity.  In addition,
graphs of (each) car position, car velocity and action space against the time steps are provided.
"""

# help("modules")

import gym  # pip install gym
import keyboard  # pip install keyboard
import tensorflow as tf  # pip install --upgrade tensorflow
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCar-v0')
# Look up in documentation what action_space and what observation_space are for your environment.

print(env.action_space)
# >> Discrete(3)
# Push car to the left (backward), neutral, or to the right (forward)
# 0 Power Back
# 1 Neutral
# 2 Forward

print(env.observation_space)
# >> Box(2,)
# Num 	Observation 	Min 	Max
# 0 	Car Position 	-1.2 	0.6
# 1 	Car Velocity 	-0.07 	0.07

print('*' * 30)

observation = env.reset()
print('INITIAL OBSERVATION:')
print(observation)  # [Car Position, Car Velocity]
print('=' * 30)

##########################
# LISTS FOR OUTPUT GRAPHS
##########################
time_steps = []
action_space = []
car_position = []
car_velocity = []

##########################
# INITIALIZATION VARIABLES
##########################
backwards = False
neutral = False
forward = False
car_pos_new = 0
car_vel_new = 0
car_pos_prev = observation[0]
car_vel_prev = observation[1]

##########################
# TIME STEP LOOP
##########################
for t in range(1000):

    env.render()

    ###########################
    # SIMPLE HARD CODE POLICY
    ###########################

    print('NEW car_pos and car_vel:{} {}'.format(car_pos_new, car_vel_new))
    print('PRE car_pos and car_vel:{} {}'.format(car_pos_prev, car_vel_prev))

    if car_pos_new >= car_pos_prev:  # car moving forward
        neutral = False
        backwards = False
        forward = True  # Go Forward
        print('GOING FORWARD')

    else:  # car is speeding up
        neutral = False
        backwards = True  # Go Backwards
        forward = False
        print('GOING BACKWARDS')

    ################
    # INITIALIZATION
    ################
    # Start off in neutral at time step 1
    if t == 0:
            neutral = True
            backwards = False
            forward = False

    ###########
    # ACTIONS
    ###########
    if backwards:
        action = 0
        action_space.append(0)

    if neutral:
        action = 1
        action_space.append(1)

    if forward:
        action = 2
        action_space.append(2)

    # Previous Observations
    car_pos_prev = observation[0]
    car_vel_prev = observation[1]

    observation, reward, done, info = env.step(action)
    car_position.append(observation[0])
    car_velocity.append(observation[1])
    time_steps.append(t)

    # New Observations
    car_pos_new = observation[0]
    car_vel_new = observation[1]

    print('Time Step: {}'.format(t))
    print('observation: {}'.format(observation))
    print('reward: {}'.format(reward))
    print('done: {}'.format(done))
    print('info: {}'.format(info))
    print('=' * 30)

    # Press q to exit render window
    if keyboard.is_pressed('q'):
        env.render(close=True)

    if done:
        print("Done after {} time steps".format(t+1))
        break

env.close()

# print(car_position)
# print(car_velocity)
# print(action_space)
print("Mean car position: {}".format(np.mean(car_position)))
print("Mean car velocity: {}".format(np.mean(car_velocity)))

plt.figure()
plt.xlabel('Time Step', fontsize=8)
plt.ylabel('Action', fontsize=8)
plt.plot(time_steps, action_space)

plt.figure()
plt.xlabel('Time Step', fontsize=8)
plt.ylabel('Car Position', fontsize=8)
plt.plot(time_steps, car_position)

plt.figure()
plt.xlabel('Time Step', fontsize=8)
plt.ylabel('Car Velocity', fontsize=8)
plt.plot(time_steps, car_velocity)

plt.show()
