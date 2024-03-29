﻿# MountainCar

The goal of this project was to try and solve the MountainCar Environmental from OpenAI Gym.

The hard-coded policy achieves this goal by giving the car specific instructions to move forwards or backwards based on the cars position and velocity on the hills.

The Neural Network Policy was a bit more tricky.  This is because I had to create training data for the model to learn from.  I had the environment go through a bunch of test runs using random actions, and if the cars position was greater than -0.5 (which is about halfway up the right hill) in the run, then a reward is given and the that run's data is added to the training data.  This ensures that the model is training off of good (rewarded) data.  Once the training data is gathered, it is then manipulated and shuffled so that it can be fed into the model.  

The neural netowork model in this code uses a cross entropy loss function.  Ideally, I would have liked the model to be trained on a higher number of epochs and batches; however, since I am running this model off of my CPU, it would have taken far too long (if it even would have been able to) for my computer to train on.  After the model trains off the training data, it takes in an assigned X value (X being the position and velocity of the Car) and determines the action probabilities probabilities if it should go push left [1, 0, 0], stay in neutral [0, 1, 0], or push right[0, 0, 1].  
