import time

import numpy as np
import matplotlib.pyplot as plt

import sys

import gym

# save image

def show_image(obs):
    imgplot = plt.imshow(obs)
    plt.show()

number_of_saved_images = 0

# observation (210, 160, 3)

env = gym.make('SpaceInvaders-v0')
for i_episode in range(10):
    observation = env.reset()


    for t in range(10000):
        env.render()

        fp = 'saved_images/saved_image_{}.png'.format(number_of_saved_images)
        np.save(fp, observation)
        number_of_saved_images += 1

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
