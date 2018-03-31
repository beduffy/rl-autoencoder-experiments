import time
import sys

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import gym

from vae_atari import ConvVAE

model_fp = 'conv-vae-epoch-50-maybe-best.pkl'

model = ConvVAE()
model.load_state_dict(torch.load(model_fp))

print(model)

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((110, 84), Image.NEAREST),  # 84 in DQN paper
        lambda x: np.array(x)[-84:, :],
        transforms.ToPILImage(),  # can do so much better than this
        transforms.Resize(64, Image.NEAREST),  # todo could also just adapt architecture to deal with 84x84
        transforms.ToTensor()
    ])

open_ai_env_name = 'SpaceInvaders-v0'
env = gym.make(open_ai_env_name)

observation = env.reset()

# print(observation)
print(observation.shape)

def run_model(observation):

    input_image = transform(observation).view(1, 64, 64, 3)

    input_image = Variable(input_image, volatile=True)
    reconstructed_image, mu, logvar = model(input_image)

    plt.imshow(transform(observation).permute(1, 2, 0))
    plt.show()
    plt.imshow(reconstructed_image.data[0].permute(1, 2, 0))
    plt.show()

for i_episode in range(10):
    observation = env.reset()


    for t in range(10000):
        env.render()

        action = env.action_space.sample()
        action = 4

        observation, reward, done, info = env.step(action)

        if t > 300:
            run_model(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
