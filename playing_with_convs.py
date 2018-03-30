import sys
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def try_vae_flow():
    layer1_en = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # nn.MaxPool2d(2)
            )
    layer2_en = nn.Sequential(
        nn.Conv2d(16, 36, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(36),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )
    fc_en_1 = nn.Linear(21 * 21 * 36, 36)
    fc_en_2 = nn.Linear(21 * 21 * 36, 36)

    layer1_de = nn.ConvTranspose2d(36, 16, kernel_size=5, stride=2)
    layer2_de = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)

    input = torch.autograd.Variable(torch.randn(16, 84, 84, 3))

    #mu, logvar = self.encode(x)
    x = input.permute(0, 3, 1, 2)

    out = layer1_en(x)
    out = layer2_en(out)
    out = out.view(out.size(0), -1)

    mu, logvar = fc_en_1(out), fc_en_2(out)


    #z = self.reparameterize(mu, logvar)
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    z = eps.mul(std).add_(mu)

    #return self.decode(z), mu, logvar
    z = z.view(z.size(0), 36, 1, 1)

    # h3 = self.relu(self.layer1_de(z, output_size=(53, 40)))
    h3 = layer1_de(z, output_size=(5, 5))
    h4 = layer2_de(h3, output_size=(13, 13))
    h5 = layer2_de(h4, output_size=(30, 30))
    h6 = layer2_de(h5, output_size=(20, 20))
    h6 = layer2_de(h6, output_size=(20, 20))

# try_vae_flow()
# sys.exit()

# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)


# exact output size can be also specified as an argument
input = torch.autograd.Variable(torch.randn(1, 16, 12, 12))
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
print(h.size())
#h.Size([1, 16, 6, 6])
output = upsample(h, output_size=input.size())
print(output.size())
#torch.Size([1, 16, 12, 12])

print('Beginning my test')
upsample = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2)
upsample = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=3)
input = torch.autograd.Variable(torch.randn(1, 16, 1, 1))
print(input.size())
h = upsample(input, (1, 1))
print(h.size())
