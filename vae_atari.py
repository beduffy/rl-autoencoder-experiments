from __future__ import print_function
import sys
import argparse
import glob

import cv2
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




parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((110, 80), Image.NEAREST), # 84 in DQN paper
            transforms.Resize((110, 84), Image.NEAREST), # 84 in DQN paper
            # transforms.Resize((80, 110), Image.NEAREST), # but 80 is half of 160
            # transforms.RandomCrop(84), # todo take bottom
            # transforms.Resize(64),
            # transforms.Lambda(lambda x: x[:80, :]),
            lambda x: np.array(x)[-84:, :],
            transforms.ToPILImage(), # can do so much better than this
            transforms.Resize(64, Image.NEAREST), # todo could also just adapt architecture to deal with 84x84
            transforms.ToTensor()
            # normalize,
        ])

def load_all_images():
    image_npy_files = glob.glob('saved_images/saved_image*')
    # print(image_npy_files)
    print('Number of files:', len(image_npy_files))

    all_images = []
    for idx, image_path in enumerate(image_npy_files):
        loaded_obs = np.load(image_path)
        # resized_loaded_obs = cv2.resize(loaded_obs, dsize=(110, 80), interpolation=cv2.INTER_CUBIC)
        # resized_loaded_obs = cv2.resize(loaded_obs, dsize=(80, 110), interpolation=cv2.INTER_NEAREST)

        # all_images.append(transform(Image.fromarray(loaded_obs)))
        all_images.append(transform(loaded_obs))
        # all_images.append(transform(resized_loaded_obs))

        # print(transform(loaded_obs).shape)
        # print(loaded_obs.shape)
        # plt.imshow(loaded_obs)
        # plt.show()
        # plt.imshow(transform(loaded_obs).permute(1, 2, 0))
        # plt.show()
        # sys.exit()

    all_images = np.concatenate(all_images).reshape(-1, 64, 64, 3)
    # all_images = np.concatenate(all_images).reshape(-1, 84, 84, 3)
    print(all_images.shape)

    return all_images


all_images = load_all_images()
train_loader = torch.utils.data.DataLoader(all_images, batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(210 * 160 * 3, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 210 * 160 * 3)

        # self.fc1 = nn.Linear(210 * 160 * 3, 20000)
        # self.fc1_p_2 = nn.Linear(20000, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc3_p_2 = nn.Linear(400, 20000)
        # self.fc4 = nn.Linear(20000, 210 * 160 * 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc1_p_2(h1))
        # return self.fc21(h1), self.fc22(h1)
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc3_p_2(h3))

        return self.sigmoid(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 210 * 160 * 3))
        z = self.reparameterize(mu, logvar)
        # return self.decode(z).view(-1, 210, 160, 3), mu, logvar
        return self.decode(z).view(-1, 210, 160, 3), mu, logvar

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # Created from diagram in appendix of https://worldmodels.github.io/

        self.latent_dim_size = 32

        # Convolution layers
        self.layer1_en = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.layer2_en = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.layer3_en = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.layer4_en = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc_to_mu = nn.Linear(2 * 2 * 256, self.latent_dim_size)
        self.fc_to_sigma = nn.Linear(2 * 2 * 256, self.latent_dim_size)

        self.fc_from_z = nn.Linear(self.latent_dim_size, 1024)

        # Deconvolution layers
        self.layer1_de = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.layer2_de = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.layer3_de = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
        )
        self.layer4_de = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = x.permute(0, 3, 1, 2)

        out = self.layer1_en(x)
        out = self.layer2_en(out)
        out = self.layer3_en(out)
        out = self.layer4_en(out)

        out = out.view(out.size(0), -1)

        return self.fc_to_mu(out), self.fc_to_sigma(out)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out = self.fc_from_z(z)
        out = out.view(out.size(0), -1, 1, 1)

        out = self.layer1_de(out)
        out = self.layer2_de(out)
        out = self.layer3_de(out)
        out = self.layer4_de(out)

        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# model = VAE()
model = ConvVAE()
if args.cuda:
    model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=0.008) # 0.008 got down to ~1050 # 0.0075 got to 1036
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 210 * 160 * 3), size_average=False)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        # data.view(-1, 210 * 160 * 3)
        data = Variable(data.float())
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        # todo use vizdom to visualise
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    # for i, (data, _) in enumerate(test_loader):
    for i, (data) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            data = data.permute(0, 3, 1, 2)
            # print(data[:n].shape)
            # print(recon_batch.shape)
            # print(recon_batch.view(args.batch_size, 210, 160, 3)[:n].shape)
            # # comparison = torch.cat([data[:n],
            # #                         recon_batch.view(args.batch_size, 210, 160, 3)[:n]])
            # print(data[0].shape, recon_batch[0].shape)
            # print(type(data[0]), type(recon_batch[0]))
            # print(type(data[0].data), type(recon_batch[0].data))

            #.permute(1, 2, 0)

            # comparison = torch.cat([data[0], recon_batch[0].view(3, 210, 160)])
            comparison = torch.cat([data[0], recon_batch[0]])
            # print(comparison.data.cpu())
            # print(type(comparison.data.cpu()))
            # print(type(comparison.data.cpu().numpy()))
            # print(comparison.data.cpu().numpy().shape)

            image_data = comparison.data.cpu()#.numpy()

            # save_image(image_data, 'atari_results/reconstruction_' + str(epoch) + '.png', nrow=1)

            # save_image(comparison.data.cpu().numpy(),
            #            'atari_results/reconstruction_' + str(epoch) + '.png')
                       # 'atari_results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(train_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, model.latent_dim_size))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()

    save_image(sample.data.view(64, 3, 64, 64),
               'atari_results/sample_' + str(epoch) + '.png')

    if epoch == 30:
        plt.imshow(sample.data[0].permute(1, 2, 0))
        plt.show()
