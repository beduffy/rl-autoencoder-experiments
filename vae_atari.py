from __future__ import print_function
import sys
import argparse
import glob

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
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
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

def preprocess_image(obs):
    pass

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),

            transforms.ToPILImage(),
            transforms.Resize((110, 84)),
            transforms.RandomCrop(84),
            transforms.Resize(64),
            transforms.ToTensor(),
            # normalize,
        ])

def load_all_images():
    image_npy_files = glob.glob('saved_images/saved_image*')
    # print(image_npy_files)
    print('Number of files:', len(image_npy_files))

    all_images = []

    for image_path in image_npy_files[0:10]:
        loaded_obs = np.load(image_path)
        # all_images.append(transform(Image.fromarray(loaded_obs)))
        all_images.append(transform(loaded_obs))
#
    all_images = np.concatenate(all_images).reshape(-1, 64, 64, 3)
    print(all_images.shape)

    # print(all_images[300])
    # sys.exit()
    # imgplot = plt.imshow(all_images[300])
    # plt.show()

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

        # self.layer1_en = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2)
        # )
        # self.layer2_en = nn.Sequential(
        #     nn.Conv2d(16, 36, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(36),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2)
        # )
        # self.fc_en_1 = nn.Linear(53 * 40 * 36, 36)
        # self.fc_en_2 = nn.Linear(53 * 40 * 36, 36)
        #
        # # self.layer1_de = nn.Sequential(
        # #     nn.ConvTranspose2d(36, 16, kernel_size=5, stride=2, padding=2, output_size=(53, 40)),
        # #     nn.BatchNorm2d(16),
        # #     nn.ReLU(),
        # #     # nn.MaxPool2d(2)
        # # )
        # # self.layer2_de = nn.Sequential(
        # #     nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_size=torch.Size(105, 90)),
        # #     nn.BatchNorm2d(3),
        # #     nn.ReLU(),
        # #     # nn.MaxPool2d(2, padding=1)
        # # )
        #
        # self.layer1_de = nn.ConvTranspose2d(36, 16, kernel_size=5, stride=2, padding=2)
        # self.layer2_de = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2)

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

        self.fc_to_mu = nn.Linear(2 * 2 * 256, 32)
        self.fc_to_sigma = nn.Linear(2 * 2 * 256, 32)

        self.fc_from_z = nn.Linear(32, 1024)

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

        # out = self.layer1_en(x)
        # out = self.layer2_en(out)
        # out = out.view(out.size(0), -1)

        out = self.layer1_en(x)
        out = self.layer2_en(out)
        out = self.layer3_en(out)
        out = self.layer4_en(out)

        out = out.view(out.size(0), 2, 2, 256)

        return self.fc_en_1(out), self.fc_en_2(out)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(z.size(0), 36, 1, 1)
        # h3 = self.layer1_de(z)
        # h4 = self.layer2_de(h3)

        # h3 = self.relu(self.layer1_de(z, output_size=(53, 40)))
        h3 = self.relu(self.layer1_de(z, output_size=(2, 2)))
        h4 = self.relu(self.layer2_de(h3, output_size=(10, 10)))
        # h4 = self.layer2_de(h3)

        return self.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
#model = ConvVAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 210 * 160 * 3), size_average=False)

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
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 3, 210, 160),
               'atari_results/sample_' + str(epoch) + '.png')
