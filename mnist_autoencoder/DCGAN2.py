# With help from
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and architecture from
# https://github.com/AKASHKADEL/dcgan-mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision as tv
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 1024

train_dataset = tv.datasets.MNIST(root='.', train=True, download=True)  # , transform=transform)
with torch.no_grad():
    DATA = train_dataset.data.float()
    # Pixels have integer values between 0 and 255
    DATA = DATA / 255
    # Now pixels have real values between 0 and 1
    DATA = (DATA - 0.5) * 2
    # Now pixels have real values between -1 and 1
    # And this is crucial for learning! If the pixels take
    # different set of values everything fails miserably
    DATA = DATA.unsqueeze(3)  # one channel for grayscale pixels
dataloader = torch.utils.data.DataLoader(DATA, batch_size=bs, shuffle=True, drop_last=True)

mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
hidden_size = 256

width = 28
height = 28
latent_width = 1
latent_height = 1
latent_channels = 64
img_channels = 1


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # It seems that somehow, having stride==2 is crucial for learning
            # You can take a look at DCGAN1.py, which uses stride==1
            # and it fails to generate any digits
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            # Leaky ReLU makes a huge difference!
            # If you use just a regular ReLU, the discriminator
            # won't propagate gradient well enough and as a result
            # generator won't know how to improve itself. The results of training with
            # ReLU are just black images with nothing on them
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.reshape(bs, 1)


G = Generator(img_channels, latent_channels, 32).to(device)
D = Discriminator(img_channels, 32).to(device)
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

n_epoch = 2000
D_losses, G_losses = [], []
real_scores, fake_scores = [], []
for epoch in range(1, n_epoch + 1):
    D_loss_total = 0
    G_loss_total = 0
    fake_score_total = 0
    real_score_total = 0
    for batch_idx, images in enumerate(dataloader):
        # =======================Train the discriminator=======================#

        # train discriminator on real
        images = Variable(images.to(device))
        real_labels = Variable(torch.ones(bs, 1).to(device))
        fake_labels = Variable(torch.zeros(bs, 1).to(device))
        # images.size() = (bs, width, height, channels)
        images = images.transpose(1, 3)
        # images.size() = (bs, channels, height, width)
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # train discriminator on fake
        z = Variable(torch.randn(bs, latent_channels, 1, 1).to(device))
        fake_images = G(z)

        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # gradient backprop & optimize ONLY D's parameters
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        G.zero_grad()
        d_loss.backward()
        D_optimizer.step()

        D_loss_total += d_loss.item()
        real_score_total += real_score.sum().item()
        fake_score_total += fake_score.sum().item()

        # =======================Train the generator=======================#

        z = Variable(torch.randn(bs, latent_channels, 1, 1).to(device))

        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        # gradient backprop & optimize ONLY G's parameters
        G.zero_grad()
        D.zero_grad()
        g_loss.backward()
        G_optimizer.step()

        G_loss_total += g_loss.item()
    fake_score_total /= len(DATA)
    real_score_total /= len(DATA)
    D_loss_total /= len(DATA)
    D_loss_total /= len(DATA)
    D_losses.append(D_loss_total)
    G_losses.append(G_loss_total)
    real_scores.append(real_score_total)
    fake_scores.append(fake_score_total)
    with torch.no_grad():
        test_bach_size = 4
        test_z = Variable(torch.randn(test_bach_size, latent_channels, 1, 1).to(device))
        generated = G(test_z)
        # generated.size() == (batch, channels, height, width)
        generated = generated.transpose(1, 3)
        # generated.size() == (batch, width, height, channels)
        generated = generated.squeeze(3)
        # generated.size() == (batch, width, height)
        generated = generated.reshape(-1, 28)
        generated = generated.detach()
        generated = ((generated / 2) + 0.5)
        generated = generated.cpu().numpy()
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(G_losses, label="generator loss")
        plt.plot(D_losses, label="discriminator loss")
        plt.legend(loc="upper left")
        plt.subplot(1, 3, 2)
        plt.plot(real_scores, label="real score")
        plt.plot(fake_scores, label="fake score")
        plt.legend(loc="upper left")
        plt.subplot(1, 3, 3)
        plt.imshow(generated, cmap='gray')
        plt.pause(interval=0.01)
    # print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
    #     (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
