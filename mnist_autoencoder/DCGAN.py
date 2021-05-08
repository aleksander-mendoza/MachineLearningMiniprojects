# With help from
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision as tv
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 128

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

    def __init__(self):
        super(Generator, self).__init__()
        #
        self.conv1 = nn.ConvTranspose2d(latent_channels, 32, kernel_size=4)
        self.conv2 = nn.ConvTranspose2d(32, 32, kernel_size=5)
        self.conv3 = nn.ConvTranspose2d(32, 32, kernel_size=5)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(16, 8, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(8, 4, kernel_size=5)
        self.conv7 = nn.ConvTranspose2d(4, img_channels, kernel_size=5)
        #

    def forward(self, x):
        x = F.relu(self.conv1(x), True)
        x = F.relu(self.conv2(x), True)
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv4(x), True)
        x = F.relu(self.conv5(x), True)
        x = F.relu(self.conv6(x), True)
        x = torch.tanh(self.conv7(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5)
        self.lin = nn.Linear(32 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, True)  # Leaky ReLU makes a huge difference!
        x = self.conv2(x)  # If you use just a regular ReLU, the discriminator
        x = F.leaky_relu(x, True)  # won't propagate gradient well enough and as a result
        x = self.conv3(x)  # generator won't know how to improve itself. The results of training with
        x = F.leaky_relu(x, True)  # ReLU are just black images with nothing on them
        x = self.conv4(x)
        x = F.leaky_relu(x, True)
        x = self.conv5(x)
        x = F.leaky_relu(x, True)
        x = self.conv6(x)
        x = F.leaky_relu(x, True)
        x = x.reshape(bs, -1)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x


G = Generator().to(device)
D = Discriminator().to(device)
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
        generated = generated.view(-1, 28)
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
