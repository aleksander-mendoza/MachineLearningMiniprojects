# With help from
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# If you get 503 while downloading MNIST then download it manually
# wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# tar -zxvf MNIST.tar.gz
BATCH_SIZE = 32

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='.', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def imshow(inp):
    inp = inp.cpu().detach().numpy()
    mean = 0.1307
    std = 0.3081
    inp = ((mean * inp) + std)
    plt.clf()
    plt.imshow(inp, cmap='gray')
    plt.pause(interval=0.01)


def bimshow(batch):
    with torch.no_grad():
        output = model(batch.to(DEVICE)).cpu()
        imshow(torch.cat((batch.view(-1, 28), output.view(-1, 28)), 1))


width = 28
height = 28
latent_width = 1
latent_height = 1
latent_channels = 64
generator_hidden_channels = 1
img_channels = 1


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        #
        self.conv1 = nn.ConvTranspose2d(latent_channels, generator_hidden_channels * 8, kernel_size=4)
        self.conv2 = nn.ConvTranspose2d(generator_hidden_channels * 8, generator_hidden_channels * 8, kernel_size=5)
        self.conv3 = nn.ConvTranspose2d(generator_hidden_channels * 8, generator_hidden_channels * 8, kernel_size=5)
        self.conv4 = nn.ConvTranspose2d(generator_hidden_channels * 8, generator_hidden_channels * 4, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(generator_hidden_channels * 4, generator_hidden_channels * 2, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(generator_hidden_channels * 2, generator_hidden_channels, kernel_size=5)
        self.conv7 = nn.ConvTranspose2d(generator_hidden_channels, img_channels, kernel_size=5)
        #

    def forward(self, x):
        x = F.relu(self.conv1(x), True)
        x = F.relu(self.conv2(x), True)
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv4(x), True)
        x = F.relu(self.conv5(x), True)
        x = F.relu(self.conv6(x), True)
        x = F.relu(self.conv7(x), True)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        # x = torch.sigmoid(x)
        return x


# Defining Parameters

EPOCHS = 1000
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
distance = nn.MSELoss()
optimizerG = torch.optim.Adam(G.parameters())
optimizerD = torch.optim.Adam(D.parameters())
outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(trainset), position=1)
outer_bar.set_description("Epochs")
for epoch in range(EPOCHS):
    inner_bar.reset()
    for data in dataloader:
        real_img, _ = data
        real_img = real_img.to(DEVICE)
        loss_real = - torch.log(D(real_img))
        loss_real.backward()
        optimizerD.step()

        mean_loss_real = loss_real.mean().item()

        latent = torch.rand(BATCH_SIZE, latent_channels, latent_width, latent_height, device=DEVICE)
        fake_img = G(latent)
        loss_fake = torch.log(1 - D(fake_img.detach()))

        loss = distance(output, img)
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()

        inner_bar.update(BATCH_SIZE)
        inner_bar.set_description("Avg loss %.2f" % (loss.item() / BATCH_SIZE))
    outer_bar.update(1)
    bimshow(next(iter(dataloader))[0])
