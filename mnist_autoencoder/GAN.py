import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision as tv
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 1024*2

# # MNIST Dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
#
# train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
#
# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
train_dataset = tv.datasets.MNIST(root='.', train=True, download=True) #, transform=transform)
with torch.no_grad():
    DATA = train_dataset.data.float()
    # Pixels have integer values between 0 and 255
    DATA = DATA / 255
    # Now pixels have real values between 0 and 1
    DATA = (DATA - 0.5) * 2
    # Now pixels have real values between -1 and 1
    # And this is crucial for learning! If the pixels take
    # different set of values everything fails miserably
dataloader = torch.utils.data.DataLoader(DATA, batch_size=bs, shuffle=True, drop_last=True)

z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
hidden_size = 256

# Discriminator
D = nn.Sequential(
    nn.Linear(mnist_dim, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()).to(device)

# Generator
G = nn.Sequential(
    nn.Linear(z_dim, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, mnist_dim),
    nn.Tanh()).to(device)

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
        images = Variable(images.view(bs, mnist_dim).to(device))
        real_labels = Variable(torch.ones(bs, 1).to(device))
        fake_labels = Variable(torch.zeros(bs, 1).to(device))

        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # train discriminator on facke
        z = Variable(torch.randn(bs, z_dim).to(device))
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


        z = Variable(torch.randn(bs, z_dim).to(device))

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
        test_z = Variable(torch.randn(test_bach_size, z_dim).to(device))
        generated = G(test_z)
        generated = generated.view(-1, 28)
        generated = generated.detach()
        generated = ((generated/2) + 0.5)
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
