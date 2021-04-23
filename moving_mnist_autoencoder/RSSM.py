import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torchvision.transforms import transforms
from tqdm import tqdm
from torch.distributions.normal import Normal

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class RSSM(nn.Module):

    def __init__(self, width, height, bottleneck):
        super(RSSM, self).__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.hidden_size = (self.width - 4 * 3) * (self.height - 4 * 3) * 8
        self.observation_to_latent = nn.Linear(self.hidden_size, bottleneck*2)
        self.latent_to_latent1 = nn.Linear(bottleneck, bottleneck)
        self.latent_to_latent2 = nn.Linear(bottleneck, bottleneck * 2)
        self.latent_to_observation = nn.Linear(bottleneck, self.hidden_size)
        self.conv4 = nn.ConvTranspose2d(8, 4, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(4, 2, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(2, 1, kernel_size=5)
        self.latent_criterion = nn.MSELoss()

    def forward(self, x, epsilon):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        batch_size = x.size()[0]

        x = x.transpose(0, 1)  # shape (t, b, c, h, w)        #   time, batch, channel, height, width
        output = torch.empty_like(x)
        latent = None
        latent_loss = 0
        for time_step, frame in enumerate(x):  # frame shape    (batch, channel, height, width)
            frame = F.relu(self.conv1(frame), True)
            frame = F.relu(self.conv2(frame), True)
            frame = F.relu(self.conv3(frame), True)
            frame = frame.view(batch_size, self.hidden_size)
            posterior_latent = self.observation_to_latent(frame)
            posterior_mean, posterior_log_variance = torch.chunk(posterior_latent, 2, dim=1)
            posterior_standard_deviation = torch.exp(0.5*posterior_log_variance)
            eps = torch.randn_like(posterior_standard_deviation)  # `randn_like` as we need the same size
            if time_step < 2:
                latent = posterior_mean + (eps * posterior_standard_deviation)  # sampling
            else:
                prior_latent = self.latent_to_latent1(latent)
                prior_latent = self.latent_to_latent2(prior_latent)
                prior_mean, prior_log_variance = torch.chunk(prior_latent, 2, dim=1)
                prior_standard_deviation = torch.exp(0.5 * prior_log_variance)
                kld = kl_divergence(Normal(posterior_mean, posterior_standard_deviation), Normal(prior_mean, prior_standard_deviation))
                latent_loss = latent_loss + kld.sum()
                if epsilon < random.random():  # teacher forcing
                    latent = prior_mean + (eps * prior_standard_deviation)  # sampling
                else:
                    latent = posterior_mean + (eps * posterior_standard_deviation)  # sampling
            frame = F.relu(self.latent_to_observation(latent), True)
            frame = frame.view(batch_size, 8, self.width - 4 * 3, self.height - 4 * 3)
            frame = F.relu(self.conv4(frame), True)
            frame = F.relu(self.conv5(frame), True)
            frame = F.relu(self.conv6(frame), True)
            # frame = torch.sigmoid(frame)
            output[time_step] = frame
        output = output.transpose(0, 1)
        return output, latent_loss


MEAN = 12.6026
STD = 51.1410
URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
FILE = "mnist_test_seq.npy"
FILE_NORMALIZED = "mnist_test_seq.pt"
if not os.path.isfile(FILE_NORMALIZED):
    if not os.path.isfile(FILE):
        import requests

        print("Downloading data")
        open(FILE, 'wb').write(requests.get(URL).content)
    print("Loading data")
    data = np.load(FILE)  # (frames, batches, width, height)
    data = data.swapaxes(0, 1)  # (batches, frames, width, height)
    data = data[:128]  # the dataset is too large and a smaller subset should be fine
    print("Normalizing data")
    data = torch.from_numpy(data)
    data = data.type(torch.float)
    MEAN = data.mean()
    STD = data.std()
    print("mean=", MEAN, "std=", STD)
    data = (data-MEAN)/STD
    data = data.unsqueeze(2)  # number of channels is 1
    print("Saving")
    torch.save(data, FILE_NORMALIZED)
else:
    print("Loading preprocessed data")
    data = torch.load(FILE_NORMALIZED)

print("Running")

BATCH_SIZE = 4

loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

model = RSSM(64, 64, 64).to(DEVICE)

criterion = nn.MSELoss()

optim = torch.optim.Adam(model.parameters())
losses = []

batch_bar = tqdm(total=data.size()[0], position=2, desc="samples")


for epoch in tqdm(range(1024), position=1, desc="epoch"):
    total_loss = 0
    batch_bar.reset()
    for batch in loader:
        batch = batch.to(DEVICE)
        y_hat, loss = model(batch, 0.5 if epoch < 100 else 0)
        loss = 0.005*loss + criterion(y_hat, batch)
        total_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        batch_bar.update(BATCH_SIZE)
    losses.append(total_loss)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="total loss")
    plt.legend(loc="upper left")
    # plt.pause(interval=0.01)
    plt.subplot(1, 2, 2)
    batch = torch.cat((batch[0].view(-1, 64), y_hat[0].view(-1, 64)), 1)
    batch = ((MEAN * batch) + STD)
    batch = batch.cpu().detach().numpy()
    plt.imshow(batch, cmap='gray')
    plt.pause(interval=0.01)
