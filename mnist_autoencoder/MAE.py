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


class Autoencoder(nn.Module):

    def __init__(self, width, height, bottleneck):
        self.width = width
        self.height = height
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.hidden_size = (self.width - 4 * 3) * (self.height - 4 * 3) * 8
        self.lin1 = nn.Linear(self.hidden_size, bottleneck)
        self.lin2 = nn.Linear(bottleneck, self.hidden_size)
        self.conv4 = nn.ConvTranspose2d(8, 4, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(4, 2, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(2, 1, kernel_size=5)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, self.hidden_size)
        x = self.lin1(x)
        x = F.relu(x, True)
        x = self.lin2(x)
        x = F.relu(x, True)
        x = x.view(batch_size, 8, self.width - 4 * 3, self.height - 4 * 3)
        x = self.conv4(x)
        x = F.relu(x, True)
        x = self.conv5(x)
        x = F.relu(x, True)
        x = self.conv6(x)
        x = F.relu(x, True)
        # x = torch.sigmoid(x)
        return x


# Defining Parameters

EPOCHS = 1000
batch_size = 128
model = Autoencoder(28, 28, 32).to(DEVICE)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(trainset), position=1)

outer_bar.set_description("Epochs")
for epoch in range(EPOCHS):
    inner_bar.reset()
    for data in dataloader:
        img, _ = data
        img = img.to(DEVICE)
        # ===================forward=====================
        output = model(img)
        loss = distance(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        inner_bar.update(BATCH_SIZE)
        inner_bar.set_description("Avg loss %.2f" % (loss.item() / BATCH_SIZE))
    outer_bar.update(1)
    bimshow(next(iter(dataloader))[0])
