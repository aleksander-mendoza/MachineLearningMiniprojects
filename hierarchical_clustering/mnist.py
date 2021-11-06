import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from matplotlib.gridspec import GridSpec
from sklearn.cluster import AgglomerativeClustering
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

if not os.path.isfile('../mnist_autoencoder/vae.pth'):
    print("Run mnist_autoencoder/VAE.py to generate vae.pth first")
    exit()

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
SAMPLE_COUNT = 128
transform = transforms.Compose([transforms.ToTensor()])
trainset = tv.datasets.MNIST(root='../mnist_autoencoder', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=SAMPLE_COUNT, shuffle=True, num_workers=0)


class VariationalAutoencoder(nn.Module):

    def __init__(self, width, height, bottleneck):
        self.width = width
        self.height = height
        super(VariationalAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.hidden_size = (self.width - 4 * 3) * (self.height - 4 * 3) * 8
        self.lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, bottleneck)
        self.log_var = nn.Linear(self.hidden_size, bottleneck)
        self.lin2 = nn.Linear(bottleneck, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.conv4 = nn.ConvTranspose2d(8, 4, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(4, 2, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(2, 1, kernel_size=5)

    def encode(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, self.hidden_size)
        x = x + self.lin1(x)
        x = F.relu(x, True)
        return x

    def decode(self, x):
        batch_size = x.size()[0]
        x = self.mu(x)
        x = self.lin2(x)
        x = F.relu(x, True)
        x = x + self.lin3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, 8, self.width - 4 * 3, self.height - 4 * 3)
        x = self.conv4(x)
        x = F.relu(x, True)
        x = self.conv5(x)
        x = F.relu(x, True)
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


torch.set_grad_enabled(False)
model = VariationalAutoencoder(28, 28, 4).to(DEVICE)
model.load_state_dict(torch.load('../mnist_autoencoder/vae.pth'))
samples = next(iter(dataloader))
embedded = model.encode(samples[0].to(DEVICE)).cpu()
embed2d = TSNE(n_components=2).fit_transform(embedded)
fig = plt.figure(constrained_layout=True)
gs = GridSpec(1, 2, figure=fig)
ax_scatter = fig.add_subplot(gs[0, 0])
ax_img = fig.add_subplot(gs[0, 1])
cluster = AgglomerativeClustering(compute_full_tree=True, n_clusters=10, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(embed2d)
sc = ax_scatter.scatter(embed2d[:, 0], embed2d[:, 1], c=cluster_labels)
for i, (label, position) in enumerate(zip(samples[1], embed2d)):
    ax_scatter.annotate(str(label.item()), position)


def click(event):
    if event.inaxes == ax_scatter and event.button is not None:
        cont, ind = sc.contains(event)
        if cont:
            ind = ind["ind"][0]
            sample = samples[0][ind]
            variational_vec = embedded[ind].to(DEVICE)
            variational_vec = variational_vec.unsqueeze(0)
            decoded = model.decode(variational_vec).cpu()
            sample = torch.cat((decoded.squeeze(0).squeeze(0), sample.view(-1, 28)))
            ax_img.imshow(sample, cmap='gray')
            fig.canvas.draw()


fig.canvas.mpl_connect("button_press_event", click)
plt.show()

