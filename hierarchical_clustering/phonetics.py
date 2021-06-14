import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from matplotlib.gridspec import GridSpec
from sklearn.cluster import AgglomerativeClustering
import re
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

if not os.path.isfile('../phonetics/cnn.pth'):
    print("Run phonetics/CNN.py to generate cnn.pth first")
    exit()

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

OUT_LOOKUP = ['', 'b', 'a', 'ʊ', 't', 'k', 'ə', 'z', 'ɔ', 'ɹ', 's', 'j', 'u', 'm', 'f', 'ɪ', 'o', 'ɡ', 'ɛ', 'n',
              'e', 'd',
              'ɫ', 'w', 'i', 'p', 'ɑ', 'ɝ', 'θ', 'v', 'h', 'æ', 'ŋ', 'ʃ', 'ʒ', 'ð']

IN_LOOKUP = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']

IN_ALPHABET = {letter: idx for idx, letter in enumerate(IN_LOOKUP)}
BATCH_SIZE = 128
OUT_ALPHABET = {letter: idx for idx, letter in enumerate(OUT_LOOKUP)}
TOTAL_OUT_LEN = 0
DATA: [(torch.tensor, torch.tensor)] = []
TEXT: [str] = []
MAX_LEN = 32
DATA_FILE = '../phonetics/en_US.txt'

with open(DATA_FILE) as f:
    for line in f:
        text, phonemes = line.split("\t")
        phonemes = phonemes.strip().split(",")[0]
        phonemes = re.sub(r'[/\'ˈˌ]', '', phonemes)
        text = re.sub(r'[^a-z]', '', text.strip())
        TEXT.append(text)
        assert len(text) <= MAX_LEN, text
        text = torch.tensor([IN_ALPHABET[letter] for letter in text], dtype=torch.int)
        DATA.append((text, phonemes))


def collate(batch: [(torch.tensor, str)]):
    batch_text = torch.zeros((len(batch), len(IN_ALPHABET), MAX_LEN))
    batch_phonemes = list(map(lambda x: x[1], batch))
    for i, (sample, _) in enumerate(batch):
        for chr_pos, index in enumerate(sample):
            batch_text[i, index, chr_pos] = 1
    return batch_text, batch_phonemes


class CNN(nn.Module):
    def __init__(self, kernel_size, hidden_layers, channels, embedding_size):
        super(CNN, self).__init__()
        self.input_conv = nn.Conv1d(in_channels=len(IN_ALPHABET), out_channels=channels, kernel_size=kernel_size)
        self.conv_hidden = nn.ModuleList(
            [nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size) for _ in
             range(hidden_layers)])
        self.last_layer_size = (MAX_LEN - (kernel_size - 1) * (hidden_layers + 1)) * channels
        self.lin = nn.Linear(self.last_layer_size, embedding_size)

    def forward(self, x):
        x = self.input_conv(x)
        x = F.relu(x, inplace=True)
        for c in self.conv_hidden:
            x = c(x)
            x = F.relu(x, inplace=True)
        x = x.view(x.size()[0], self.last_layer_size)
        x = self.lin(x)
        return x


torch.set_grad_enabled(False)
model = CNN(kernel_size=3, hidden_layers=14, channels=MAX_LEN, embedding_size=MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load('../phonetics/cnn.pth'))
data_loader = DataLoader(dataset=DATA, drop_last=True,
                             batch_size=3 * BATCH_SIZE,
                             collate_fn=collate,
                             shuffle=True)
samples = next(iter(data_loader))
embedded = model(samples[0].to(DEVICE)).cpu()
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

