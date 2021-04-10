# https://arxiv.org/pdf/2001.11692.pdf


import math
import unicodedata
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
import random
import os
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance

DATA_FILE = 'en_US.txt'
EPOCHS = 4000
TEACHER_FORCING_PROBABILITY = 0.4
LEARNING_RATE = 0.01
BATCH_SIZE = 512
plt.ion()
if not os.path.isfile(DATA_FILE):
    import requests

    open(DATA_FILE, 'wb').write(
        requests.get('https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/' + DATA_FILE).content)

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

OUT_LOOKUP = ['', 'b', 'a', 'ʊ', 't', 'k', 'ə', 'z', 'ɔ', 'ɹ', 's', 'j', 'u', 'm', 'f', 'ɪ', 'o', 'ɡ', 'ɛ', 'n',
              'e', 'd',
              'ɫ', 'w', 'i', 'p', 'ɑ', 'ɝ', 'θ', 'v', 'h', 'æ', 'ŋ', 'ʃ', 'ʒ', 'ð']

IN_LOOKUP = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']

IN_ALPHABET = {letter: idx for idx, letter in enumerate(IN_LOOKUP)}

OUT_ALPHABET = {letter: idx for idx, letter in enumerate(OUT_LOOKUP)}

TOTAL_OUT_LEN = 0

DATA: [(torch.tensor, torch.tensor)] = []

MAX_LEN = 32

with open(DATA_FILE) as f:
    for line in f:
        text, phonemes = line.split("\t")
        phonemes = phonemes.strip().split(",")[0]
        phonemes = re.sub(r'[/\'ˈˌ]', '', phonemes)
        text = re.sub(r'[^a-z]', '', text.strip())
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


outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(DATA), position=1)


def dist(a: [str], b: [str]):
    return torch.tensor([levenshtein_distance(a[i], b[i]) for i in range(len(a))], dtype=torch.float, device=DEVICE)


def train_model(model):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr=LEARNING_RATE)
    loss_snapshots = []
    outer_bar.reset()
    outer_bar.set_description("Epochs")
    data_loader = DataLoader(dataset=DATA, drop_last=True,
                             batch_size=3 * BATCH_SIZE,
                             collate_fn=collate,
                             shuffle=True)
    for epoch in range(EPOCHS):
        total_loss = 0
        inner_bar.reset()

        for batch_text, batch_phonemes in data_loader:
            optimizer.zero_grad()
            anchor, positive, negative = batch_text.to(DEVICE).split(BATCH_SIZE)
            ph_anchor = batch_phonemes[:BATCH_SIZE]
            ph_positive = batch_phonemes[BATCH_SIZE:2 * BATCH_SIZE]
            ph_negative = batch_phonemes[2 * BATCH_SIZE:]
            embedded_anchor = model(anchor)
            embedded_positive = model(positive)
            embedded_negative = model(negative)
            estimated_pos_dist = torch.linalg.norm(embedded_anchor - embedded_positive, dim=1)
            estimated_neg_dist = torch.linalg.norm(embedded_anchor - embedded_negative, dim=1)
            estimated_pos_neg_dist = torch.linalg.norm(embedded_positive - embedded_negative, dim=1)
            actual_pos_dist = dist(ph_anchor, ph_positive)
            actual_neg_dist = dist(ph_anchor, ph_negative)
            actual_pos_neg_dist = dist(ph_positive, ph_negative)
            loss = sum(abs(estimated_neg_dist - actual_neg_dist)
                       + abs(estimated_pos_dist - actual_pos_dist)
                       + abs(estimated_pos_neg_dist - actual_pos_neg_dist)
                       + (estimated_pos_dist - estimated_neg_dist - (actual_pos_dist - actual_neg_dist)).clip(min=0))
            loss.backward()
            optimizer.step()
            inner_bar.update(3 * BATCH_SIZE)
            loss_scalar = loss.item()
            total_loss += loss_scalar
            inner_bar.set_description("loss %.2f" % loss_scalar)
        loss_snapshots.append(total_loss/len(DATA)*3)
        plt.clf()
        plt.plot(loss_snapshots, label="Avg loss ")
        plt.legend(loc="upper left")
        plt.pause(interval=0.01)
        # print()
        # print("Total epoch loss:", total_loss)
        # print("Total epoch avg loss:", total_loss / TOTAL_TRAINING_OUT_LEN)
        # print("Training snapshots:", train_snapshots)
        # print("Training snapshots(%):", train_snapshots_percentage)
        # print("Evaluation snapshots:", eval_snapshots)
        # print("Evaluation snapshots(%):", eval_snapshots_percentage)
        outer_bar.set_description("Epochs")
        outer_bar.update(1)


train_model(CNN(kernel_size=3, hidden_layers=14, channels=MAX_LEN, embedding_size=MAX_LEN).to(DEVICE))
