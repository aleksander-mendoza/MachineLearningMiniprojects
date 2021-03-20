import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import numpy as np
import re
import random
import math
import time

device = torch.device('cpu')

out_alphabet = {'': 0, 'b': 1, 'a': 2, 'ʊ': 3, 't': 4, 'k': 5, 'ə': 6, 'z': 7, 'ɔ': 8, 'ɹ': 9, 's': 10, 'j': 11,
                'u': 12,
                'm': 13, 'f': 14, 'ɪ': 15, 'o': 16, 'ɡ': 17, 'ɛ': 18, 'n': 19, 'e': 20, 'd': 21, 'ɫ': 22, 'w': 23,
                'i': 24,
                'p': 25, 'ɑ': 26, 'ɝ': 27, 'θ': 28, 'v': 29, 'h': 30, 'æ': 31, 'ŋ': 32, 'ʃ': 33, 'ʒ': 34, 'ð': 35,
                '^': 36, '$': 37}
out_lookup = ['', 'b', 'a', 'ʊ', 't', 'k', 'ə', 'z', 'ɔ', 'ɹ', 's', 'j', 'u', 'm', 'f', 'ɪ', 'o', 'ɡ', 'ɛ', 'n',
              'e', 'd',
              'ɫ', 'w', 'i', 'p', 'ɑ', 'ɝ', 'θ', 'v', 'h', 'æ', 'ŋ', 'ʃ', 'ʒ', 'ð', '^', '$']

in_alphabet = {'': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
               'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
               'x': 24, 'y': 25, 'z': 26, '$': 27, '^': 28}
in_lookup = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z', '$', '^']


class PhoneticDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file) as f:
            for line in f:
                text, phonemes = line.strip().split("\t")
                phonemes = phonemes.split(",")[0]
                phonemes = '^' + re.sub(r'[/\'ˈˌ]', '', phonemes) + '$'
                text = '^' + re.sub(r'[^a-z]', '', text) + '$'
                text = torch.tensor([in_alphabet[letter] for letter in text], dtype=torch.int)
                phonemes = torch.tensor([out_alphabet[letter] for letter in phonemes], dtype=torch.int)
                self.data.append((text, phonemes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch: [(torch.tensor, torch.tensor)]):
    batch.sort(reverse=True, key=lambda x: len(x[0]))
    in_lengths = [len(entry[0]) for entry in batch]
    print(in_lengths)
    max_in_len = max(in_lengths)
    out_lengths = [len(entry[1]) for entry in batch]
    max_out_len = max(out_lengths)
    padded_in = torch.zeros((len(batch), max_in_len), dtype=torch.int)
    padded_out = torch.zeros((len(batch), max_out_len), dtype=torch.long)
    for i in range(0, len(batch)):
        padded_in[i, :len(batch[i][0])] = batch[i][0]
        padded_out[i, :len(batch[i][1])] = batch[i][1]
    return padded_in, in_lengths, padded_out, out_lengths


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=len(in_alphabet),
                                      embedding_dim=hidden_size,
                                      padding_idx=in_alphabet[''])
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)

    def forward(self, padded_in, in_lengths):
        batch_size = len(in_lengths)
        hidden = self.init_hidden(batch_size)
        # print("hidden=", hidden.size())
        # print("padded_in=", padded_in.shape)
        # print("in_lengths=", in_lengths)
        embedded = self.embedding(padded_in)
        # print("embedded=", embedded.size())
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, in_lengths, batch_first=True)
        # print("packed=", packed.data.size())
        gru_out, hidden = self.gru(packed, hidden)
        # print("gru_out=", gru_out.data.size())
        # print("hidden=", hidden.size())
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # print("unpacked=", unpacked.size())
        return unpacked, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=len(out_alphabet),
                                      embedding_dim=hidden_size,
                                      padding_idx=out_alphabet[''])
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, len(out_alphabet))
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, padded_out, hidden):
        padded_out = padded_out.unsqueeze(1)
        embedded = self.embedding(padded_out)
        gru_out, hidden = self.gru(embedded, hidden)
        lin = self.out(gru_out)
        probabilities = self.softmax(lin)
        return probabilities, hidden


def train(epochs, teacher_forcing_probability, learning_rate=0.01,batch_size=8):
    encoder = EncoderRNN(3)
    decoder = DecoderRNN(3)
    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()),
                                  lr=learning_rate)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
                                  lr=learning_rate)
    criterion = nn.NLLLoss()
    # https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt
    dataset = PhoneticDataset('en_US.txt')
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate)
    for epoch in range(epochs):
        for batch_in, i_lengths, batch_out, o_length in data_loader:
            loss = 0
            encoder_output, hidden = encoder.forward(batch_in, i_lengths)
            output = batch_out[:, 0]
            for i in range(len(batch_out) - 1):
                if random.random() < teacher_forcing_probability:
                    out = batch_out[:, i].unsqueeze(1)
                else:
                    out = output
                output, hidden = decoder.forward(out, hidden)
                loss += criterion(output.squeeze(1), batch_out[:, i + 1])
                output = torch.argmax(output, 2)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print('epochs: ' + str(epoch))
            print('total loss: ' + str(loss))
            print()


train(1, 0)
