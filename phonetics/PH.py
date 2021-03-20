import math
import unicodedata
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
import random
import os
from tqdm import tqdm

DATA_FILE = 'en_US.txt'
EPOCHS = 200
TEACHER_FORCING_PROBABILITY = 0.4
LEARNING_RATE = 0.01
BATCH_SIZE = 32

if not os.path.isfile(DATA_FILE):
    import requests

    open(DATA_FILE, 'wb').write(
        requests.get('https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/' + DATA_FILE).content)

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

OUT_LOOKUP = ['', 'b', 'a', 'ʊ', 't', 'k', 'ə', 'z', 'ɔ', 'ɹ', 's', 'j', 'u', 'm', 'f', 'ɪ', 'o', 'ɡ', 'ɛ', 'n',
              'e', 'd',
              'ɫ', 'w', 'i', 'p', 'ɑ', 'ɝ', 'θ', 'v', 'h', 'æ', 'ŋ', 'ʃ', 'ʒ', 'ð', '^', '$']

IN_LOOKUP = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z', '$', '^']


def extract_alphabet():
    """
    This function has been used to extract the alphabet but then it was hard-coded directly into
    code in order to speed up loading time
    """
    with open(DATA_FILE) as f:
        in_alph = set()
        out_alph = set()
        for line in f:
            text, phonemes = line.strip().split("\t")
            phonemes = phonemes.split(",")[0]
            for letter in phonemes:
                out_alph.add(letter)
                in_alph.add(letter)
        return in_alph, out_alph


IN_ALPHABET = {letter: idx for idx, letter in enumerate(IN_LOOKUP)}

OUT_ALPHABET = {letter: idx for idx, letter in enumerate(OUT_LOOKUP)}

TOTAL_OUT_LEN = 0

DATA: [(torch.tensor, torch.tensor)] = []

with open(DATA_FILE) as f:
    for line in f:
        text, phonemes = line.strip().split("\t")
        phonemes = phonemes.split(",")[0]
        phonemes = '^' + re.sub(r'[/\'ˈˌ]', '', phonemes) + '$'
        text = '^' + re.sub(r'[^a-z]', '', text) + '$'
        text = torch.tensor([IN_ALPHABET[letter] for letter in text], dtype=torch.int)
        phonemes = torch.tensor([OUT_ALPHABET[letter] for letter in phonemes], dtype=torch.int)
        DATA.append((text, phonemes))
        TOTAL_OUT_LEN += len(phonemes)

print("Number of samples ", len(DATA))
random.shuffle(DATA)
TRAINING_SET_SIZE = int(len(DATA) * 0.8)
TRAINING_SET_SIZE -= TRAINING_SET_SIZE % BATCH_SIZE
EVAL = DATA[TRAINING_SET_SIZE:]
DATA = DATA[:TRAINING_SET_SIZE]
assert len(DATA) % BATCH_SIZE == 0
print("Training samples ", len(DATA))
print("Evaluation samples ", len(EVAL))


def shuffle_but_keep_sorted_by_output_lengths(data: [(torch.tensor, torch.tensor)]):
    random.shuffle(data)
    data.sort(reverse=True, key=lambda x: len(x[1]))  # sort with respect to output lengths


def collate(batch: [(torch.tensor, torch.tensor)]):
    batch.sort(reverse=True, key=lambda x: len(x[0]))  # sort with respect to input lengths
    in_lengths = [len(entry[0]) for entry in batch]
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
        self.hidden_layers = 1
        self.embedding = nn.Embedding(num_embeddings=len(IN_ALPHABET),
                                      embedding_dim=hidden_size,
                                      padding_idx=IN_ALPHABET[''])
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=self.hidden_layers,
                          batch_first=True)

    def forward(self, padded_in, in_lengths, hidden):
        batch_size = len(in_lengths)
        assert hidden.size() == (self.hidden_layers, batch_size, self.hidden_size)
        embedded = self.embedding(padded_in)
        assert embedded.size() == (batch_size, max(in_lengths), self.hidden_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, in_lengths, batch_first=True)
        gru_out, hidden = self.gru(packed, hidden)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        assert unpacked.size() == (batch_size, max(in_lengths), self.hidden_size)
        assert hidden.size() == (self.hidden_layers, batch_size, self.hidden_size)
        return unpacked, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = 1
        self.embedding = nn.Embedding(num_embeddings=len(OUT_ALPHABET),
                                      embedding_dim=hidden_size,
                                      padding_idx=OUT_ALPHABET[''])
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=self.hidden_layers,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, len(OUT_ALPHABET))
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, padded_out, hidden):
        batch_size = len(padded_out)
        padded_out = padded_out.unsqueeze(1)
        seq_length = 1
        assert hidden.size() == (self.hidden_layers, batch_size, self.hidden_size)
        embedded = self.embedding(padded_out)
        assert embedded.size() == (batch_size, self.hidden_layers, self.hidden_size)
        gru_out, hidden = self.gru(embedded, hidden)
        assert hidden.size() == (self.hidden_layers, batch_size, self.hidden_size)
        assert gru_out.size() == (batch_size, seq_length, self.hidden_size)
        lin = self.out(gru_out)
        assert lin.size() == (batch_size, seq_length, len(OUT_ALPHABET))
        probabilities = self.softmax(lin)
        assert probabilities.size() == (batch_size, seq_length, len(OUT_ALPHABET))
        return probabilities, hidden


def run(encoder, decoder, batch_in, i_lengths, batch_out, o_lengths, teacher_forcing_prob, criterion):
    batch_in = batch_in.to(DEVICE)
    batch_out = batch_out.to(DEVICE)
    out_seq_len = batch_out.size()[1]
    in_seq_len = batch_in.size()[1]
    assert batch_in.size() == (BATCH_SIZE, in_seq_len)
    assert batch_out.size() == (BATCH_SIZE, out_seq_len)
    loss = 0
    hidden = encoder.init_hidden(BATCH_SIZE, device=DEVICE)
    encoder_output, hidden = encoder(batch_in, i_lengths, hidden)
    output = batch_out[:, 0]
    for i in range(out_seq_len - 1):
        if random.random() < teacher_forcing_prob:
            out = batch_out[:, i]
        else:
            out = output
        output, hidden = decoder(out, hidden)
        output = output.squeeze(1)
        loss += criterion(output, batch_out[:, i + 1])
        output = torch.argmax(output, 1)
    return loss


eval_bar = tqdm(total=len(EVAL), position=2)


def eval_model(encoder, decoder):
    eval_bar.reset()
    eval_bar.set_description("Evaluation")
    with torch.no_grad():
        loss = 0
        for batch_in, i_lengths, batch_out, o_lengths in DataLoader(dataset=EVAL, drop_last=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    collate_fn=collate):
            loss += run(encoder=encoder,
                        decoder=decoder,
                        criterion=lambda x, y: (torch.argmax(x, 1) == y).sum(),
                        i_lengths=i_lengths,
                        o_lengths=o_lengths,
                        batch_in=batch_in,
                        batch_out=batch_out,
                        teacher_forcing_prob=0).item()
            eval_bar.update(BATCH_SIZE)
        return loss


outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(DATA), position=1)


def train_model(encoder, decoder):
    encoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, encoder.parameters()),
                                  lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(filter(lambda x: x.requires_grad, decoder.parameters()),
                                  lr=LEARNING_RATE)
    criterion = nn.NLLLoss(ignore_index=OUT_ALPHABET[''])
    total_losses = []
    eval_snapshots = [eval_model(encoder, decoder)]
    eval_snapshots_percentage = [eval_snapshots[0] / TOTAL_OUT_LEN]
    outer_bar.reset()

    outer_bar.set_description("Epochs")
    for epoch in range(EPOCHS):
        shuffle_but_keep_sorted_by_output_lengths(DATA)
        total_loss = 0
        inner_bar.reset()
        for batch_in, i_lengths, batch_out, o_lengths in DataLoader(dataset=DATA, drop_last=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    collate_fn=collate):
            loss = run(encoder=encoder,
                       decoder=decoder,
                       criterion=criterion,
                       i_lengths=i_lengths,
                       o_lengths=o_lengths,
                       batch_in=batch_in,
                       batch_out=batch_out,
                       teacher_forcing_prob=TEACHER_FORCING_PROBABILITY)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            inner_bar.update(BATCH_SIZE)
            loss_scalar = loss.item()
            total_loss += loss_scalar
            inner_bar.set_description("Avg loss %.2f" % (loss_scalar / batch_out.size()[1]))
        total_losses.append(total_loss / TOTAL_OUT_LEN)
        eval_snapshots.append(eval_model(encoder, decoder))
        eval_snapshots_percentage.append(eval_snapshots[-1] / TOTAL_OUT_LEN)
        outer_bar.set_description("Epochs (loss %.2f)" % total_losses[-1])
        outer_bar.update(1)
        plt.plot(total_losses)
        plt.plot(eval_snapshots)
        plt.plot(eval_snapshots_percentage)
        plt.pause(interval=0.01)
        print("Evaluation snapshots:", eval_snapshots)
        print("Evaluation snapshots(%):", eval_snapshots_percentage)


plt.show()

train_model(EncoderRNN(3).to(DEVICE), DecoderRNN(3).to(DEVICE))
