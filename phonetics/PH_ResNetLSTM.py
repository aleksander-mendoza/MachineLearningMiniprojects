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
EPOCHS = 4000
TEACHER_FORCING_PROBABILITY = 0.4
LEARNING_RATE = 0.01
BATCH_SIZE = 1024
plt.ion()
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


IN_ALPHABET = {letter: idx for idx, letter in enumerate(IN_LOOKUP)}

OUT_ALPHABET = {letter: idx for idx, letter in enumerate(OUT_LOOKUP)}

TOTAL_OUT_LEN = 0

DATA: [(torch.tensor, torch.tensor)] = []

with open(DATA_FILE) as f:
    for line in f:
        text, phonemes = line.split("\t")
        phonemes = phonemes.strip().split(",")[0]
        phonemes = '^' + re.sub(r'[/\'ˈˌ]', '', phonemes) + '$'
        text = '^' + re.sub(r'[^a-z]', '', text.strip()) + '$'
        text = torch.tensor([IN_ALPHABET[letter] for letter in text], dtype=torch.int)
        phonemes = torch.tensor([OUT_ALPHABET[letter] for letter in phonemes], dtype=torch.int)
        DATA.append((text, phonemes))
random.shuffle(DATA)
# DATA = DATA[:2000]
print("Number of samples ", len(DATA))
TRAINING_SET_SIZE = int(len(DATA) * 0.5)
TRAINING_SET_SIZE -= TRAINING_SET_SIZE % BATCH_SIZE
EVAL = DATA[TRAINING_SET_SIZE:]
DATA = DATA[:TRAINING_SET_SIZE]
assert len(DATA) % BATCH_SIZE == 0
print("Training samples ", len(DATA))
print("Evaluation samples ", len(EVAL))
print("Input alphabet ", IN_LOOKUP)
print("Output alphabet ", OUT_LOOKUP)
TOTAL_TRAINING_OUT_LEN = 0
TOTAL_EVALUATION_OUT_LEN = 0
for text, phonemes in DATA:
    TOTAL_TRAINING_OUT_LEN += len(phonemes)
for text, phonemes in EVAL:
    TOTAL_EVALUATION_OUT_LEN += len(phonemes)
TOTAL_EVALUATION_OUT_LEN -= len(EVAL)  # do not count the beginning of line ^ character
TOTAL_TRAINING_OUT_LEN -= len(DATA)
print("Total output length in training set", TOTAL_TRAINING_OUT_LEN)
print("Total output length in evaluation set", TOTAL_EVALUATION_OUT_LEN)

def shuffle_but_keep_sorted_by_output_lengths(data: [(torch.tensor, torch.tensor)]):
    random.shuffle(data)
    data.sort(reverse=True, key=lambda x: len(x[1]))  # sort with respect to output lengths


def collate(batch: [(torch.tensor, torch.tensor)]):
    batch.sort(reverse=True, key=lambda x: len(x[0]))  # sort with respect to input lengths
    max_in_len = max(map(lambda entry: len(entry[0]), batch))
    out_lengths = [len(entry[1]) for entry in batch]
    max_out_len = max(out_lengths)
    padded_in = torch.zeros((len(batch), max_in_len), dtype=torch.int)
    padded_out = torch.zeros((len(batch), max_out_len), dtype=torch.long)
    for i in range(0, len(batch)):
        padded_in[i, :len(batch[i][0])] = batch[i][0]
        padded_out[i, :len(batch[i][1])] = batch[i][1]
    rightmost = len(batch)
    in_lengths = []
    for i in range(max_in_len):
        while padded_in[rightmost - 1, i] == 0:
            rightmost -= 1
        in_lengths.append(rightmost)
    return padded_in, in_lengths, padded_out, out_lengths


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = layers
        self.embedding = nn.Embedding(num_embeddings=len(IN_ALPHABET),
                                      embedding_dim=hidden_size,
                                      padding_idx=IN_ALPHABET[''])
        self.gru = nn.ModuleList([nn.LSTMCell(input_size=hidden_size,
                                              hidden_size=hidden_size) for _ in range(self.hidden_layers)])
        # self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, padded_in, in_lengths):
        batch_size = padded_in.size()[0]
        seq_size = padded_in.size()[1]
        x = self.embedding(padded_in)
        hidden_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=DEVICE)
        cell_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=DEVICE)
        assert x.size() == (batch_size, seq_size, self.hidden_size)
        for symbol, batch_size_per_symbol in enumerate(in_lengths):
            y = x[:, symbol, :]
            hidden_state_new = hidden_state.clone()
            cell_state_new = cell_state.clone()
            for layer, gru in enumerate(self.gru):
                hidden_state1 = hidden_state[layer, :batch_size_per_symbol, :]
                cell_state1 = cell_state[layer, :batch_size_per_symbol, :]
                (hidden_state2, cell_state2) = gru(y[:batch_size_per_symbol], (hidden_state1, cell_state1))
                hidden_state_new[layer, :batch_size_per_symbol, :] = hidden_state2
                cell_state_new[layer, :batch_size_per_symbol, :] = cell_state2
                y = hidden_state2 + hidden_state1
            hidden_state = hidden_state_new
            cell_state = cell_state_new
        assert x.size() == (batch_size, seq_size, self.hidden_size)
        return hidden_state[-1]


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = layers
        self.embedding = nn.Embedding(num_embeddings=len(OUT_ALPHABET),
                                      embedding_dim=hidden_size,
                                      padding_idx=OUT_ALPHABET[''])
        self.gru = nn.ModuleList([nn.LSTMCell(input_size=hidden_size,
                                              hidden_size=hidden_size) for _ in range(self.hidden_layers)])
        self.out = nn.Linear(hidden_size, len(OUT_ALPHABET))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, padded_out, hidden):
        batch_size = len(padded_out)

        hidden_state, cell_state = hidden
        assert hidden_state.size() == (self.hidden_layers, batch_size, self.hidden_size)
        assert cell_state.size() == (self.hidden_layers, batch_size, self.hidden_size)
        padded_out = padded_out.unsqueeze(1)
        embedded = self.embedding(padded_out)
        embedded = embedded.squeeze(1)
        assert embedded.size() == (batch_size, self.hidden_size)
        y = embedded
        hidden_state_new = torch.zeros_like(hidden_state)
        cell_state_new = torch.zeros_like(cell_state)
        for layer, gru in enumerate(self.gru):
            (hidden_state2, cell_state2) = gru(y, (hidden_state[layer], cell_state[layer]))
            hidden_state_new[layer] = hidden_state2
            cell_state_new[layer] = cell_state2
            y = hidden_state2+hidden_state[layer]
        # assert hidden.size() == (self.hidden_layers, batch_size, self.hidden_size)
        assert y.size() == (batch_size,  self.hidden_size)
        lin = self.out(y)
        assert lin.size() == (batch_size,  len(OUT_ALPHABET))
        probabilities = self.softmax(lin)
        assert probabilities.size() == (batch_size,  len(OUT_ALPHABET))
        return probabilities, (hidden_state_new, cell_state_new)

    def init_hiddden(self, hidden):
        batch_size = hidden.size()[0]
        hidden_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=DEVICE)
        cell_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=DEVICE)
        hidden_state[0] = hidden
        return hidden_state, cell_state


def run(encoder, decoder, batch_in, i_lengths, batch_out, o_lengths, teacher_forcing_prob, criterion):
    batch_in = batch_in.to(DEVICE)
    batch_out = batch_out.to(DEVICE)
    out_seq_len = batch_out.size()[1]
    in_seq_len = batch_in.size()[1]
    assert batch_in.size() == (BATCH_SIZE, in_seq_len)
    assert batch_out.size() == (BATCH_SIZE, out_seq_len)
    loss = 0
    total_correct_predictions = 0
    encoder_output = encoder(batch_in, i_lengths)
    output = batch_out[:, 0]
    hidden = decoder.init_hiddden(encoder_output)
    for i in range(out_seq_len - 1):
        if random.random() < teacher_forcing_prob:
            out = batch_out[:, i]
        else:
            out = output
        output, hidden = decoder(out, hidden)
        output = output.squeeze(1)
        expected_output = batch_out[:, i + 1]
        if criterion is not None:
            loss += criterion(output, expected_output)
        argmax_output = torch.argmax(output, 1)
        with torch.no_grad():
            total_correct_predictions += (argmax_output == expected_output).sum().item()
        output = argmax_output
    return loss, total_correct_predictions


eval_bar = tqdm(total=len(EVAL), position=2)


def eval_model(encoder, decoder):
    eval_bar.reset()
    eval_bar.set_description("Evaluation")
    with torch.no_grad():
        total_correct_predictions = 0
        for batch_in, i_lengths, batch_out, o_lengths in DataLoader(dataset=EVAL, drop_last=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    collate_fn=collate):
            loss, correct_predictions = run(encoder=encoder,
                                            decoder=decoder,
                                            criterion=None,
                                            i_lengths=i_lengths,
                                            o_lengths=o_lengths,
                                            batch_in=batch_in,
                                            batch_out=batch_out,
                                            teacher_forcing_prob=0)
            total_correct_predictions += correct_predictions
            eval_bar.update(BATCH_SIZE)
        return total_correct_predictions


outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(DATA), position=1)


def train_model(encoder, decoder):
    if os.path.isfile('encoder.pt'):
        encoder.load_state_dict(torch.load('encoder.pt'))
    if os.path.isfile('decoder.pt'):
        decoder.load_state_dict(torch.load('decoder.pt'))
    encoder_optimizer = optim.Adam(filter(lambda x: x.requires_grad, encoder.parameters()),
                                   lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()),
                                   lr=LEARNING_RATE)
    criterion = nn.NLLLoss(ignore_index=OUT_ALPHABET[''])
    train_snapshots_percentage = [0]
    train_snapshots = [0]
    eval_snapshots = [eval_model(encoder, decoder)]
    eval_snapshots_percentage = [eval_snapshots[0] / TOTAL_EVALUATION_OUT_LEN]
    best_model_so_far = eval_snapshots[0]
    outer_bar.reset()
    outer_bar.set_description("Epochs")
    for epoch in range(EPOCHS):
        shuffle_but_keep_sorted_by_output_lengths(DATA)
        total_loss = 0
        total_correct_predictions = 0
        inner_bar.reset()
        for batch_in, i_lengths, batch_out, o_lengths in DataLoader(dataset=DATA, drop_last=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    collate_fn=collate):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss, correct_predictions = run(encoder=encoder,
                                            decoder=decoder,
                                            criterion=criterion,
                                            i_lengths=i_lengths,
                                            o_lengths=o_lengths,
                                            batch_in=batch_in,
                                            batch_out=batch_out,
                                            teacher_forcing_prob=TEACHER_FORCING_PROBABILITY)
            total_correct_predictions += correct_predictions
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            inner_bar.update(BATCH_SIZE)
            loss_scalar = loss.item()
            total_loss += loss_scalar
            inner_bar.set_description("Avg loss %.2f" % (loss_scalar / batch_out.size()[1]))
        train_snapshots.append(total_correct_predictions)
        train_snapshots_percentage.append(total_correct_predictions / TOTAL_TRAINING_OUT_LEN)
        new_score = eval_model(encoder, decoder)
        eval_snapshots.append(new_score)
        eval_snapshots_percentage.append(new_score / TOTAL_EVALUATION_OUT_LEN)
        if new_score > best_model_so_far:
            best_model_so_far = new_score
            torch.save(encoder.state_dict(), 'encoder.pt')
            torch.save(decoder.state_dict(), 'decoder.pt')
        plt.clf()
        plt.plot(train_snapshots_percentage, label="Training %")
        plt.plot(eval_snapshots_percentage, label="Evaluation %")
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


train_model(EncoderRNN(512, 2).to(DEVICE), DecoderRNN(512, 2).to(DEVICE))
