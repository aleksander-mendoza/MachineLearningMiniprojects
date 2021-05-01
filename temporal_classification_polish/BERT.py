# data from  https://git.wmi.amu.edu.pl/kubapok/retroc2

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline
from transformers import XLMTokenizer, RobertaModel
import torch
import os
import matplotlib.pyplot as plt
from transformers.tokenization_utils_base import TruncationStrategy

if not os.path.isdir('retroc2'):
    print('First run\ngit clone https://git.wmi.amu.edu.pl/kubapok/retroc2')
    exit()

tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

data_dir = 'retroc2/train'
top = torch.nn.Linear(768, 1)
dataset = []
with open(data_dir + '/train.tsv') as t:
    for data in t:
        data = data.strip().split('\t')
        year = data[0]
        text = data[4]
        dataset.append((year, text))


def collate(b):
    sentences = [sentence for sentence, _ in b]
    years = torch.tensor([year for _, year in b])
    years = years.unsqueeze(1)
    sentences = tokenizer.encode(sentences, return_tensors='pt')
    return sentences, years


trainloader = torch.utils.data.DataLoader(dataset, collate=collate, batch_size=4, shuffle=True)
optimizer = torch.optim.Adam(top.parameters())
criterion = torch.nn.MSELoss()
losses = []
for epoch in range(2):
    total_loss = 0
    for sentences, years in trainloader:
        outputs = model(sentences)
        predicted = top(outputs)
        loss = criterion(outputs, predicted)
        total_loss += loss.item()
        top.zero_grad()
        loss.backward()
        top.step()
    losses.append(total_loss)
    plt.clf()
    plt.plot(losses)
    plt.pause(interval=0.001)
