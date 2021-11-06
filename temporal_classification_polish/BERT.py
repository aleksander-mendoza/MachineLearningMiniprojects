# data from  https://git.wmi.amu.edu.pl/kubapok/retroc2

from transformers import BertForSequenceClassification, BertTokenizer, BertModel
from transformers import pipeline
from transformers import XLMTokenizer, RobertaModel, AutoModel, AutoTokenizer
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers.tokenization_utils_base import TruncationStrategy

if not os.path.isdir('retroc2'):
    print('First run\ngit clone https://git.wmi.amu.edu.pl/kubapok/retroc2')
    exit()

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
allegro = False
if allegro:
    tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")
else:
    model = BertModel.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")

max_len = 512
out_len = 768
model = model.to(DEVICE)
data_dir = 'retroc2/train'
top = torch.nn.Linear(out_len, 1).to(DEVICE)
dataset = []
with open(data_dir + '/train.tsv') as t:
    for data in t:
        data = data.strip().split('\t')
        year = float(data[0])
        text = data[4]
        dataset.append([text, year])

year_tensor = torch.tensor(list(map(lambda x:x[1], dataset)))
year_std, year_mean = torch.std_mean(year_tensor)
year_std = year_std.item()
year_mean = year_mean.item()
year_variance = year_mean*year_mean
for text_year in dataset:
    text_year[1] = (text_year[1]-year_mean)/year_std
del year_tensor

def collate(b):
    sentences = [sentence for sentence, _ in b]
    years = torch.tensor([year for _, year in b])
    years = years.unsqueeze(1)
    encoded = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    return encoded, years


EPOCHS = 2
epoch_bar = tqdm(total=EPOCHS, position=1)
batch_bar = tqdm(total=len(dataset), position=2)
BATCH=4
trainloader = torch.utils.data.DataLoader(dataset, collate_fn=collate, batch_size=BATCH, shuffle=True)
optimizer = torch.optim.Adam(top.parameters())
criterion = torch.nn.MSELoss()
losses = []
for epoch in range(EPOCHS):
    total_mse = 0
    batch_bar.reset()
    for sentences, years in trainloader:
        years = years.to(DEVICE)
        input_ids = sentences['input_ids'].to(DEVICE)
        attention_mask = sentences['attention_mask'].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        predicted = top(outputs.pooler_output)
        loss = criterion(years, predicted)
        # mse == (years*year_std+year_mean - (predicted*year_std+year_mean))^2 / BATCH
        # mse == (years*year_std - predicted*year_std)^2 / BATCH
        # mse == (years - predicted)^2*year_std^2 / BATCH
        # mse == (years - predicted)^2 / BATCH *year_variance
        # mse == loss * year_variance
        total_mse = loss.item()*year_variance
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_bar.set_description("mse "+str(total_mse))
        batch_bar.update(BATCH)
    total_mse /= len(dataset)/BATCH
    losses.append(total_mse)
    plt.clf()
    plt.plot(losses)
    plt.pause(interval=0.001)
    epoch_bar.update(1)
