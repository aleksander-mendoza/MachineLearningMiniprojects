from logging.config import valid_ident

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from tqdm import tqdm
import minerl
import numpy as np

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
crop = transforms.RandomCrop(32, padding=4)
flip = transforms.RandomHorizontalFlip()
ORIGINAL_HEIGHT = 64
ORIGINAL_WIDTH = 64
HEIGHT = 32
WIDTH = 32
assert WIDTH == HEIGHT
assert ORIGINAL_HEIGHT == ORIGINAL_WIDTH
CHANNELS = 3
IMG_PAIRS_IN_BATCH = 4


class Head(torch.nn.Module):

    def __init__(self):
        super(Head, self).__init__()
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.lin1(x), True)
        x = self.lin2(x)
        return x


model = torchvision.models.resnet50(pretrained=True).to(DEVICE)
model.fc = nn.Linear(2048, 128)  # By default the pretrained ResnetHas 1000 classes, but we only want 128
head = Head().to(DEVICE)


def augmentations(img):
    crop_h = 16
    crop_w = 16
    i = torch.randint(HEIGHT - crop_h + 1, size=(1,)).item()
    j = torch.randint(WIDTH - crop_w + 1, size=(1,)).item()
    img = F2.crop(img, i, j, crop_h, crop_w)
    img = F2.resize(img, (HEIGHT, WIDTH))
    fn_idx = torch.randperm(4)
    for fn_id in fn_idx:
        if fn_id == 0 and torch.rand(1) < 0.2:
            brg = float(torch.empty(1).uniform_(0.6, 2))
            img = F2.adjust_brightness(img, brg)
        elif fn_id == 1 and torch.rand(1) < 0.2:
            con = float(torch.empty(1).uniform_(0.6, 2))
            img = F2.adjust_contrast(img, con)
        elif fn_id == 2 and torch.rand(1) < 0.2:
            sat = float(torch.empty(1).uniform_(0.6, 2))
            img = F2.adjust_saturation(img, sat)
        elif fn_id == 3 and torch.rand(1) < 0.2:
            hue = float(torch.empty(1).uniform_(-0.5, 0.5))
            img = F2.adjust_hue(img, hue)

    if torch.rand(1) < 0.2:
        img = F2.hflip(img)
    if torch.rand(1) < 0.2:
        img = F2.vflip(img)

    return img

t = np.load('data/MineRLNavigateDense-v0/v3_calculating_fava_bean_siren-3_155-844/rendered.npz')
optim_model = torch.optim.Adam(model.parameters(), lr=0.00001)
optim_head = torch.optim.Adam(head.parameters(), lr=0.00001)
VISUALIZE_AUGMENTATIONS = 60
criterion = torch.nn.CrossEntropyLoss()
labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
labels = labels.to(DEVICE)
mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
labels = labels[~mask].view(labels.shape[0], -1)
zeros = torch.zeros(BATCH_SIZE * 2, dtype=torch.long).to(DEVICE)
TEMPERATURE = 0.05
EPOCHS = 1000000
data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='data', num_workers=1, minimum_size_to_dequeue=2)
epoch_bar = tqdm(total=EPOCHS, position=1, desc="Epoch")
losses = []
SEQ_LEN = IMG_PAIRS_IN_BATCH*2

for epoch in range(EPOCHS):
    total_loss = 0

    if VISUALIZE_AUGMENTATIONS == 0:
        plt.clf()
        plt.plot(losses)
        plt.pause(interval=0.001)

    for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, preload_buffer_size=1,
                                                                           num_epochs=1, seq_len=SEQ_LEN):
        for video_in_batch in current_state['pov']:
            video_in_batch = torch.from_numpy(video_in_batch)
            assert video_in_batch.size() == torch.Size([SEQ_LEN, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, CHANNELS]), video_in_batch.size()
            video_in_batch = video_in_batch.transpose(1, 3)
            assert video_in_batch.size() == torch.Size([SEQ_LEN, CHANNELS, ORIGINAL_WIDTH, ORIGINAL_HEIGHT]), video_in_batch.size()
            video_in_batch = F.interpolate(video_in_batch, size=WIDTH)
            assert video_in_batch.size() == torch.Size([SEQ_LEN, CHANNELS, WIDTH, HEIGHT]), video_in_batch.size()
            video_in_batch = video_in_batch.split(2, 0)
            assert video_in_batch[0].size() == torch.Size([2, CHANNELS, HEIGHT, WIDTH]), video_in_batch[0].size()
            assert len(video_in_batch) == IMG_PAIRS_IN_BATCH
            for batch in video_in_batch:
                # Normally frames have dimension (64,64,3)

                x1 = augmentations(batch)
                x2 = augmentations(batch)
                if VISUALIZE_AUGMENTATIONS > 0:
                    plt.clf()
                    plt.subplot(1, 3, 1)
                    plt.imshow((batch.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
                    plt.subplot(1, 3, 2)
                    plt.imshow((x2.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
                    plt.subplot(1, 3, 3)
                    plt.imshow((x1.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
                    plt.pause(interval=VISUALIZE_AUGMENTATIONS)

                x = torch.cat([x1, x2])
                x = x.to(DEVICE)
                h = model(x)
                h = h.reshape(BATCH_SIZE * 2, -1)
                z = head(h)

                s = F.normalize(z, dim=1)
                cosine_similarity_matrix = torch.matmul(s, s.T)

                similarity_matrix = cosine_similarity_matrix[~mask].view(BATCH_SIZE * 2, BATCH_SIZE * 2 - 1)
                # assert similarity_matrix.shape == labels.shape

                # select and combine multiple positives
                positives = similarity_matrix[labels.bool()].view(BATCH_SIZE * 2, 1)

                # select only the negatives the negatives
                negatives = similarity_matrix[~labels.bool()].view(BATCH_SIZE * 2, BATCH_SIZE * 2 - 2)

                logits = torch.cat([positives, negatives], dim=1)

                logits = logits / TEMPERATURE
                loss = criterion(logits, zeros)
                total_loss += loss.item()
                optim_model.zero_grad()
                optim_head.zero_grad()
                loss.backward()
                optim_model.step()
                optim_head.step()
                batch_bar.set_description("Loss " + str(loss.item()))
                batch_bar.update(BATCH_SIZE)
    total_loss = BATCH_SIZE * 2 * total_loss / len(data)
    losses.append(total_loss)
    epoch_bar.update(1)
    print("total loss=", total_loss)
