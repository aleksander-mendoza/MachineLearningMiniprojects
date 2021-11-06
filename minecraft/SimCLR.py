import os
import random

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
import cv2

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


model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 128)  # By default the pretrained ResnetHas 1000 classes, but we only want 128
head = Head()
head = head.to(DEVICE)
model = model.to(DEVICE)


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


def get_model_memory_usage(m):
    mem_params = sum([param.nelement() * param.element_size() for param in m.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in m.buffers()])
    return mem_params + mem_bufs


optim_model = torch.optim.Adam(model.parameters(), lr=0.00001)
optim_head = torch.optim.Adam(head.parameters(), lr=0.00001)
VISUALIZE_AUGMENTATIONS = 0
criterion = torch.nn.CrossEntropyLoss()
labels = torch.cat([torch.arange(IMG_PAIRS_IN_BATCH) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
labels = labels.to(DEVICE)
mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
labels = labels[~mask].view(labels.shape[0], -1)
zeros = torch.zeros(IMG_PAIRS_IN_BATCH * 2, dtype=torch.long).to(DEVICE)
TEMPERATURE = 0.05
EPOCHS = 1000000
data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='data', num_workers=1, minimum_size_to_dequeue=2)
losses = []
SEQ_LEN = IMG_PAIRS_IN_BATCH * 2
ALL_MISSIONS = [d for d in os.listdir('data') if d.startswith('MineRL')]
ALL_RECORDINGS = [mission + '/' + recording for mission in ALL_MISSIONS for recording in os.listdir('data/' + mission)]
print("Found " + str(len(ALL_RECORDINGS)) + " recordings")

for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    total_loss = 0

    if VISUALIZE_AUGMENTATIONS == 0:
        plt.clf()
        plt.plot(losses)
        plt.pause(interval=0.001)
    random.shuffle(ALL_RECORDINGS)
    for recording in tqdm(ALL_RECORDINGS, desc="Batch"):
        video = cv2.VideoCapture('data/' + recording + '/recording.mp4')
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        anchors = []
        targets = []
        for pair in range(IMG_PAIRS_IN_BATCH):
            def read(offset):
                assert video.isOpened()
                while True:
                    video.set(cv2.CAP_PROP_POS_FRAMES, offset)
                    was_successful, frame = video.read()
                    if not was_successful:
                        offset -= 1
                        print("Error ", offset, "/", frame_count)
                        continue
                    frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                    frame = torch.from_numpy(frame)
                    frame = frame.transpose(0, 2)
                    frame = frame / 255
                    assert frame.size() == torch.Size([CHANNELS, HEIGHT, WIDTH])
                    return frame, offset


            frame_offset = random.randint(0, frame_count - 100)  # -100 gives margin of safety,
            # because video might end earlier than frame_count
            target, frame_offset = read(frame_offset)
            targets.append(target)
            frame_offset -= random.randint(0, 7)
            anchor, _ = read(frame_offset)
            anchors.append(anchor)

        video.release()
        anchors = torch.stack(anchors)
        targets = torch.stack(targets)
        original = anchors
        anchors = augmentations(anchors)
        targets = augmentations(targets)
        if VISUALIZE_AUGMENTATIONS > 0:
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow((original.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
            plt.subplot(1, 3, 2)
            plt.imshow((anchors.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
            plt.subplot(1, 3, 3)
            plt.imshow((targets.transpose(3, 1).reshape(-1, WIDTH, CHANNELS)))
            plt.pause(interval=VISUALIZE_AUGMENTATIONS)

        x = torch.cat([anchors, targets])
        x = x.to(DEVICE)
        h = model(x)
        h = h.reshape(IMG_PAIRS_IN_BATCH * 2, -1)
        z = head(h)

        s = F.normalize(z, dim=1)
        cosine_similarity_matrix = torch.matmul(s, s.T)

        similarity_matrix = cosine_similarity_matrix[~mask].view(IMG_PAIRS_IN_BATCH * 2, IMG_PAIRS_IN_BATCH * 2 - 1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(IMG_PAIRS_IN_BATCH * 2, 1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(IMG_PAIRS_IN_BATCH * 2, IMG_PAIRS_IN_BATCH * 2 - 2)

        logits = torch.cat([positives, negatives], dim=1)

        logits = logits / TEMPERATURE
        loss = criterion(logits, zeros)
        total_loss += loss.item()
        optim_model.zero_grad()
        optim_head.zero_grad()
        loss.backward()
        optim_model.step()
        optim_head.step()
    total_loss = IMG_PAIRS_IN_BATCH * 2 * total_loss / len(ALL_RECORDINGS)
    losses.append(total_loss)
