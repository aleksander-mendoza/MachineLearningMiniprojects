import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from captcha.image import ImageCaptcha
import random
import select
import sys
import math

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
height = 64
width = 256
channels = 3

BATCH_SIZE = 64


class Block(nn.Module):

    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)

        x0 = self.conv2(x)
        x0 = F.relu(x0, True)

        x0 = self.conv3(x0)
        x0 = F.relu(x0, True)

        x0 = self.conv4(x0)
        x0 = F.relu(x0, True)

        x = x0 + x1
        x = F.relu(x, True)
        return x


class Bottleneck(nn.Module):

    def __init__(self, channels, bottleneck_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=bottleneck_channels, kernel_size=1, padding=0,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3,
                               padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=channels, kernel_size=1, padding=0,
                               stride=1)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = F.relu(x0, True)

        x0 = self.conv2(x0)
        x0 = F.relu(x0, True)

        x0 = self.conv3(x0)

        return x + x0


class Downsample(nn.Module):

    def __init__(self, channels, bottleneck_channels, output_channels):
        super(Downsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=bottleneck_channels, kernel_size=1, padding=0,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=3,
                               padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=output_channels, kernel_size=1, padding=0,
                               stride=2)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=output_channels, kernel_size=1, padding=0,
                               stride=2)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = F.relu(x0, True)

        x0 = self.conv2(x0)
        x0 = F.relu(x0, True)

        x0 = self.conv3(x0)

        x1 = self.conv4(x)

        x = x0 + x1
        x = F.relu(x, True)
        return x


class Resnet(nn.Module):

    def __init__(self, bottleneck):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.layer1 = Block(64)
        self.layer2 = Bottleneck(64, 16)
        self.layer3 = Bottleneck(64, 16)
        self.layer4 = Downsample(64, 16, 128)
        self.layer5 = Bottleneck(128, 32)
        self.layer6 = Block(128)
        self.layer7 = Bottleneck(128, 32)
        self.layer8 = Downsample(128, 32, 256)
        self.layer9 = Bottleneck(256, 64)
        self.layer10 = Block(256)
        self.layer11 = Bottleneck(256, 64)
        self.avg1 = nn.AvgPool2d(4)
        self.lin1 = nn.Linear(1024, bottleneck)

    def forward(self, x):
        batch_size = x.size()[0]
        # ====== Encoder ======
        # 8, 3, 256, 64
        x = self.conv1(x)
        x = F.relu(x, True)
        # 8, 64, 128, 32
        x = self.max1(x)
        # 8, 64, 64, 16
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 8, 64, 64, 16
        x = self.layer4(x)
        # 8, 128, 32, 8
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # 8, 128, 32, 8
        x = self.layer8(x)
        # 8, 256, 16, 4
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        # 8, 256, 16, 4
        x = self.avg1(x)
        # 8, 256, 4, 1

        x = x.reshape(batch_size, 1024)
        x = self.lin1(x)
        x = F.relu(x, True)
        return x


class Head(torch.nn.Module):

    def __init__(self):
        super(Head, self).__init__()
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.lin1(x), True)
        x = self.lin2(x)
        return x


model = Resnet(128).to(DEVICE)
head = Head().to(DEVICE)


def augmentations(img):
    crop_h = 32
    crop_w = 128
    i = torch.randint(height - crop_h + 1, size=(1,)).item()
    j = torch.randint(width - crop_w + 1, size=(1,)).item()
    img = F2.crop(img, i, j, crop_h, crop_w)
    img = F2.resize(img, (height, width))
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


MAX_LEN = 6
alphabet = [chr(i) for i in range(ord('a'), ord('z'))] + \
           [chr(i) for i in range(ord('A'), ord('Z'))] + \
           [chr(i) for i in range(ord('0'), ord('9'))]
fonts = []
for path in Path('/usr/share/fonts/').rglob('*.ttf'):
    fonts.append(str(path))
clear_font = ImageFont.FreeTypeFont(fonts[2], size=64)


def gen_captcha(text):
    image = ImageCaptcha(width=width, height=height, fonts=fonts[2:3])
    data = image.generate(text)
    image = Image.open(data)
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image = image.view(height, width, 3)
    image = image.float()
    image.div_(255)
    return image


def gen_str():
    s = ""
    size = int(math.log2(random.randrange(2, 2 ** (MAX_LEN - 1))))
    for i in range(size):
        idx = random.randrange(0, len(alphabet))
        s += alphabet[idx]
    return s


optim_model = torch.optim.Adam(model.parameters(), lr=0.00001)
optim_head = torch.optim.Adam(head.parameters(), lr=0.00001)
VISUALIZE_AUGMENTATIONS = 0
criterion = torch.nn.CrossEntropyLoss()
labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
labels = labels.to(DEVICE)
mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
labels = labels[~mask].view(labels.shape[0], -1)
zeros = torch.zeros(BATCH_SIZE * 2, dtype=torch.long).to(DEVICE)
TEMPERATURE = 0.05
EPOCHS = 1000000
losses = []
freeze_weights = False
for epoch in range(EPOCHS):
    i, _, _ = select.select([sys.stdin], [], [], 0)
    if i:
        line = sys.stdin.readline().strip()
        if line == "exit":
            break
        elif line == "freeze":
            freeze_weights = not freeze_weights
            print("Freeze weights: "+str(freeze_weights))
        elif line == "save":
            torch.save(model.state_dict(), 'simclr.pth')
            print("Model saved")
        else:
            print("Unknown command: "+line)
    captchas = [gen_captcha(gen_str()) for _ in range(BATCH_SIZE)]
    captchas = torch.stack(captchas)
    c = captchas.transpose(1, 3).transpose(2, 3)
    x1 = augmentations(c)
    x2 = augmentations(c)
    if VISUALIZE_AUGMENTATIONS > 0:
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(captchas.reshape(-1, width, channels))
        plt.subplot(1, 3, 2)
        plt.imshow(x2.transpose(2, 3).transpose(1, 3).reshape(-1, width, channels))
        plt.subplot(1, 3, 3)
        plt.imshow(x1.transpose(2, 3).transpose(1, 3).reshape(-1, width, channels))
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
    if not freeze_weights:
        optim_model.zero_grad()
        optim_head.zero_grad()
        loss.backward()
        optim_model.step()
        optim_head.step()
    if VISUALIZE_AUGMENTATIONS == 0:
        losses.append(loss.item())
        plt.clf()
        plt.plot(losses)
        plt.pause(interval=0.001)

open('captcha_recognizer_rnn.py')
