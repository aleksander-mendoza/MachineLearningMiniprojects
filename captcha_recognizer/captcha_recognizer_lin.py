from PIL import Image, ImageDraw, ImageFont
from captcha.image import ImageCaptcha
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import math
from matplotlib import pyplot as plt

BATCH_SIZE = 64

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

fonts = []
for path in Path('/usr/share/fonts/').rglob('*.ttf'):
    fonts.append(str(path))
clear_font = ImageFont.FreeTypeFont(fonts[2], size=64)

W = 256
H = 64
MAX_LEN = 6
alphabet = [chr(i) for i in range(ord('a'), ord('z'))] + \
           [chr(i) for i in range(ord('A'), ord('Z'))] + \
           [chr(i) for i in range(ord('0'), ord('9'))]


def gen_captcha(text):
    image = ImageCaptcha(width=W, height=H, fonts=fonts[2:3])
    data = image.generate(text)
    image = Image.open(data)
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image = image.view(H, W, 3)
    image = image.float()
    image.div_(255)
    return image


def gen_str():
    s = ""
    t = torch.empty(MAX_LEN, dtype=torch.long)
    size = int(math.log2(random.randrange(2, 2 ** (MAX_LEN - 1))))
    for i in range(size):
        idx = random.randrange(0, len(alphabet))
        s += alphabet[idx]
        t[i] = idx
    for i in range(size, MAX_LEN):
        t[i] = len(alphabet)
    return s, t


def gen_batch():
    strings = []
    tensors = []
    for _ in range(BATCH_SIZE):
        s, t = gen_str()
        strings.append(s)
        tensors.append(t)
    captchas = [gen_captcha(string) for string in strings]

    tensors = torch.stack(tensors)
    captchas = torch.stack(captchas)
    return strings, tensors, captchas


def show_batch(batch, halt=False):
    strings, one_hot, captchas = batch
    plt.imshow(captchas.view(-1, W, 3))
    if halt:
        plt.show()
    else:
        plt.pause(interval=0.001)


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


class ResnetRNN(nn.Module):

    def __init__(self, bottleneck):
        super(ResnetRNN, self).__init__()
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
        self.lin2 = nn.Linear(bottleneck, (len(alphabet) + 1)*MAX_LEN)

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

        x = x.view(batch_size, 1024)
        x = self.lin1(x)
        x = F.relu(x, True)
        x = self.lin2(x)
        x = x.reshape(BATCH_SIZE, len(alphabet)+1, MAX_LEN)
        x = F.log_softmax(x, dim=1)
        return x


EPOCHS = 1000
model = ResnetRNN(32).to(DEVICE)
distance = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
outer_bar = tqdm(total=EPOCHS, position=0)


def train(visualize):
    # inner_bar = tqdm(total=len(trainset), position=1)
    losses = []
    accuracies = []
    outer_bar.set_description("Epochs")
    for epoch in range(EPOCHS):
        # inner_bar.reset()
        # for data in dataloader:
        strings, expected, captchas = gen_batch()
        expected = expected.to(DEVICE)
        # ===================forward=====================
        output = model(captchas.transpose(1, 3).to(DEVICE))
        # output.size() == (BATCH_SIZE, len(alphabet)+1, MAX_LEN)
        loss = distance(output, expected)
        correct = (output.argmax(dim=1) == expected).sum().item()
        total = BATCH_SIZE * MAX_LEN
        accuracy = correct / total
        accuracies.append(accuracy)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if visualize:
            show_batch((strings, expected, captchas))
        else:
            plt.clf()
            plt.plot(losses, label="Avg loss")
            plt.plot(accuracies, label="Accuracy")
            plt.legend(loc="upper left")
            plt.pause(interval=0.01)
        # inner_bar.update(BATCH_SIZE)
        # inner_bar.set_description("Avg loss %.2f" % (loss.item() / BATCH_SIZE))
        outer_bar.update(1)


train(False)
