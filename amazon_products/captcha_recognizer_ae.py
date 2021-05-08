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

BATCH_SIZE = 8

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

fonts = []
for path in Path('/usr/share/fonts/').rglob('*.ttf'):
    fonts.append(str(path))
clear_font = ImageFont.FreeTypeFont(fonts[2], size=64)

W = 256
H = 64


def gen_captcha(text):
    image = ImageCaptcha(width=W, height=H, fonts=fonts[2:3])
    data = image.generate(text)
    image = Image.open(data)
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image = image.view(H, W, 3)
    image = image.float()
    image.div_(255)
    return image


def gen_text(text):
    im = Image.new("L", (W, H))
    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(text, font=clear_font)
    assert w < W
    draw.text(((W - w) / 2, (H - h) / 2), text, (255), font=clear_font)
    im = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
    im = im.view(H, W, 1)
    im = im.float()
    im.div_(255)
    return im


def demo():
    plt.imshow(gen_text("a"), cmap="gray")
    plt.show()
    plt.imshow(gen_text("aHe"), cmap="gray")
    plt.show()
    plt.imshow(gen_captcha("a"))
    plt.show()
    plt.imshow(gen_captcha("abb"))
    plt.show()
    plt.imshow(gen_captcha("a wfee w"))
    plt.show()
    plt.imshow(gen_captcha("a wfee w tre"))
    plt.show()


def gen_str():
    s = ""
    for _ in range(int(math.log2(random.randint(2, 2 ** 5)))):
        s += chr(ord('a') + random.randint(0, ord('z') - ord('a')))
    return s


def gen_batch():
    strings = [gen_str() for _ in range(BATCH_SIZE)]
    captchas = [gen_captcha(string) for string in strings]
    texts = [gen_text(string) for string in strings]
    captchas = torch.stack(captchas)
    texts = torch.stack(texts)
    return strings, captchas, texts


def show_batch(batch, halt=False):
    strings, captchas, texts = batch
    plt.subplot(1, 2, 1)
    plt.imshow(captchas.view(-1, W, 3))
    plt.subplot(1, 2, 2)
    plt.imshow(texts.view(-1, W, 1) * 255, cmap="gray")
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


class Upsample(nn.Module):

    def __init__(self, channels, bottleneck_channels, output_channels):
        super(Upsample, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=channels, out_channels=bottleneck_channels, kernel_size=1,
                                        padding=0,
                                        stride=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels,
                                        kernel_size=3,
                                        output_padding=1, stride=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=bottleneck_channels, out_channels=output_channels, kernel_size=1,
                                        padding=0,
                                        output_padding=2)
        self.conv4 = nn.ConvTranspose2d(in_channels=channels, out_channels=output_channels, kernel_size=1, padding=0,
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


class ResnetAutoencoder(nn.Module):

    def __init__(self, bottleneck):
        super(ResnetAutoencoder, self).__init__()
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
        self.lin2 = nn.Linear(bottleneck, 1024)
        self.layer12 = Upsample(256, 64, 128)
        self.layer13 = Upsample(128, 32, 64)
        self.layer14 = Upsample(64, 16, 32)

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
        # ====== Bottleneck ======
        x = x.view(batch_size, 1024)
        x = self.lin1(x)
        x = F.relu(x, True)
        # ====== Decoder ======
        x = self.lin2(x)
        x = F.relu(x, True)
        x = x.view(batch_size, 256, 4, 1)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        return x


EPOCHS = 1000
model = ResnetAutoencoder(32).to(DEVICE)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
outer_bar = tqdm(total=EPOCHS, position=0)


def train(visualize):
    # inner_bar = tqdm(total=len(trainset), position=1)
    outer_bar.set_description("Epochs")
    for epoch in range(EPOCHS):
        # inner_bar.reset()
        # for data in dataloader:
        strings, captchas, texts = gen_batch()
        # ===================forward=====================
        output = model(captchas.transpose(1, 3).to(DEVICE)).transpose(1, 3)
        loss = distance(output, texts)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if visualize:
            show_batch((strings, captchas, output.cpu()))
        # inner_bar.update(BATCH_SIZE)
        # inner_bar.set_description("Avg loss %.2f" % (loss.item() / BATCH_SIZE))
        outer_bar.update(1)


train(True)
