import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
crop = transforms.RandomCrop(32, padding=4)
flip = transforms.RandomHorizontalFlip()
h = 32
w = 32
c = 3
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
data = torch.from_numpy(trainset.data).float()
data.div_(255)
print("SHAPE = (B, W, H, C)")
print("SHAPE =", data.size())
use_precomputed = True
if use_precomputed:
    std, mean = torch.tensor([0.2470, 0.2435, 0.2616]), torch.tensor([0.4914, 0.4822, 0.4465])
else:
    std, mean = torch.std_mean(data, dim=(0, 1, 2))
print("MEAN =", mean)
print("STD =", std)
# data.sub_(mean).div_(std)
data = data.transpose(1, 3)
print("SHAPE = (B, C, H, W)")
print("SHAPE =", data.size())
trainloader = torch.utils.data.DataLoader(
    data, batch_size=4, shuffle=True, num_workers=2)


# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))

    def forward(self, x):
        x = F.relu(self.conv1(x), True)
        x = F.relu(self.conv2(x), True)
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv4(x), True)
        return x


class Head(torch.nn.Module):

    def __init__(self):
        super(Head, self).__init__()
        self.lin1 = nn.Linear(32, 8)
        self.lin2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.lin1(x), True)
        x = self.lin2(x)
        return x


model = CNN().to(DEVICE)
head = Head().to(DEVICE)


def augmentations(img):
    crop_h = 16
    crop_w = 16
    i = torch.randint(h - crop_h + 1, size=(1,)).item()
    j = torch.randint(w - crop_w + 1, size=(1,)).item()
    img = F2.crop(img, i, j, crop_h, crop_w)
    img = F2.resize(img, (h, w))
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


VISUALIZE_AUGMENTATIONS = 1

for epoch in range(100):
    for batch in trainloader:
        x1 = augmentations(batch)
        x2 = augmentations(batch)
        if VISUALIZE_AUGMENTATIONS > 0:
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow((batch.transpose(3, 1).reshape(-1, w, c)))
            plt.subplot(1, 3, 2)
            plt.imshow((x2.transpose(3, 1).reshape(-1, w, c)))
            plt.subplot(1, 3, 3)
            plt.imshow((x1.transpose(3, 1).reshape(-1, w, c)))
            plt.pause(interval=VISUALIZE_AUGMENTATIONS)

        # x1 = model(x1)
        # x2 = model(x2)
