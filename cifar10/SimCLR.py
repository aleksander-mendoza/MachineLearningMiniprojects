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


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
crop = transforms.RandomCrop(32, padding=4)
flip = transforms.RandomHorizontalFlip()
height = 32
width = 32
channels = 3
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
data = torch.from_numpy(trainset.data).float()
data = data
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
BATCH_SIZE = 512
trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# class CNN(torch.nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(5, 5))
#         self.conv3 = nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(5, 5))
#         self.conv4 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(5, 5))
#         self.conv5 = nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(5, 5))
#         self.conv6 = nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(5, 5))
#         self.conv7 = nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(5, 5))
#         self.lin = nn.Linear(18*4*4, 256)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x), True)
#         x = F.relu(self.conv2(x), True)
#         x = F.relu(self.conv3(x), True)
#         x = F.relu(self.conv4(x), True)
#         x = F.relu(self.conv5(x), True)
#         x = F.relu(self.conv6(x), True)
#         x = F.relu(self.conv7(x), True)
#         x = x.reshape(BATCH_SIZE*2, 18*4*4)
#         x = F.relu(self.lin(x), True)
#         return x


class Head(torch.nn.Module):

    def __init__(self):
        super(Head, self).__init__()
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.lin1(x), True)
        x = self.lin2(x)
        return x


model = torchvision.models.resnet18(pretrained=False, num_classes=128).to(DEVICE)
head = Head().to(DEVICE)


def augmentations(img):
    crop_h = 16
    crop_w = 16
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


optim = torch.optim.Adam(model.parameters(), lr=0.00001)
VISUALIZE_AUGMENTATIONS = 0
criterion = torch.nn.CrossEntropyLoss()
labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
labels = labels.to(DEVICE)
mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
labels = labels[~mask].view(labels.shape[0], -1)
zeros = torch.zeros(BATCH_SIZE*2, dtype=torch.long).to(DEVICE)
TEMPERATURE = 0.05
EPOCHS = 1000000
epoch_bar = tqdm(total=EPOCHS, position=1, desc="Epoch")
batch_bar = tqdm(total=len(data), position=2)
losses = []
for epoch in range(EPOCHS):
    total_loss = 0
    batch_bar.reset()
    if VISUALIZE_AUGMENTATIONS == 0:
        plt.clf()
        plt.plot(losses)
        plt.pause(interval=0.001)
    for batch in trainloader:
        x1 = augmentations(batch)
        x2 = augmentations(batch)
        if VISUALIZE_AUGMENTATIONS > 0:
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow((batch.transpose(3, 1).reshape(-1, width, channels)))
            plt.subplot(1, 3, 2)
            plt.imshow((x2.transpose(3, 1).reshape(-1, width, channels)))
            plt.subplot(1, 3, 3)
            plt.imshow((x1.transpose(3, 1).reshape(-1, width, channels)))
            plt.pause(interval=VISUALIZE_AUGMENTATIONS)

        x = torch.cat([x1, x2])
        x = x.to(DEVICE)
        h = model(x)
        h = h.reshape(BATCH_SIZE*2, -1)
        z = head(h)

        s = F.normalize(z, dim=1)
        cosine_similarity_matrix = torch.matmul(s, s.T)

        similarity_matrix = cosine_similarity_matrix[~mask].view(BATCH_SIZE*2, BATCH_SIZE*2 - 1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(BATCH_SIZE*2, 1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(BATCH_SIZE*2, BATCH_SIZE*2-2)

        logits = torch.cat([positives, negatives], dim=1)

        logits = logits / TEMPERATURE
        loss = criterion(logits, zeros)
        total_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        batch_bar.set_description("Loss "+str(loss.item()))
        batch_bar.update(BATCH_SIZE)
    total_loss = BATCH_SIZE*2*total_loss/len(data)
    losses.append(total_loss)
    epoch_bar.update(1)
    print("total loss=", total_loss)



