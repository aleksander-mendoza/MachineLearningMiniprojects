import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# If you get 503 while downloading MNIST then download it manually
# wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# tar -zxvf MNIST.tar.gz

BATCH_SIZE = 32

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

transform = transforms.Compose([transforms.ToTensor()])
trainset = tv.datasets.MNIST(root='.', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)


def imshow():
    fig, ax = plt.subplots(1, 2)

    def hover(event):
        if event.inaxes == ax[0] and event.button is not None:
            with torch.no_grad():
                x = torch.tensor([event.xdata, event.ydata], dtype=torch.float)
                x = (x-0.5)*10
                x = x.unsqueeze(0)
                x = model.decode(x.to(DEVICE)).cpu()
                x = x.squeeze(0).squeeze(0)
                ax[1].imshow(x, cmap='gray')
                fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", hover)
    ax[0].set_title("Click on canvas to generate image\nClose window to train next epoch")
    plt.show()


class VariationalAutoencoder(nn.Module):

    def __init__(self, width, height, bottleneck):
        self.width = width
        self.height = height
        super(VariationalAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.hidden_size = (self.width - 4 * 3) * (self.height - 4 * 3) * 8
        self.lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, bottleneck)
        self.log_var = nn.Linear(self.hidden_size, bottleneck)
        self.lin2 = nn.Linear(bottleneck, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.conv4 = nn.ConvTranspose2d(8, 4, kernel_size=5)
        self.conv5 = nn.ConvTranspose2d(4, 2, kernel_size=5)
        self.conv6 = nn.ConvTranspose2d(2, 1, kernel_size=5)

    def decode(self, x):
        batch_size = x.size()[0]
        x = self.lin2(x)
        x = F.relu(x, True)
        x = x + self.lin3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, 8, self.width - 4 * 3, self.height - 4 * 3)
        x = self.conv4(x)
        x = F.relu(x, True)
        x = self.conv5(x)
        x = F.relu(x, True)
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, self.hidden_size)
        x = x + self.lin1(x)
        x = F.relu(x, True)

        mu = self.mu(x)
        log_var = 0.5+self.log_var(x)
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        x = mu + (eps * std)  # sampling

        x = self.decode(x)

        return x, mu, log_var


# Defining Parameters

EPOCHS = 1000
model = VariationalAutoencoder(28, 28, 2).to(DEVICE)
distance = nn.BCELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters())
outer_bar = tqdm(total=EPOCHS, position=0)
inner_bar = tqdm(total=len(trainset), position=1)
outer_bar.set_description("Epochs")

imshow()
for epoch in range(EPOCHS):
    inner_bar.reset()
    for data in dataloader:
        img, _ = data
        img = Variable(img).to(DEVICE)
        # ===================forward=====================
        output, mean, log_variance = model(img)
        KLD = - 0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        loss = distance(output, img) + KLD
        # print(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        inner_bar.update(BATCH_SIZE)
        inner_bar.set_description("Avg loss %.2f" % (loss.item() / BATCH_SIZE))
    # ===================log========================
    outer_bar.update(1)
    torch.save(model.state_dict(), 'vae.pth')
    imshow()
