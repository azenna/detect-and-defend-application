import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as tud
import random

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, data_loader, epoch=5):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    # do training
    for e in range(epoch):
        for i, (inputs, labels) in enumerate(data_loader, 0):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

    return loss.item()


def test_model(model):
    batch_size = 4
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    testset = torchvision.datasets.MNIST(
        root="./classifier/data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct // total


def random_dataloader(samples):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    batch_size = 4
    dataset = torchvision.datasets.MNIST(
        root="./classifier/data", train=True, download=True, transform=transform
    )

    # random sample
    dataset = [random.choice(dataset) for i in range(samples)]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return dataloader
