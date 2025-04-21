import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Set dataset path
basepath = os.getcwd()

# Load EMNIST dataset
trainset = torchvision.datasets.EMNIST(
    root=os.path.join(basepath, 'data'),
    split="balanced",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Compute mean and standard deviation for normalization
mean = trainset.data.float().mean() / 255
std = trainset.data.float().std() / 255
print(f"Mean: {mean}, Std: {std}")

# Define transformations (normalized using computed values)
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),  # Rotate 90 degrees clockwise
    # transforms.Lambda(lambda img: transforms.functional.hflip(img)),       # Horizontal flip
    transforms.ToTensor()
])

# Load datasets with normalization
train_dataset = torchvision.datasets.EMNIST("data", split="balanced", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST("data", split="balanced", train=False, download=True, transform=transform)

# Check dataset size
print(train_dataset.data.shape, test_dataset.data.shape)

# Define DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Display some images
examples = enumerate(test_loader)
batch_idx, (example_data, example_target) = next(examples)
print(batch_idx, example_data.shape, example_target.shape)

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation=None)
    plt.title(f"Ground truth: {example_target[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  # 1x20x4x4
        self.fc2 = nn.Linear(50, 47)  # 47 classes in EMNIST (Balanced)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # Fixed dropout
        x = self.fc2(x)  # Removed ReLU here
        return F.log_softmax(x, dim=1)

# Initialize Model, Loss, and Optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\t Loss: {loss.item():.6f}")
    torch.save(model.state_dict(), "cnn_emnist.pth")

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            test_loss += criterion(output, label).item()  # Fixed loss accumulation
            pred = output.argmax(dim=1, keepdim=True)  # Get predictions
            correct += pred.eq(label.view_as(pred)).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

# Train the model
num_epochs = 1
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()
