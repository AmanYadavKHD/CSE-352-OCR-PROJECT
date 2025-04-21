import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.architecture import CNN

# Clear terminal
os.system('clear')

# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n\nCommencing Data Processing")

# Load dataset
train_dataset_x = np.load("cache/train_dataset_x.npy")
train_dataset_y = np.load("cache/train_dataset_y.npy")
test_dataset_x = np.load("cache/test_dataset_x.npy")
test_dataset_y = np.load("cache/test_dataset_y.npy")

# Convert to torch tensors and move to device
train_dataset_x = torch.tensor(train_dataset_x, dtype=torch.float32).to(device)
train_dataset_y = torch.tensor(train_dataset_y, dtype=torch.long).to(device)
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32).to(device)
test_dataset_y = torch.tensor(test_dataset_y, dtype=torch.long).to(device)

# Check for NaN values
if torch.isnan(train_dataset_x).any() or torch.isnan(train_dataset_y).any():
    print("NaN detected in training data. Replacing NaNs with 0.")
    train_dataset_x = torch.nan_to_num(train_dataset_x, nan=0.0)
    train_dataset_y = torch.nan_to_num(train_dataset_y, nan=0)

if torch.isnan(test_dataset_x).any() or torch.isnan(test_dataset_y).any():
    print("NaN detected in test data. Replacing NaNs with 0.")
    test_dataset_x = torch.nan_to_num(test_dataset_x, nan=0.0)
    test_dataset_y = torch.nan_to_num(test_dataset_y, nan=0)

print(f"Training dataset size: {train_dataset_x.size()}")
print(f"Testing dataset size: {test_dataset_x.size()}")

# Create dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(train_dataset_x, train_dataset_y)
test_dataset = torch.utils.data.TensorDataset(test_dataset_x, test_dataset_y)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

print("Data Processing Complete\n\n")

# Initialize model and move to device
print("Initializing model\n\n")
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 🔻 Lower LR for stability

# Initialize GradScaler for mixed precision
scaler = torch.amp.GradScaler("cuda")

# Variables to track loss values and accuracy for plotting
train_losses = []
test_losses = []
accuracies = []

# Clear CUDA cache
torch.cuda.empty_cache()

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0  
    print(len(train_loader))

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda"):
            output = model(data)
            loss = criterion(output, targets)

        # Check for NaN loss
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at batch {batch_idx}, skipping batch.")
            continue


        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        running_loss += loss.item()
        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

    avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch} - Avg Training Loss: {avg_train_loss:.6f}")

    torch.save(model.state_dict(), "model.pth")

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            output = model(data)
            loss = criterion(output, targets)
            
            test_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.6f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    test_losses.append(test_loss)
    accuracies.append(accuracy)

# Train the model
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss', marker='s')
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('training_val_loss_plot.png')

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy (%)', color='green', marker='^')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('accuracy_plot.png')
