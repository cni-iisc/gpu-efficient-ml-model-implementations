import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from time import time

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using {device} device")
# Define the neural network architecture

net_size = 500

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, net_size)
        self.fc2 = nn.Linear(net_size, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the MNIST dataset
train_dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root="data", train=False, transform=ToTensor())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

# Initialize the neural network
model = NeuralNetwork().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.7)

start_time = time()

# Training loop with 10 iterations
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    end_time = time()
    total_time = end_time - start_time
    print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}% Time = {total_time:.2f}s Time per epoch = {total_time/(epoch+1):.2f}s")