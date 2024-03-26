import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Hyperparameters
random_seed = 123
learning_rate = 0.1
num_epochs = 10
batch_size = 64

# Architecture
num_features = 28*28
num_classes = 10

# Loading dataset
train_dataset = datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=True)

test_dataset = datasets.MNIST(root='data',train=False,transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
    
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    

model = LogisticRegression(num_features=num_features,num_classes=num_classes)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

torch.manual_seed(random_seed)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100

# Training the model for num_epochs
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        
        # note that the PyTorch implementation of
        # CrossEntropyLoss works with logits, not
        # probabilities
        cost = F.cross_entropy(logits, targets)
        # cost = F.nll_loss(probas, targets)
        optimizer.zero_grad()
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' %(epoch+1, num_epochs, batch_idx,len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (epoch+1, num_epochs,compute_accuracy(model, train_loader)))

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))