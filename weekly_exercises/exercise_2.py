import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exercise_answers
from exercise_answers import answers_1

# Code from last week
train_dataset, test_dataset, train_loader, test_loader = answers_1.prepare_data()

# Layers and Dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.dropout1 = nn.Dropout(0.2)
        #### WRITE YOUR CODE BELOW ####
        self.fc2 = None # TODO: Write a Linear nn layer that takes 512 features and outputs 256
        self.dropout2 = None # TODO: Use the nn.Dropout function to set 20% elements to 0
        #### WRITE YOUR CODE ABOVE ####
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = Net()

# Loss Function and Learning Rate
criterion = None # TODO: Review the documentation for the nn.CrossEntropyLoss() function
optimizer = optim.Adam(model.parameters()) # TODO: Change the learning rate in the function to 0.001

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        val_loss = validate(model, test_loader, criterion)
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.3f}, Validation Loss: {val_loss:.3f}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def validate(model, test_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(test_loader)

# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

test(model, test_loader)