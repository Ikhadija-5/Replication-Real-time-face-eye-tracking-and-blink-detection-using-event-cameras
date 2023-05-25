import torch
import torch.nn as nn
import model
from data import ObjectDetectionDataset
import torch.optim as optim
from model import GR_YOLO
from torch.utils.data import DataLoader, random_split



def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[Epoch %d, Batch %d] Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

def evaluate_model(model, test_data, test_labels):
    inputs = torch.tensor(test_data.values).float()
    labels = torch.tensor(test_labels.values)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")