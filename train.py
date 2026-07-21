import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import GTZANDdataset
from model import CNN

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset

mel_dir = "/Users/oldsakura/Desktop/COMP9444/mel"
dataset = GTZANDdataset(mel_dir)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Model

model = CNN(num_classes=10)
model = model.to(DEVICE)


# Loss Function

criterion = nn.CrossEntropyLoss()


# Optimizer

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# Save Model

torch.save(model.state_dict(), "cnn.pth")
print("Training Finished.")