import torch
import torch.nn as nn
import pandas as pd

from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import GTZANDdataset
from model import CNN, CRNN

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

Model = CNN
Mel_dir = "/Users/oldsakura/Desktop/COMP9444/mel"

Train_feature = "X_train.csv"
Test_feature = "X_test.csv"

Train_label = "y_train.csv"
Test_label = "y_test.csv"

NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = pd.read_csv(Train_feature)
x_test = pd.read_csv(Test_feature)

y_train = pd.read_csv(Train_label)
y_test = pd.read_csv(Test_label)

# Dataset

if Model in [CNN, CRNN]:
    train_files = x_train["file_name"].tolist()
    test_files = x_test["file_name"].tolist()

    train_dataset = GTZANDdataset(Mel_dir, train_files)
    test_dataset = GTZANDdataset(Mel_dir, test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


# Model
if Model in [CNN, CRNN]:
    model = Model(NUM_CLASSES).to(DEVICE)

    #Loss

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


#elif Model == Random_Forest:


#elif Model == SVG:



# Training
def train_dl_model():
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

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

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Save Model

    torch.save(model.state_dict(), f"{Model.__name__.lower()}.pth")
    print("Training Finished.")

if __name__ == "__main__":
    if Model in [CNN,CRNN]:
        train_dl_model()