import os
import numpy as np

import torch
import torch.nn as nn
import pandas as pd

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
from model import CNN, CRNN

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class GTZANDdataset(Dataset):
    def __init__(self, mel_dir, x_df, y_df):
        self.mel_dir = mel_dir

        assert len(x_df) == len(y_df)
        assert (x_df["filename"] == y_df["filename"]).all()

        self.genre_to_label = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9
        }

        self.samples = []
        for i in range(len(x_df)):
            filename = x_df.iloc[i]["filename"]
            genre = y_df.iloc[i]["genre"]
            file_path = os.path.join(mel_dir, genre, filename + ".npy")

            #check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)

            label = self.genre_to_label[genre]
            self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        mel = np.load(file_path)

        # Standardize each spectrogram
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)
        label = torch.tensor(label, dtype = torch.long)
        return mel, label




BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 50
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Mel_dir = "mel"

#Load CSV
Train_feature = "X_train.csv"

Train_label = "y_train.csv"

x_train = pd.read_csv(Train_feature)

y_train = pd.read_csv(Train_label)


#Train

def train_dl_model(model, train_loader, optimizer, criterion):

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

    train_accuracy = 100 * correct / total

    train_loss = total_loss / len(train_loader)

    return train_accuracy , train_loss

#Validation

def validation(model, validation_loader):
        
        model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in validation_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, dim=1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        validation_accuracy = 100 * correct  / total

        validation_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro"
        )

        return validation_accuracy, validation_f1
    
#Cross Validation

def cross_validation(train_dataset, Model):

    labels = [label for _, label in train_dataset.samples]
    kf = StratifiedKFold(
        n_splits = 5, 
        shuffle = True, 
        random_state = 42
    )
    
    fold_f1_scores = []

    fold_results = []

    for fold, (train_idx, validation_idx) in enumerate(kf.split(train_dataset.samples, labels)):

        print(f"Fold {fold + 1}")
        best_f1 = 0

        train_subset = Subset(train_dataset, train_idx)
        validation_subset = Subset(train_dataset,validation_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_subset,
            batch_size = BATCH_SIZE,
            shuffle = False
        )

        model = Model(NUM_CLASSES).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        for epoch in range(EPOCHS):
            train_accuracy , train_loss = train_dl_model(model, train_loader, optimizer, criterion)
            validation_accuracy, validation_f1 = validation(model, validation_loader)

            if validation_f1 > best_f1:
                best_f1 = validation_f1

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Loss: {train_loss:.4f} "
                f"Train Accuracy: {train_accuracy:.2f}% "
                f"Validation Accuracy: {validation_accuracy:.2f}% "
                f"Macro F1:{validation_f1:.4f}"
            )

        fold_f1_scores.append(best_f1)

        fold_results.append({
            "Model": Model.__name__,
            "Fold": fold + 1,
            "Macro F1": round(best_f1, 4)
        })
        
    avg_f1 = sum(fold_f1_scores) / len(fold_f1_scores)
    
    print(f"Average Macro F1 :{avg_f1:.4f}") 
   
    return avg_f1, fold_results
   

#train_dataset = GTZANDdataset(Mel_dir, x_train, y_train)
    
#results = []
#fold_summary = []

#for Model in [CNN, CRNN]:
    #print(f"{Model.__name__}:")

    #avg_f1, fold_results = cross_validation(train_dataset, Model)
    #fold_summary.extend(fold_results)

    #results.append({
    #"Model": Model.__name__,
    #"Average Macro F1": round(avg_f1, 4)
    #})

        
#results_df = pd.DataFrame(results)
#results_df.to_csv("experiment_results.csv", index=False)
#fold_df = pd.DataFrame(fold_summary)
#fold_df.to_csv("fold_results.csv",index=False)
#print(results_df)
 