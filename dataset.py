import os
import torch
import numpy as np
from torch.utils.data import Dataset

class GTZANDdataset(Dataset):
    def __init__(self, mel_dir, file_list = None):
        self.mel_dir = mel_dir
        self.labels = {
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
        genres = sorted(os.listdir(mel_dir))
        for genre in genres:
            genre_path = os.path.join(mel_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            
            files = sorted(os.listdir(genre_path))
            for file in files:
                if not file.endswith(".npy"):
                    continue
                if file_list is not None:
                    if file not in file_list:
                        continue
                file_path = os.path.join(genre_path, file)
                label = self.labels[genre]
                self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        mel = np.load(file_path)
        mel = torch.tensor(mel, dtype = torch.float32)
        mel = mel.unsqueeze(0)
        label = torch.tensor(label, dtype = torch.long)
        return mel, label
