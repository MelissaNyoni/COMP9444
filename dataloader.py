import torch
from torch.utils.data import DataLoader
from dataset import GTZANDdataset

mel_dir = "/Users/oldsakura/Desktop/COMP9444/mel"

dataset = GTZANDdataset(mel_dir)
print(f'Dataset size: {len(dataset)}')

loader = DataLoader(dataset, batch_size = 32, shuffle = True)

for image, labels in loader:
    print(image.shape)
    print(labels.shape)
    print(labels)
    break