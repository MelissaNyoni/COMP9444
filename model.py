import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 3,
            padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3, 
            padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(start_dim = 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
            in_features = 64,
            out_features = num_classes
            )
        )
        
        

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        x = self.gap(x)
        x = self.flatten(x)

        x = self.classifier(x)

        return x
    


class CRNN(nn.Module):
    def __init__(self,
                 num_classes = 10,
                 freq_bins_after_pool = 8,
                 lstm_hidden_size = 128,
                 lstm_num_layers = 2,
                 dropout = 0.3
                 ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 3,
            padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3, 
            padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((freq_bins_after_pool,None))

        self.flatten = nn.Flatten(start_dim = 2)

        lstm_input_size = 64 * freq_bins_after_pool

        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            dropout = dropout if lstm_num_layers > 1 else 0
        )


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
            in_features = lstm_hidden_size,
            out_features = num_classes
            )
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.freq_pool(x)
            
        x = x.permute(0,3,1,2)

        x = self.flatten(x)

        x,_ = self.lstm(x)

        x = x[:, -1, :]

        x = self.classifier(x)

        return x
            