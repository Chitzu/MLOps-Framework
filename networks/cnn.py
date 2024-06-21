import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_channels=8):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, n_channels, 7),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, 2 * n_channels, 5),
            nn.ReLU(),
            nn.BatchNorm2d(2 * n_channels),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * n_channels, 4 * n_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * n_channels, 8 * n_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(8 * n_channels, 8 * n_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(8 * n_channels, 8 * n_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_channels),
            nn.Dropout(0.3)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(8 * n_channels, 8 * n_channels, 5),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(8 * n_channels, 8 * n_channels, 5),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_channels),
            nn.MaxPool2d(3),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 4)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

