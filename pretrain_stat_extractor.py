import torch
import torch.nn as nn
import torch.nn.functional as F
from crafter import Env
from crafter.engine import Textures


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            16 * 3 * 3, 64
        )  # After pooling, image size is reduced to 3x3
        self.fc2 = nn.Linear(64, 9)  # Output 9 classes (1-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 16 * 3 * 3)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected
        x = self.fc2(
            x
        )  # Output layer (no activation because we'll use CrossEntropyLoss)
        return x


def prepare_data(env: Env):
    textures: Textures = env._textures
    textures.get("grass", 64)
