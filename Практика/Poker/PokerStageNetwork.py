import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from . import *

class PokerStageNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=350, num_layers=12, output_size=3):
        super(PokerStageNetwork, self).__init__()

        layers = []

        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


