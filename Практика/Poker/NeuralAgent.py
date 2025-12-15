import torch.optim as optim
import numpy as np
import random
from .Player import *
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class NeuralAgent(Player):
    def __init__(self, name="NeuralAgent", stack=100, state_size=7, action_size=3):
        super().__init__(name, stack)
        self.model = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.model.state_dict())  # Копируем веса
        self.target_net.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.epsilon = 0.2  # стартуем с полного рандома
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.95

    def reset_for_new_hand(self):
        super().reset_for_new_hand()
        self.epsilon = max(self.epsilon_min,self.epsilon * self.epsilon_decay)


