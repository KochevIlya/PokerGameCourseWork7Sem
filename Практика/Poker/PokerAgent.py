import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .Player import *

class PokerNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=16):
        super(PokerNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # Функция активации
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Один выход
            nn.Sigmoid()  # Прижимает выход к диапазону [0, 1]
        )

    def forward(self, x):
        return self.model(x)


class PokerAgent(Player):
    def __init__(self,name="Agent", learning_rate=0.01, gamma=0.99):
        """
        gamma (дисконтирующий фактор): насколько важен будущий выигрыш по сравнению с текущим.
        """
        super().__init__(name)
        self.gamma = gamma

        # Создаем 4 независимые нейросети для каждого раунда
        self.networks = {
            'preflop': PokerNet(),
            'flop': PokerNet(),
            'turn': PokerNet(),
            'river': PokerNet()
        }

        # Создаем оптимизаторы для каждой сети (они обновляют веса)
        self.optimizers = {
            name: optim.Adam(net.parameters(), lr=learning_rate)
            for name, net in self.networks.items()
        }

        # Функция потерь (MSE - среднеквадратичная ошибка)
        self.criterion = nn.MSELoss()

        # Память для хранения состояния текущего раунда, чтобы обучиться позже
        self.last_state = None
        self.last_stage = None
        self.last_prediction = None