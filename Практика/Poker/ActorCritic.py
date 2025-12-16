import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from .Player import *

class ActorCriticNet(nn.Module):
    # Теперь сеть принимает размер двух разных векторов состояния
    def __init__(self, actor_state_size, critic_state_size, action_size):
        super().__init__()

        # 1. Actor (Политика): Берет S_actor
        self.actor_net = nn.Sequential(
            nn.Linear(actor_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        # 2. Critic (Ценность): Берет S_critic
        self.critic_net = nn.Sequential(
            nn.Linear(critic_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # Метод forward теперь принимает S_actor и S_critic
    def forward(self, s_actor, s_critic):
        # Actor использует S_actor
        action_logits = self.actor_net(s_actor)
        if s_critic is not None:
            state_value = self.critic_net(s_critic)

        return action_logits, state_value  # (Action Logits, V(s))


class NeuralACAgent(Player):
    def __init__(self, name="NeuralAgent", stack=100, actor_size=6, critic_size=7, action_size=3):
        super().__init__(name, stack)

        # Передаем два размера в конструктор
        self.ac_net = ActorCriticNet(actor_size, critic_size, action_size)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=3e-4)
        self.gamma = 0.99

        self.memory = deque(maxlen=20000)
        # ... остальные параметры RL (optimizer, gamma) остаются прежними ...

    def get_memory(self):
        return self.memory
    def set_memory(self, memory):
        self.memory = memory

    def reset_for_new_hand(self):
        super().reset_for_new_hand()


