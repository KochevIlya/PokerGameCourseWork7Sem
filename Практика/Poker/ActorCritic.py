import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from .Player import *

class ActorCriticNet(nn.Module):
    def __init__(self, actor_state_size, critic_state_size, action_size):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(actor_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(critic_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s_actor, s_critic):
        action_logits = self.actor_net(s_actor)
        if s_critic is not None:
            state_value = self.critic_net(s_critic)

        return action_logits, state_value


class NeuralACAgent(Player):
    def __init__(self, name="NeuralACAgent", stack=100, actor_size=10, critic_size=None, action_size=3):
        super().__init__(name, stack)

        if critic_size is None:
            critic_size = actor_size + 1

        self.actor_size = actor_size
        # Передаем два размера в конструктор
        self.ac_net = ActorCriticNet(actor_size, critic_size, action_size)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=5e-5)
        self.gamma = 0.99

        self.memory = deque(maxlen=20000)

    def get_memory(self):
        return self.memory
    def set_memory(self, memory):
        self.memory = memory

    def reset_for_new_hand(self):
        super().reset_for_new_hand()


