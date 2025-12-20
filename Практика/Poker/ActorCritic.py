import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from .Player import *

class ActorCriticNet(nn.Module):
    def __init__(self, actor_state_size, critic_state_size, action_size, history_len=10, action_input_dim=3,
                 lstm_hidden=32):
        super().__init__()

        # --- ИЗМЕНЕНИЕ: Две раздельные LSTM ---
        self.actor_lstm = nn.LSTM(input_size=action_input_dim,
                                  hidden_size=lstm_hidden,
                                  batch_first=True)

        self.critic_lstm = nn.LSTM(input_size=action_input_dim,
                                   hidden_size=lstm_hidden,
                                   batch_first=True)

        self.actor_net = nn.Sequential(
            nn.Linear(actor_state_size + lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(critic_state_size + lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s_actor, s_critic, history):
        # Ветка Актера
        actor_lstm_out, _ = self.actor_lstm(history)
        actor_context = actor_lstm_out[:, -1, :]  # Берем последний выход

        # Ветка Критика
        critic_lstm_out, _ = self.critic_lstm(history)
        critic_context = critic_lstm_out[:, -1, :]  # Берем последний выход

        # Объединение
        actor_input = torch.cat([s_actor, actor_context], dim=1)
        critic_input = torch.cat([s_critic, critic_context], dim=1)

        action_logits = self.actor_net(actor_input)

        if s_critic is not None:
            state_value = self.critic_net(critic_input)

        return action_logits, state_value


class NeuralACAgent(Player):
    def __init__(self, name="NeuralACAgent", stack=100, actor_size=10, critic_size=None, action_size=3, history_len=10, action_vector_size=3):
        super().__init__(name, stack)

        if critic_size is None:
            critic_size = actor_size + 1

        self.history_len = history_len
        self.action_vector_size = action_vector_size

        self.actor_size = actor_size
        # Передаем два размера в конструктор
        self.ac_net = ActorCriticNet(actor_size, critic_size, action_size,
                                     history_len=history_len,
                                     action_input_dim=action_vector_size)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=5e-4)
        self.gamma = 0.99

        self.memory = deque(maxlen=20000)

    def get_memory(self):
        return self.memory
    def set_memory(self, memory):
        self.memory = memory

    def reset_for_new_hand(self):
        super().reset_for_new_hand()


