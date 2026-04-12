import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from .Player import *

class ActorCriticNet(nn.Module):
    def __init__(self, actor_state_size, critic_state_size, action_size, history_len=10, action_input_dim=3,
                 lstm_hidden=16):
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

    def forward(self, s_actor, s_critic, history, actor_hidden=None, critic_hidden=None):
        # Ветка Актера
        actor_lstm_out, next_actor_hidden = self.actor_lstm(history, actor_hidden)
        actor_context = actor_lstm_out[:, -1, :]  # Берем последний выход

        # Ветка Критика
        critic_lstm_out, next_critic_hidden = self.critic_lstm(history, critic_hidden)
        critic_context = critic_lstm_out[:, -1, :]  # Берем последний выход

        # Объединение
        actor_input = torch.cat([s_actor, actor_context], dim=1)
        critic_input = torch.cat([s_critic, critic_context], dim=1)

        action_logits = self.actor_net(actor_input)

        if s_critic is not None:
            state_value = self.critic_net(critic_input)

        return action_logits, state_value, next_actor_hidden, next_critic_hidden


class NeuralACAgent(Player):
    def __init__(self, name="NeuralACAgent", stack=100, actor_size=10, critic_size=None, action_size=3, history_len=10, action_vector_size=3):
        super().__init__(name, stack)

        if critic_size is None:
            critic_size = actor_size + 1

        self.actor_size = actor_size
        self.critic_size = critic_size  # Сохраняем для корректного save/load
        self.history_len = history_len
        self.action_vector_size = action_vector_size

        # Передаем два размера в конструктор
        self.ac_net = ActorCriticNet(actor_size, critic_size, action_size,
                                     history_len=history_len,
                                     action_input_dim=action_vector_size)

        # Раздельные Learning Rate: Critic учится быстрее, Actor — медленнее
        self.optimizer = optim.Adam([
            # Actor — медленное обучение (стабильная стратегия)
            {'params': self.ac_net.actor_net.parameters(), 'lr': 1e-4},
            {'params': self.ac_net.actor_lstm.parameters(), 'lr': 1e-4},

            # Critic — быстрое обучение (точная оценка значений)
            {'params': self.ac_net.critic_net.parameters(), 'lr': 5e-4},
            {'params': self.ac_net.critic_lstm.parameters(), 'lr': 5e-4},
        ], lr=1e-4)  # базовый LR (fallback)

        self.gamma = 0.99

        self.actor_hidden = None
        self.critic_hidden = None

    def reset_for_new_hand(self):
        super().reset_for_new_hand()
        self.actor_hidden = None
        self.critic_hidden = None


