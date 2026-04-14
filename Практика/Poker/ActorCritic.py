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

        # Critic с Skip Connections: ключевые признаки (hand_strength, avg_opp_strength)
        # передаются напрямую во второй слой, чтобы не "размываться"
        self.critic_fc1 = nn.Linear(critic_state_size + lstm_hidden, 128)
        self.critic_layer_norm = nn.LayerNorm(128)  # Нормализация для выравнивания масштаба
        self.critic_fc2 = nn.Linear(128 + 2, 64)  # 128 + 2 key features
        self.critic_fc3 = nn.Linear(64, 1)

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
            # === SKIP CONNECTIONS для Critic ===
            # Выделяем ключевые признаки: hand_strength[0] + avg_opp_strength[10]
            key_features = s_critic[:, [0, 10]]  # [Batch, 2]

            # Первый слой + ReLU
            out = F.relu(self.critic_fc1(critic_input))  # [Batch, 128]

            # LayerNorm для выравнивания масштаба признаков
            out = self.critic_layer_norm(out)

            # Skip Connection: конкатенация с ключевыми признаками
            combined = torch.cat([out, key_features], dim=1)  # [Batch, 130]

            # Второй слой + финальный
            out = F.relu(self.critic_fc2(combined))  # [Batch, 64]
            state_value = self.critic_fc3(out)  # [Batch, 1]

        return action_logits, state_value, next_actor_hidden, next_critic_hidden

    def get_critic_value(self, s_critic, history, critic_hidden=None):
        """
        Возвращает предсказанное значение Критика для заданного состояния.
        Используется для валидации (понимает ли сеть силу руки).
        """
        with torch.no_grad():
            # Создаём фиктивный actor input (не важен для Critic с Skip Connection)
            s_actor = s_critic[:, :s_critic.size(1) - 1]  # Убираем avg_opp_strength
            
            _, value, _, _ = self.forward(s_actor, s_critic, history, 
                                          actor_hidden=None, critic_hidden=critic_hidden)
            return value.item()


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

        # Learning Rates: Актор БЫСТРЕЕ Критика
        # Актор должен успевать пробовать стратегии, пока Критик ещё сомневается
        # ФОРСИРОВАНИЕ: actor_lr поднят до 1e-3 для разгона логитов
        self.optimizer = optim.Adam([
            # Actor — ускоренное обучение (пробует стратегии)
            {'params': self.ac_net.actor_net.parameters(), 'lr': 1e-3},
            {'params': self.ac_net.actor_lstm.parameters(), 'lr': 1e-3},

            # Critic — медленное обучение (не обнуляет Advantage слишком быстро)
            {'params': self.ac_net.critic_fc1.parameters(), 'lr': 1e-4},
            {'params': self.ac_net.critic_layer_norm.parameters(), 'lr': 1e-4},
            {'params': self.ac_net.critic_fc2.parameters(), 'lr': 1e-4},
            {'params': self.ac_net.critic_fc3.parameters(), 'lr': 1e-4},
            {'params': self.ac_net.critic_lstm.parameters(), 'lr': 1e-4},
        ], lr=1e-4)  # базовый LR (fallback)

        self.gamma = 0.99

        self.actor_hidden = None
        self.critic_hidden = None

    def reset_for_new_hand(self):
        super().reset_for_new_hand()
        self.actor_hidden = None
        self.critic_hidden = None


