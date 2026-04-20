import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from .Player import *
from .Logger import *
class ActorCriticNet(nn.Module):
    def __init__(self, actor_state_size, critic_state_size, action_size):
        super().__init__()

        # --- СЕТЬ АКТЕРА ---
        self.actor_fc1 = nn.Linear(actor_state_size, 128)
        # Вход второго слоя: 128 (от первого слоя) + 1 (проброшенная hand_strength) = 129
        self.actor_fc2 = nn.Linear(128 + 1, 64)
        # Вход финального слоя: 64 (от второго слоя) + 1 (проброшенная hand_strength) = 65
        self.actor_head = nn.Linear(64 + 1, action_size)

        # --- СЕТЬ КРИТИКА ---
        self.critic_fc1 = nn.Linear(critic_state_size, 128)
        # Вход второго слоя: 128 + 2 (проброшенные hand_strength и opp_strength) = 130
        self.critic_fc2 = nn.Linear(128 + 2, 64)
        # Вход финального слоя: 64 + 2 = 66
        self.critic_head = nn.Linear(64 + 2, 1)

    def forward(self, s_actor, s_critic):
        # 1. Извлекаем нужные фичи с помощью срезов [:, 0:1].
        # Срез [:, 0:1] берет элемент с индексом 0, но сохраняет двумерную форму тензора (batch_size, 1), что нужно для конкатенации.
        actor_hand_strength = s_actor[:, 0:1]

        # --- ПРЯМОЙ ПРОХОД АКТЕРА ---
        a1 = F.relu(self.actor_fc1(s_actor))
        # Склеиваем выход 1-го слоя и силу руки: размерность станет (batch_size, 129)
        a1_cat = torch.cat([a1, actor_hand_strength], dim=1)

        a2 = F.relu(self.actor_fc2(a1_cat))
        # Склеиваем выход 2-го слоя и силу руки: размерность станет (batch_size, 65)
        a2_cat = torch.cat([a2, actor_hand_strength], dim=1)

        action_logits = self.actor_head(a2_cat)
        action_logits = torch.clamp(action_logits, min=-4, max=4)

        # --- ПРЯМОЙ ПРОХОД КРИТИКА ---
        state_value = None
        if s_critic is not None:
            critic_hand_strength = s_critic[:, 0:1]
            critic_opp_strength = s_critic[:, -1:] # -1 берет гарантированно последний элемент массива
            # Склеиваем две фичи вместе: размерность станет (batch_size, 2)
            critic_injected_features = torch.cat([critic_hand_strength, critic_opp_strength], dim=1)

            c1 = F.relu(self.critic_fc1(s_critic))
            # Склеиваем выход 1-го слоя критика с двумя фичами: размерность (batch_size, 130)
            c1_cat = torch.cat([c1, critic_injected_features], dim=1)

            c2 = F.relu(self.critic_fc2(c1_cat))
            # Склеиваем выход 2-го слоя критика с двумя фичами: размерность (batch_size, 66)
            c2_cat = torch.cat([c2, critic_injected_features], dim=1)

            state_value = self.critic_head(c2_cat)
            StaticLogger.print(f"Critic:  {state_value}")

        logit_gap = action_logits.max().item() - action_logits.min().item()
        StaticLogger.print(f"Logit Gap (Confidence): {logit_gap:.4f}")

        return action_logits, state_value


class NeuralACAgent(Player):
    def __init__(self, name="NeuralACAgent", stack=100, actor_size=7, critic_size=None, action_size=3):
        super().__init__(name, stack)

        if critic_size is None:
            critic_size = actor_size + 1

        self.actor_size = actor_size
        self.critic_size = critic_size
        # Передаем два размера в конструктор
        self.ac_net = ActorCriticNet(actor_size, critic_size, action_size)
        actor_params = [p for n, p in self.ac_net.named_parameters() if "actor" in n]
        critic_params = [p for n, p in self.ac_net.named_parameters() if "critic" in n]

        self.optimizer = optim.Adam([
            {
                'params': actor_params, 'lr': 1e-4,
                'weight_decay': 1e-3
             }, # Был 5e-5, стал 1e-4 (в 2 раза больше)
            {'params': critic_params, 'lr': 1e-4} # Оставляем старый или можно чуть меньше
        ])
        self.gamma = 0.99

        self.memory = deque(maxlen=20000)

    def get_memory(self):
        return self.memory
    def set_memory(self, memory):
        self.memory = memory

    def reset_for_new_hand(self):
        super().reset_for_new_hand()


