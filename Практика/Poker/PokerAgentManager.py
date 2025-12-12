import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .PlayerManager import PlayerManager
from .HandCalculator import HandCalculator


class PokerAgentManager(PlayerManager):
    def __init__(self, player):
        super().__init__(player)
        # Храним тензоры для обучения
        self.last_prediction_tensor = None
        self.last_stage_name = None

    def get_prediction_data(self, stage, state_vector):
        """
        Возвращает тензор (для обучения) и число (для логики).
        Не сохраняет состояние в self, чтобы не перезатереть память раньше времени.
        """
        # Превращаем данные в тензор PyTorch
        state_tensor = torch.FloatTensor(state_vector)

        # Получаем предсказание (тензор с градиентом!)
        # Важно: НЕ делаем detach(), иначе обучение сломается
        prediction_tensor = self.player.networks[stage](state_tensor)

        return prediction_tensor, prediction_tensor.item()

    def make_decision(self, probability):
        if probability < 0.3:
            return "fold"
        elif probability < 0.7:
            return "call"
        else:
            return "raise"

    def train_step(self, next_stage_value, reward=None):
        """
        Обучает сеть, которая сделала ПРЕДЫДУЩИЙ ход (self.last_prediction_tensor).
        """
        if self.last_prediction_tensor is None:
            return

        # Берем оптимизатор той стадии, которая сделала предсказание (например, preflop)
        optimizer = self.player.optimizers[self.last_stage_name]
        optimizer.zero_grad()

        if reward is not None:
            # Конец игры (победа/поражение)
            target = torch.tensor([reward], dtype=torch.float32)

            # Если эпизод закончен, очищаем память, чтобы не переносить состояние в новую игру
            should_clear_memory = True
        else:
            # TD-Learning: обучаем старую сеть предсказывать то, что сейчас думает новая сеть
            # next_stage_value - это просто число (float), градиенты через него не идут, это наша цель
            target = torch.tensor([next_stage_value], dtype=torch.float32) * self.player.gamma
            should_clear_memory = False

        # Считаем ошибку между тем, что сеть предсказала РАНЬШЕ, и тем, что мы узнали СЕЙЧАС
        loss = self.player.criterion(self.last_prediction_tensor, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if should_clear_memory:
            self.last_prediction_tensor = None
            self.last_stage_name = None

    def ask_decision(self, current_bet_normalized, current_stack_normalized, pot_normalize, community_cards,
                     stage="preflop"):

        # 1. Подготовка данных
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        state_vector = torch.FloatTensor([
            hand_strength,
            current_bet_normalized,
            current_stack_normalized,
            pot_normalize
        ])

        # 2. Сначала обучаем предыдущий шаг (если он есть)
        if self.last_prediction_tensor is not None:
            # Бутстрап на основании reward или просто 0
            self.train_step(next_stage_value=0.0)

        # 3. Новый forward (после обучения!)
        current_tensor = self.player.networks[stage](state_vector)
        current_prob = current_tensor.item()

        # 4. Запоминаем текущий forward для следующего train_step
        self.last_prediction_tensor = current_tensor
        self.last_stage_name = stage

        # 5. Выбор действия
        decision = self.make_decision(current_prob)
        self.player.set_decision(decision)

        return self.player.decision

