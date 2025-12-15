import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .PlayerManager import PlayerManager
from .HandCalculator import HandCalculator
from .NeuralAgent import *
from .Logger import *

STAGES ={ "preflop": 0, "flop": 1, "turn": 2, "river": 3, }
ACTIONS = { 0 : "fold", 1 : "raise", 2 : "call", }

class NeuralAgentManager(PlayerManager):
    def __init__(self, player:NeuralAgent):
        super().__init__(player)
        self.episode_memory = []  # <-- ВАЖНО


    def act(self, state):
        if random.random() < self.player.epsilon:
            return int(torch.randint(0, 3, (1,)).item())

        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.player.model(state)
        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.player.model(state)
        q_value = q_values[action]

        with torch.no_grad():
            if done:
                target = torch.tensor(reward, dtype=torch.float32)
            else:
                target = reward + self.player.gamma * torch.max(self.player.model(next_state))

        loss = (q_value - target) ** 2

        self.player.optimizer.zero_grad()
        loss.backward()
        self.player.optimizer.step()

    def ask_decision(self, state_vector:list):

        action_idx = self.act(state_vector)
        action = ACTIONS[action_idx]
        self.player.set_decision(action)

        self.last_state = state_vector
        self.last_action = action_idx

        StaticLogger.print(f"\nИгрок {self.player.name}")
        StaticLogger.print(f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        StaticLogger.print(f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print(f"Ваш выбор: {self.player.decision}")
        return self.player.decision

    def build_state_vector(self, current_bet_normalized, current_stack_normalized, pot_normalize, community_cards, opponents_decision_value,
                           stage="preflop"):
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        stage = STAGES[stage] / len(STAGES)

        state_vector = [hand_strength, current_bet_normalized, current_stack_normalized, pot_normalize, stage]
        state_vector.append(opponents_decision_value)
        state_vector.append(self.decision_value)
        return state_vector

    def decay_epsilon(self):
        if self.player.epsilon > self.player.epsilon_min:
            self.player.epsilon *= self.player.epsilon_decay