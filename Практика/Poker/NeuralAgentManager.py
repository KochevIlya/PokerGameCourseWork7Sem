import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
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
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.current_episode_memory = []

    def act(self, state):
        if random.random() < self.player.epsilon:
            return int(torch.randint(0, 3, (1,)).item())

        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.player.model(state)
        return torch.argmax(q_values).item()

    def remember_episode(self, final_reward):
        # 1. Распределяем награду по шагам текущей раздачи и кидаем в ОБЩИЙ буфер
        for s, a, _, s_next, done in self.current_episode_memory:
            # Тут можно добавить затухание награды, но для начала просто final_reward
            self.replay_buffer.append((s, a, final_reward, s_next, done))

        # Очищаем память текущей раздачи
        self.current_episode_memory = []


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

    def train_experience_replay(self):
        # Если в буфере мало данных, не учимся (ждем накопления)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. Берем СЛУЧАЙНЫЙ пакет (вот она, магия!)
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Распаковываем пакет
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 2. Считаем Q_current (через Policy Net)
        # gather выбирает значения только для совершенных действий
        q_values = self.player.model(states).gather(1, actions).squeeze(1)

        # 3. Считаем Q_target (через Target Net!)
        with torch.no_grad():
            next_q_values = self.player.target_net(next_states).max(1)[0]
            # Формула Беллмана
            expected_q_values = rewards + (self.player.gamma * next_q_values * (1 - dones))

        # 4. Считаем Loss и делаем шаг
        loss = (q_values - expected_q_values) ** 2
        loss = loss.mean()

        self.player.optimizer.zero_grad()
        loss.backward()
        self.player.optimizer.step()

    def update_target_network(self):
        # Копируем веса из policy в target
        self.player.target_net.load_state_dict(self.player.model.state_dict())

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