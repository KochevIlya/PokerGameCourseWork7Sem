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
        self.episode_memory = []
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.player.epsilon:
            return int(torch.randint(0, 3, (1,)).item())

        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.player.model(state)
        return torch.argmax(q_values).item()

    def remember_episode(self, final_reward):
        for s, a, _, s_next, done in self.player.get_memory():
            self.replay_buffer.append((s, a, final_reward, s_next, done))
        self.player.set_memory([])


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
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)


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
            # 1. Policy Net выбирает действие A' для S'
            next_action_q_model = self.player.model(next_states).max(1)[1].unsqueeze(1)
            # 2. Target Net оценивает это действие A'
            next_q_values = self.player.target_net(next_states).gather(1, next_action_q_model).squeeze(1)
            # Формула Беллмана
            expected_q_values = rewards + (self.player.gamma * next_q_values * (1 - dones))

        # 4. Считаем Loss и делаем шаг
        loss = (q_values - expected_q_values) ** 2
        loss = loss.mean()

        self.player.optimizer.zero_grad()
        loss.backward()
        self.player.optimizer.step()

    def update_target_network(self):
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

    def build_state_vector(self, community_cards, current_bet_normalized, current_stack_normalized, pot_normalize, game_decision_value,
                           stage="preflop"):
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        stage = STAGES[stage] / len(STAGES)

        state_vector = [hand_strength, current_bet_normalized, current_stack_normalized, pot_normalize, stage]
        state_vector.append(game_decision_value)
        state_vector.append(self.decision_value / self.num_bets)
        return state_vector

    def decay_epsilon(self):
        if self.player.epsilon > self.player.epsilon_min:
            self.player.epsilon *= self.player.epsilon_decay

    def save_model(self, filename="neural_agent.pth", save_dir="models"):
        """
        Сохраняет веса модели, оптимизатора и другие параметры агента
        """

        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        checkpoint = {
            'model_state_dict': self.player.model.state_dict(),
            'target_net_state_dict': self.player.target_net.state_dict(),
            'optimizer_state_dict': self.player.optimizer.state_dict(),
            'epsilon': self.player.epsilon,
            'memory': list(self.player.memory),  # Преобразуем deque в list для сохранения
            'stack': self.player.stack,
        }

        torch.save(checkpoint, filepath)

        print(f"Модель сохранена в {filepath}")
        return filepath

    def load_model(self, filename="neural_agent.pth", save_dir="models"):
        """
        Загружает веса модели и параметры агента
        """
        filepath = os.path.join(save_dir, filename)

        if not os.path.exists(filepath):
            print(f"Файл {filepath} не найден!")
            return False

        try:
            checkpoint = torch.load(filepath)

            # Загружаем состояние моделей
            self.player.model.load_state_dict(checkpoint['model_state_dict'])
            self.player.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.player.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Восстанавливаем другие параметры
            self.player.epsilon = checkpoint.get('epsilon', self.player.epsilon)
            self.player.memory = deque(checkpoint.get('memory', []), maxlen=self.player.memory.maxlen)
            self.player.stack = checkpoint.get('stack', self.player.stack)

            StaticLogger.print(f"Модель загружена из {filepath}")
            StaticLogger.print(f"Текущий epsilon: {self.player.epsilon}")
            StaticLogger.print(f"Размер памяти: {len(self.player.memory)}")
            return True

        except Exception as e:
            StaticLogger.print(f"Ошибка при загрузке модели: {e}")
            return False