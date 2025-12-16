import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from collections import deque
from .PlayerManager import PlayerManager
from .HandCalculator import HandCalculator
from .NeuralAgent import *
from .Logger import *
import torch.distributions as distributions

STAGES ={ "preflop": 0, "flop": 1, "turn": 2, "river": 3, }
ACTIONS = { 0 : "fold", 1 : "raise", 2 : "call", }

class NeuralACAgentManager(PlayerManager):
    def __init__(self, player:NeuralAgent):
        super().__init__(player)
        self.episode_data = []
        self.episode_buffer = []  # Буфер для полных эпизодов
        self.update_frequency = 50  # Обновлять веса каждые 50 раздач
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def act(self, s_actor: list, s_critic: list, can_check=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.
        """


        # 1. Конвертация состояний в тензоры
        # unsqueeze(0) добавляет размерность батча=1
        s_actor_tensor = torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0)
        s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0)

        # 2. Forward pass: Actor выбирает, Critic оценивает
        # action_logits - логиты (необработанные оценки) для действий [1, action_size]
        # value - оценка V(s) [1, 1]
        self.player.ac_net.eval()  # Переводим сеть в режим исполнения
        with torch.no_grad():
            action_logits, value = self.player.ac_net(s_actor_tensor, s_critic_tensor)
        self.player.ac_net.train()  # Возвращаем в режим обучения

        if can_check:
            action_logits[0, 0] = -1e9

        # 3. Сэмплирование действия (Exploration в A2C)
        # Создаем категориальное распределение на основе логитов Actor'а
        policy_dist = distributions.Categorical(logits=action_logits)

        # Выбираем действие путем сэмплирования
        action_tensor = policy_dist.sample()
        action_idx = action_tensor.item()

        # 4. Сбор данных для обучения

        # Логарифм вероятности выбранного действия
        log_prob = policy_dist.log_prob(action_tensor).item()

        # Оценка состояния (V(s)) от Critic'а
        value_estimate = value.item()

        # Сохраняем все данные, необходимые для расчета Advantage и Loss
        # (S_actor, S_critic, Action, Log_Prob, V(s))
        self.episode_data.append((s_actor, s_critic, action_idx, log_prob, value_estimate))

        # Сохраняем последнее состояние/действие для логирования в ask_decision
        self.last_s_actor = s_actor
        self.last_s_critic = s_critic
        self.last_action_idx = action_idx

        return action_idx

    def train_actor_critic(self, final_reward):
        """
        Выполняет один шаг обучения A2C для завершенного раунда.
        Использует S_actor для Actor Loss и S_critic для Critic Loss.
        """
        if not self.episode_data:
            return

        # 1. Разбираем сохраненные данные эпизода
        # (S_actor, S_critic, Action, Log_Prob, V(s))
        s_actors, s_critics, actions, log_probs, values = zip(*self.episode_data)

        # Конвертация в тензоры
        s_actors = torch.tensor(s_actors, dtype=torch.float32)
        s_critics = torch.tensor(s_critics, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # values - оценки V(s) из первого прохода (их можно использовать, но лучше пересчитать)

        # 2. Расчет Дисконтированных Наград (Returns)
        # Нормализация финальной награды (как в вашем коде)


        returns = []
        R = final_reward

        # Добавляем финальный V(s) последнего состояния для более точного расчета
        # Если раунд закончился фолдом, используем 0.0, иначе используем V(s) последнего состояния.
        # В этой упрощенной версии просто используем R.

        # Обратная итерация для расчета дисконтированных наград
        for i in reversed(range(len(self.episode_data))):
            # В A2C награда R в основном состоит только из финального выигрыша/проигрыша,
            # так как промежуточные награды r_step были 0.
            returns.append(R)
            R = R * self.player.gamma

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)

        # 3. Получение V(s) (Value Estimates) с помощью Critic Net и S_critic
        # Получаем V(s) для всех состояний эпизода с текущими весами Critic'а
        action_logits, critic_values = self.player.ac_net(s_actors, s_critics)
        critic_values = critic_values.squeeze(1)  # [BatchSize]

        # 4. Вычисляем log_probs заново для графа
        policy_dist = distributions.Categorical(logits=action_logits)
        log_probs_new = policy_dist.log_prob(actions)

        # 4. Вычисляем log_probs заново для графа
        policy_dist = distributions.Categorical(logits=action_logits)
        log_probs_new = policy_dist.log_prob(actions)

        # 5. Loss
        advantage = returns - critic_values.detach()
        actor_loss = -(
                    log_probs_new * advantage).mean()  # advantage без detach, если хотим учить через него (но обычно detach)
        critic_loss = F.mse_loss(critic_values, returns)
        entropy_loss = policy_dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.1 * entropy_loss

        # 6. Оптимизация
        self.player.optimizer.zero_grad()
        total_loss.backward()
        # Совет: добавьте градиентный клиппинг для стабильности AC в покере
        torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=1.0)
        self.player.optimizer.step()

        self.episode_data.clear()

    def ask_decision(self, s_actor: list, s_critic: list, can_check=False):
        """
        Интерфейс с GameManager. Принимает векторы состояния, вызывает act,
        устанавливает решение игрока и логирует ситуацию.
        """

        # 1. Получаем индекс действия из Actor-Critic
        action_idx = self.act(s_actor, s_critic, can_check)

        # 2. Преобразуем индекс в строковое решение (fold, raise, call)
        action = ACTIONS[action_idx]
        self.player.set_decision(action)

        # 3. Логирование (согласно вашему исходному коду)
        StaticLogger.print(f"\nИгрок {self.player.name}")
        StaticLogger.print(f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        # Обновление: Убедитесь, что best_hand обновлен до вызова этого метода
        StaticLogger.print(f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print(f"Ваш выбор: {self.player.decision}")

        return self.player.decision

    def build_state_vectors(self, current_bet_normalized, current_stack_normalized, pot_normalize, community_cards,
                            active_opponents_count ,
                            stage="preflop", all_player_hands=None):
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        stage = STAGES[stage] / len(STAGES)

        s_actor = [
            hand_strength,
            current_bet_normalized,
            current_stack_normalized,
            pot_normalize,
            stage,
            active_opponents_count
        ]

        all_hand_strengths = []
        for player, hand in all_player_hands:
            strength = HandCalculator.evaluate_hand_strength(hand, community_cards)
            all_hand_strengths.append(strength)

        avg_opp_strength = (sum(all_hand_strengths) - hand_strength) / max(1, len(all_hand_strengths) - 1)

        s_critic = s_actor + [avg_opp_strength]

        return s_actor, s_critic

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