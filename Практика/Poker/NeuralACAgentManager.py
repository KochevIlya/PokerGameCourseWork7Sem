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
        self.episode_buffer = []
        self.update_frequency = 50
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def act(self, s_actor: list, s_critic: list, can_check=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.
        """

        s_actor_tensor = torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0)
        s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0)

        self.player.ac_net.eval()
        with torch.no_grad():
            action_logits, value = self.player.ac_net(s_actor_tensor, s_critic_tensor)
        self.player.ac_net.train()

        if can_check:
            action_logits[0, 0] = -1e9

        policy_dist = distributions.Categorical(logits=action_logits)

        action_tensor = policy_dist.sample()
        action_idx = action_tensor.item()

        log_prob = policy_dist.log_prob(action_tensor).item()

        value_estimate = value.item()

        self.episode_data.append((s_actor, s_critic, action_idx, log_prob, value_estimate))

        self.last_s_actor = s_actor
        self.last_s_critic = s_critic
        self.last_action_idx = action_idx

        return action_idx

    def train_actor_critic(self, final_reward):
        """
        Обрабатывает завершенный эпизод:
        1. Считает дисконтированные награды (Returns).
        2. Складывает данные в общий буфер (Experience Replay).
        3. Если буфер полон — запускает обучение.
        """
        if not self.episode_data:
            return

        s_actors, s_critics, actions, _, _ = zip(*self.episode_data)

        returns = []
        R = final_reward

        for _ in reversed(range(len(self.episode_data))):
            returns.insert(0, R)
            R = R * self.player.gamma


        for i in range(len(s_actors)):
            self.episode_buffer.append((
                s_actors[i],
                s_critics[i],
                actions[i],
                returns[i]
            ))

        self.episode_data.clear()

        BATCH_SIZE = 64

        if len(self.episode_buffer) >= BATCH_SIZE:
            self._update_network()

    def _update_network(self):
        """
        Выполняет один шаг градиентного спуска на накопленном батче данных.
        """
        if not self.episode_buffer:
            return

        s_actors, s_critics, actions, returns = zip(*self.episode_buffer)

        s_actors = torch.tensor(s_actors, dtype=torch.float32)
        s_critics = torch.tensor(s_critics, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        returns = torch.tensor(returns, dtype=torch.float32)

        self.episode_buffer.clear()

        self.player.ac_net.train()

        action_logits, values = self.player.ac_net(s_actors, s_critics)

        values = values.squeeze(1)

        policy_dist = distributions.Categorical(logits=action_logits)

        log_probs = policy_dist.log_prob(actions)


        dist_entropy = policy_dist.entropy().mean()

        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()

        critic_loss = F.mse_loss(values, returns)


        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy

        self.player.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=0.5)

        self.player.optimizer.step()


        StaticLogger.print(f"Update: Loss={total_loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")

    def ask_decision(self, s_actor: list, s_critic: list, can_check=False):
        """
        Интерфейс с GameManager. Принимает векторы состояния, вызывает act,
        устанавливает решение игрока и логирует ситуацию.
        """

        action_idx = self.act(s_actor, s_critic, can_check)

        action = ACTIONS[action_idx]
        self.player.set_decision(action)

        StaticLogger.print(f"\nИгрок {self.player.name}")
        StaticLogger.print(f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")

        StaticLogger.print(f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print(f"Ваш выбор: {self.player.decision}")

        return self.player.decision

    def build_state_vectors(self, current_bet_normalized, current_stack_normalized, pot_normalize, community_cards,
                            active_opponents_count, current_decision_value,
                            stage="preflop", all_player_hands=None):
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        stage = STAGES[stage] / len(STAGES)

        s_actor = [
            hand_strength,
            current_bet_normalized,
            current_stack_normalized,
            pot_normalize,
            stage,
            current_decision_value,
            self.decision_value / self.num_bets,
        ]

        all_hand_strengths = []
        for player, hand in all_player_hands:
            strength = HandCalculator.evaluate_hand_strength(hand, community_cards)
            all_hand_strengths.append(strength)

        avg_opp_strength = (sum(all_hand_strengths) - hand_strength) / max(1, len(all_hand_strengths) - 1)

        s_critic = s_actor + [avg_opp_strength]

        return s_actor, s_critic



    def save_ac_agent(self, filename="neural_ac_agent_small_batch_calling_player.pth", save_dir="models", save_memory=True):
        """
        Сохраняет состояние NeuralACAgent (Actor-Critic)

        Args:
            self: экземпляр NeuralACAgent
            filename: имя файла для сохранения
            save_dir: директория для сохранения
            save_memory: сохранять ли память (может быть большим)
        """
        try:

            os.makedirs(save_dir, exist_ok=True)

            filepath = os.path.join(save_dir, filename)

            checkpoint = {
                'ac_net_state_dict': self.player.ac_net.state_dict(),
                'optimizer_state_dict': self.player.optimizer.state_dict(),

                'gamma': self.player.gamma,
                'actor_size': self.player.ac_net.actor_net[0].in_features if hasattr(self.player, 'ac_net') else 8,
                'critic_size': self.player.ac_net.critic_net[0].in_features if hasattr(self.player, 'ac_net') else 9,
                'action_size': self.player.ac_net.actor_net[-1].out_features if hasattr(self.player, 'ac_net') else 3,

                # Состояние агента
                'name': self.player.name,
                'stack': self.player.stack,

                # Метаданные
                'model_type': 'ActorCritic',
            }

            if save_memory and hasattr(self.player, 'memory'):
                checkpoint['memory'] = list(self.player.memory)
                checkpoint['memory_size'] = len(self.player.memory)

            torch.save(checkpoint, filepath)
            print(f"[✅] NeuralACAgent '{self.player.name}' сохранен в {filepath}")
            print(f"    Память: {checkpoint.get('memory_size', 0)} записей")

            return filepath

        except Exception as e:
            print(f"[❌] Ошибка при сохранении агента: {e}")
            return None

    def load_ac_agent(self, filename="neural_ac_agent.pth", save_dir="models",
                      load_memory=True, strict=True):
        """
        Загружает состояние NeuralACAgent

        Args:
            self: экземпляр NeuralACAgent для загрузки данных
            filename: имя файла для загрузки
            save_dir: директория с файлами
            load_memory: загружать ли память
            strict: строгая загрузка весов модели
        """
        try:
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                print(f"[❌] Файл {filepath} не найден!")
                return False

            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

            if checkpoint.get('model_type') != 'ActorCritic':
                print("[⚠️] Внимание: Загружается не Actor-Critic модель")

            self.player.ac_net.load_state_dict(checkpoint['ac_net_state_dict'], strict=strict)

            if 'optimizer_state_dict' in checkpoint:
                self.player.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'gamma' in checkpoint:
                self.player.gamma = checkpoint['gamma']
            if load_memory and 'memory' in checkpoint and hasattr(self.player, 'memory'):
                self.player.memory = deque(checkpoint['memory'], maxlen=self.player.memory.maxlen)
                print(f"    Загружено {len(self.player.memory)} записей в память")

            if 'stack' in checkpoint:
                self.player.stack = checkpoint['stack']

            print(f"[✅] NeuralACAgent загружен из {filepath}")
            print(f"    Имя: {checkpoint.get('name', 'Unknown')}")
            print(f"    Gamma: {self.player.gamma}")

            return True

        except Exception as e:
            print(f"[❌] Ошибка при загрузке агента: {e}")
            return False