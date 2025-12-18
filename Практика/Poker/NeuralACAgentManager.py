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
    def __init__(self, player: NeuralAgent):
        super().__init__(player)
        self.episode_data = []
        self.episode_buffer = []

        # --- Гиперпараметры PPO ---
        self.gamma = 0.99
        self.eps_clip = 0.2  # Рекомендуемое значение для clipping (20%)
        self.ppo_epochs = 5  # Сколько раз прогонять один батч через сеть
        self.batch_size = 256  # Увеличили батч для стабильности
        self.entropy_coef = 0.02  # Коэффициент энтропии (исследование)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        StaticLogger.print(f"NeuralACAgentManager using device: {self.device}")

        if hasattr(self.player, 'ac_net'):
            self.player.ac_net.to(self.device)

    def act(self, s_actor: list, s_critic: list, can_check=False, training_mode=False):

        s_actor_tensor = torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0).to(self.device)
        s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.player.ac_net.eval()
        with torch.no_grad():
            action_logits, value = self.player.ac_net(s_actor_tensor, s_critic_tensor)
        self.player.ac_net.train()

        if can_check:
            action_logits[0, 0] = -1e9
        policy_dist = distributions.Categorical(logits=action_logits)

        if training_mode:
            action_tensor = policy_dist.sample()
            action_idx = action_tensor.item()
        else:
            action_idx = torch.argmax(action_logits).item()
            action_tensor = torch.tensor(action_idx).to(self.device)

        log_prob = policy_dist.log_prob(action_tensor).item()
        value_estimate = value.item()

        self.episode_data.append((s_actor, s_critic, action_idx, log_prob, value_estimate))

        self.last_s_actor = s_actor
        self.last_s_critic = s_critic
        self.last_action_idx = action_idx

        return action_idx

    def train_actor_critic(self, final_reward):
        if not self.episode_data:
            return

        s_actors, s_critics, actions, log_probs, _ = zip(*self.episode_data)

        returns = []
        R = final_reward

        for _ in reversed(range(len(self.episode_data))):
            returns.insert(0, R)
            R = R * self.gamma

        for i in range(len(s_actors)):
            self.episode_buffer.append((
                s_actors[i],
                s_critics[i],
                actions[i],
                log_probs[i],
                returns[i]
            ))

        self.episode_data.clear()

        if len(self.episode_buffer) >= self.batch_size:
            self._update_network_ppo()

    def _update_network_ppo(self):

        if not self.episode_buffer:
            return

        s_actors, s_critics, actions, old_log_probs, returns = zip(*self.episode_buffer)

        s_actors = torch.tensor(s_actors, dtype=torch.float32).to(self.device)
        s_critics = torch.tensor(s_critics, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        self.episode_buffer.clear()

        for _ in range(self.ppo_epochs):
            action_logits, values = self.player.ac_net(s_actors, s_critics)
            values = values.squeeze(1)

            policy_dist = distributions.Categorical(logits=action_logits)
            new_log_probs = policy_dist.log_prob(actions)
            dist_entropy = policy_dist.entropy().mean()

            advantages = returns - values.detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy

            self.player.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=0.5)
            self.player.optimizer.step()

        StaticLogger.print(
            f"PPO Update: Loss={loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")
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

        s_actor = [
            hand_strength,
            current_bet_normalized,
            current_stack_normalized,
            pot_normalize,
            1.0 if stage == "preflop" else 0.0,
            1.0 if stage == "flop" else 0.0,
            1.0 if stage == "turn" else 0.0,
            1.0 if stage == "river" else 0.0,
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



    def save_ac_agent(self, filename="neural_ac_agent_Aggressor_small_batch_small_lr.pth", save_dir="models", save_memory=True):
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

            checkpoint = torch.load(filepath, map_location=torch.device('cuda'))

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