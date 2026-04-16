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
from .NNData import NNData
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
        self.entropy_coef = 0.05
        self.entropy_des = 0.99
        self.min_entropy = 0.01

        self.total_loss_buffer = []
        self.actor_loss_buffer = []
        self.critic_loss_buffer = []
        self.entropy_buffer = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        StaticLogger.print(f"NeuralACAgentManager using device: {self.device}")

        if hasattr(self.player, 'ac_net'):
            self.player.ac_net.to(self.device)

    def act(self, s_actor: list, s_critic: list, can_check=False, training_mode=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.
        """

        s_actor_tensor = torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0).to(self.device)
        s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.player.ac_net.eval()
        with torch.no_grad():
            action_logits, value = self.player.ac_net(s_actor_tensor, s_critic_tensor)
        self.player.ac_net.train()

        if can_check:
            action_logits[0, 0] = -1e9
        if training_mode:
            policy_dist = distributions.Categorical(logits=action_logits)
            action_tensor = policy_dist.sample()
            action_idx = action_tensor.item()
        else:
            action_idx = torch.argmax(action_logits).item()
            policy_dist = distributions.Categorical(logits=action_logits)
            action_tensor = torch.tensor(action_idx).to(self.device)

        NNData.record_action(action_idx)
        StaticLogger.print(f'A_logits: {action_logits}\n')
        log_prob = policy_dist.log_prob(action_tensor).item()
        StaticLogger.print(f'A_log_prob: {log_prob}\n')
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
            StaticLogger.print(f"\nTrain Param S_actor: {s_actors[i]} s_critic: {s_critics[i]} action: {actions[i]} Rewards: {returns[i]}\n")
            NNData.add_buffer((
                s_actors[i],
                s_critics[i],
                actions[i],
                returns[i]
            ))

        self.episode_data.clear()


        if NNData.is_full():
            self.episode_buffer = NNData.get_buffer()
            print(f"Target Network updated!!!, Length: {len(self.episode_buffer)}")
            self._update_network()


    def _update_network(self):
        """
        Выполняет один шаг градиентного спуска на накопленном батче данных.
        """
        if not self.episode_buffer:
            return
        StaticLogger.print(f"\nUpdate Network\n")
        s_actors, s_critics, actions, returns = zip(*self.episode_buffer)

        s_actors = torch.tensor(s_actors, dtype=torch.float32).to(self.device)
        s_critics = torch.tensor(s_critics, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        NNData.clear()

        self.player.ac_net.train()

        action_logits, values = self.player.ac_net(s_actors, s_critics)

        with torch.no_grad():
            # Берем средние значения логитов по всему батчу для каждого действия
            # (Допустим: 0-Fold, 1-Raise, 2-Call)
            mean_logits = action_logits.mean(dim=0)
            max_logits = action_logits.max(dim=0)[0]
            min_logits = action_logits.min(dim=0)[0]

            StaticLogger.print(f"\n--- [LOGITS RAW DATA] ---")
            StaticLogger.print(f"Mean Logits (F/R/C): {mean_logits.cpu().numpy()}")
            StaticLogger.print(f"Max Logits: {max_logits.cpu().numpy()}")
            StaticLogger.print(f"Min Logits: {min_logits.cpu().numpy()}")

            # Посмотрим на разброс (дистанцию) между самым популярным и непопулярным действием
            logit_spread = max_logits.max() - min_logits.min()
            StaticLogger.print(f"Logit Spread (Max-Min): {logit_spread.item():.4f}")
            StaticLogger.print(f"--------------------------\n")

        values = values.squeeze(1)

        policy_dist = distributions.Categorical(logits=action_logits)

        log_probs = policy_dist.log_prob(actions)

        dist_entropy = policy_dist.entropy().mean()

        NNData.add_entropy(dist_entropy.item())


        advantage = returns - values.detach()
        StaticLogger.print(f"Advantage: {advantage}\n")

        if advantage.size(0) > 1: # Проверка, что в батче больше одного элемента
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        StaticLogger.print(f"Normalized Advantage: {advantage.mean().item():.6f}\n")
        actor_loss = -(log_probs * advantage).mean()

        critic_loss = F.mse_loss(values, returns)

        self.entropy_coef = max(self.entropy_coef * self.entropy_des, self.min_entropy)
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy

        self.player.optimizer.zero_grad()
        total_loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=1.0)
        StaticLogger.print(f"Gradient Norm: {total_norm:.4f}")
        self.player.optimizer.step()

        with torch.no_grad():
            # Сигнал от награды (насколько сильно мы хотим закрепить действия)
            reward_signal = actor_loss.item()
            # Сигнал от энтропии (насколько сильно мы заставляем сеть хаотить)
            entropy_contribution = -(self.entropy_coef * dist_entropy).item()
            # Соотношение (если оно меньше 1, значит энтропия важнее наград)
            signal_ratio = abs(reward_signal) / (abs(entropy_contribution) + 1e-8)

        action_freq = NNData.get_action_freq()
        action_sum = sum(action_freq.values())
        action_freq = [i / action_sum for i in action_freq.values()]

        StaticLogger.print(f"\n--- [DEBUG LEARNING] ---")
        StaticLogger.print(f"Actor (Reward) Signal: {reward_signal:.6f}")
        StaticLogger.print(f"Entropy (Chaos) Signal: {entropy_contribution:.6f}")
        StaticLogger.print(f"Signal Ratio (Rew/Ent): {signal_ratio:.4f}")
        StaticLogger.print(f"Mean Advantage: {advantage.mean().item():.6f}")
        StaticLogger.print(f"Current Epsilon: {self.entropy_coef:.4f}")
        StaticLogger.print(f"Action Frequency: {action_freq}")
        StaticLogger.print(f"Num Actions: {action_sum}")
        StaticLogger.print(f"Current sleazy win: {NNData.get_sleazy_win()}")
        StaticLogger.print(f"------------------------\n")

        print(f"\n--- [DEBUG LEARNING] ---")
        print(f"Actor (Reward) Signal: {reward_signal:.6f}")
        print(f"Entropy (Chaos) Signal: {entropy_contribution:.6f}")
        print(f"Signal Ratio (Rew/Ent): {signal_ratio:.4f}")
        print(f"Mean Advantage: {advantage.mean().item():.6f}")
        print(f"Current Epsilon: {self.entropy_coef:.4f}")
        print(f"Action Frequency: {action_freq}")
        print(f"Num Actions: {action_sum}")
        print(f"Current sleazy win: {NNData.get_sleazy_win()}")
        print(f"------------------------\n")

        StaticLogger.print(f"Update: Loss={total_loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")
        StaticLogger.print(f"Entropy: {dist_entropy.item():.4f}")

        stats = self.validate_hand_values() # Метод должен возвращать gap

        # 2. Сохраняем gap в историю
        NNData.add_value_gap(stats)


        # 3. Фиксируем стиль игры за этот батч
        NNData.commit_action_freq()

        NNData.add_loss_actor(actor_loss.item())
        NNData.add_loss_critic(critic_loss.item())
        NNData.add_loss_buffer(total_loss.item())


    def ask_decision(self, s_actor: list, s_critic: list, can_check=False):
        """
        Интерфейс с GameManager. Принимает векторы состояния, вызывает act,
        устанавливает решение игрока и логирует ситуацию.
        """

        action_idx = self.act(s_actor, s_critic, can_check, training_mode=True)

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



    def save_ac_agent(self, filename="neural_ac_agent_for_course.pth", save_dir="models", save_memory=True):
        """
        Сохраняет состояние NeuralACAgent (Actor-Critic) с учетом новой архитектуры.
        """
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            filepath = os.path.join(save_dir, filename)
            net = self.player.ac_net

            # Теперь мы обращаемся к конкретным слоям вместо actor_net/critic_net
            checkpoint = {
                'ac_net_state_dict': net.state_dict(),
                'optimizer_state_dict': self.player.optimizer.state_dict(),
                'gamma': self.player.gamma,

                # Извлечение размеров из новых названий слоев
                'actor_size': net.actor_fc1.in_features,
                'critic_size': net.critic_fc1.in_features,
                'action_size': net.actor_head.out_features,

                'name': self.player.name,
                'stack': self.player.stack,
                'model_type': 'ActorCritic',
            }

            if save_memory and hasattr(self.player, 'memory'):
                checkpoint['memory'] = list(self.player.memory)
                checkpoint['memory_size'] = len(self.player.memory)

            torch.save(checkpoint, filepath)
            print(f"[✅] Модель сохранена: {filepath}")
            return filepath

        except Exception as e:
            print(f"[❌] Ошибка сохранения: {e}")
            return None

    def load_ac_agent(self, filename="neural_ac_agent.pth", save_dir="models",
                      load_memory=True, strict=True):
        """
        Загружает состояние NeuralACAgent.
        """
        try:
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                print(f"[❌] Файл {filepath} не найден!")
                return False

            # Загружаем на то устройство, которое используется сейчас (CPU или CUDA)
            checkpoint = torch.load(filepath, map_location=self.device)

            if checkpoint.get('model_type') != 'ActorCritic':
                print("[⚠️] Внимание: Тип модели в файле не совпадает!")

            # Загружаем веса
            self.player.ac_net.load_state_dict(checkpoint['ac_net_state_dict'], strict=strict)

            #Откручивание головы для логитов
            # with torch.no_grad():
            #     nn.init.orthogonal_(self.player.ac_net.actor_head.weight, gain=0.01)
            #     nn.init.constant_(self.player.ac_net.actor_head.bias, 0)

            if 'optimizer_state_dict' in checkpoint:
                self.player.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'gamma' in checkpoint:
                self.player.gamma = checkpoint['gamma']

            if load_memory and 'memory' in checkpoint and hasattr(self.player, 'memory'):
                self.player.memory.clear()
                self.player.memory.extend(checkpoint['memory'])
                print(f"    Загружено {len(self.player.memory)} записей в память")

            if 'stack' in checkpoint:
                self.player.stack = checkpoint['stack']

            print(f"[✅] Агент {checkpoint.get('name')} успешно загружен")
            return True

        except Exception as e:
            print(f"[❌] Ошибка загрузки: {e}")
            return False

    def validate_hand_values(self):
        """Проверка того, насколько Критик различает силу рук (Value Gap)"""
        self.player.ac_net.eval()
        with torch.no_grad():
            # 1. Тест "Сильная рука" (Тузы AA)
            # Создаем вектор размера critic_size, где 0-й элемент (hand_strength) = 0.9
            s_aa = np.zeros((1, self.player.critic_size), dtype=np.float32)
            s_aa[0, 0] = 0.9  # Сила нашей руки
            s_aa[0, -1] = 0.1 # Сила руки оппонента (последний элемент)

            # Нам нужен и s_actor (хотя мы смотрим только на критика), создадим его пустым
            s_actor_empty = torch.zeros((1, self.player.actor_size)).to(self.device)
            s_critic_aa = torch.tensor(s_aa).to(self.device)

            # Получаем logits и value. Нас интересует только второе значение.
            _, v_aa = self.player.ac_net(s_actor_empty, s_critic_aa)

            # 2. Тест "Слабая рука" (72 разномастные)
            s_72 = np.zeros((1, self.player.critic_size), dtype=np.float32)
            s_72[0, 0] = 0.1
            s_72[0, -1] = 0.9
            s_critic_72 = torch.tensor(s_72).to(self.device)

            _, v_72 = self.player.ac_net(s_actor_empty, s_critic_72)

            gap = v_aa.item() - v_72.item()

        self.player.ac_net.train()

        # Логируем результат
        msg = f"🧪 [Value Gap] AA: {v_aa.item():.4f} vs 72o: {v_72.item():.4f} | GAP: {gap:.4f}"
        print(msg)
        StaticLogger.print( msg) # Отправляем в лог обучения

        return gap