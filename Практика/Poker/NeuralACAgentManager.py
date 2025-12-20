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
        self.epsilon = 0.1
        self.epsilon_des = 0.9995
        self.min_epsilon = 0.01
        self.total_loss_buffer = []
        self.actor_loss_buffer = []
        self.critic_loss_buffer = []
        self.action_dim = 3
        self.history_len = 10

        self.history_buffer = deque(maxlen=self.history_len)
        self.reset_history()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        StaticLogger.print(f"NeuralACAgentManager using device: {self.device}")

        if hasattr(self.player, 'ac_net'):
            self.player.ac_net.to(self.device)

    def reset_history(self):
        """Очищает историю (заполняет нулями) перед новой игрой"""
        self.history_buffer.clear()
        for _ in range(self.history_len):
            self.history_buffer.append(np.zeros(self.action_dim, dtype=np.float32))

    def record_opponent_action(self, action_type):
        """
        Вызывай этот метод из GameManager, когда оппонент делает ход!
        action_type: 0-fold, 1-check, 2-call, 3-raise (пример)
        """
        vec = np.zeros(self.action_dim, dtype=np.float32)

        # Пример кодирования: [IsFold, IsCheck, IsCall, IsRaise, Amount]
        # Допустим mapping action_type: 0->Fold, 1->Check, 2->Call, 3->Raise

        if 0 <= action_type < 2:
            vec[action_type] = 1.0

        self.history_buffer.append(vec)

    def get_history_tensor(self):
        """Превращает deque в тензор [1, 10, 5]"""
        h_array = np.array(self.history_buffer)
        return torch.tensor(h_array, dtype=torch.float32).unsqueeze(0).to(self.device)

    def act(self, s_actor: list, s_critic: list, can_check=False, training_mode=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.
        """

        s_actor_tensor = torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0).to(self.device)
        s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0).to(self.device)

        history_tensor = self.get_history_tensor()

        self.player.ac_net.eval()
        with torch.no_grad():
            action_logits, value = self.player.ac_net(s_actor_tensor, s_critic_tensor, history_tensor)
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

        log_prob = policy_dist.log_prob(action_tensor).item()

        value_estimate = value.item()

        current_history_snapshot = np.array(self.history_buffer)

        self.episode_data.append((s_actor, s_critic, current_history_snapshot, action_idx, log_prob, value_estimate))

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

        s_actors, s_critics, histories, actions, _, _ = zip(*self.episode_data)

        returns = []
        R = final_reward

        for _ in reversed(range(len(self.episode_data))):
            returns.insert(0, R)
            R = R * self.player.gamma


        for i in range(len(s_actors)):
            NNData.add_buffer((
                s_actors[i],
                s_critics[i],
                histories[i],
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

        s_actors, s_critics, histories, actions, returns = zip(*self.episode_buffer)

        # ИСПРАВЛЕНИЕ: Сначала конвертируем tuple/list в numpy array, потом в tensor
        s_actors = torch.tensor(np.array(s_actors), dtype=torch.float32).to(self.device)
        s_critics = torch.tensor(np.array(s_critics), dtype=torch.float32).to(self.device)
        histories = torch.tensor(np.array(histories), dtype=torch.float32).to(self.device)

        # Для actions и returns обычно приходят просто числа, но np.array тоже не повредит для надежности
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        NNData.clear()

        self.player.ac_net.train()

        action_logits, values = self.player.ac_net(s_actors, s_critics, histories)

        values = values.squeeze(1)

        policy_dist = distributions.Categorical(logits=action_logits)

        log_probs = policy_dist.log_prob(actions)


        dist_entropy = policy_dist.entropy().mean()

        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()

        critic_loss = F.huber_loss(values, returns, delta=1.0)

        self.epsilon = max(self.epsilon * self.epsilon_des, self.min_epsilon)
        total_loss = actor_loss + 0.5 * critic_loss - self.epsilon * dist_entropy

        self.player.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=0.5)

        self.player.optimizer.step()


        StaticLogger.print(f"Update: Loss={total_loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")
        NNData.add_loss_actor(actor_loss.item())
        NNData.add_loss_critic(critic_loss.item())
        NNData.add_loss_buffer(total_loss.item())


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
        s_preflop = 1.0 if stage == "preflop" else 0.0
        s_flop = 1.0 if stage == "flop" else 0.0
        s_turn = 1.0 if stage == "turn" else 0.0
        s_river = 1.0 if stage == "river" else 0.0
        s_actor = [
            hand_strength,
            current_bet_normalized,
            current_stack_normalized,
            pot_normalize,
            s_preflop,
            s_flop,
            s_turn,
            s_river,
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



    def save_ac_agent(self, filename="neural_ac_agent_for_course_LSTM_after_calling.pth", save_dir="models", save_memory=True):
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
            lstm_hidden = self.player.ac_net.actor_lstm.hidden_size if hasattr(self.player, 'ac_net') else 32

            checkpoint = {
                'ac_net_state_dict': self.player.ac_net.state_dict(),
                'optimizer_state_dict': self.player.optimizer.state_dict(),

                'gamma': self.player.gamma,
                # Обновляем получение размеров, так как структура изменилась, но логика та же
                'actor_size': self.player.ac_net.actor_net[0].in_features - lstm_hidden if hasattr(self.player,
                                                                                                   'ac_net') else 10,
                'critic_size': self.player.ac_net.critic_net[0].in_features - lstm_hidden if hasattr(self.player,
                                                                                                     'ac_net') else 11,
                'action_size': self.player.ac_net.actor_net[-1].out_features if hasattr(self.player, 'ac_net') else 3,

                # Сохраняем параметры LSTM
                'lstm_hidden': lstm_hidden,
                'history_len': self.history_len,
                'action_vector_size': self.action_dim,

                'name': self.player.name,
                'stack': self.player.stack,
                'model_type': 'ActorCritic_DualLSTM',  # Можно поменять тип для ясности
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