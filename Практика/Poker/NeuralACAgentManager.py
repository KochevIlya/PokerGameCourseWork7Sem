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

        # --- Rolling статистика обучения ---
        self.results_buffer = deque(maxlen=500)    # (is_winner, net_profit, pot_size)
        self.loss_log = deque(maxlen=100)          # (actor_loss, critic_loss, entropy)
        self.game_counter = 0
        self.epoch_interval = 500
        self.last_update_games = 0
        self.big_blind = 10

        self.history_buffer = deque(maxlen=self.history_len)
        self.reset_history()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        StaticLogger.print_to("training", f"NeuralACAgentManager using device: {self.device}")

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

    def record_game_result(self, is_winner, net_profit, pot_size, big_blind):
        """
        Записывает результат раздачи для rolling-статистики.
        Memory Guard: deque(maxlen=500) автоматически ограничивает память.
        """
        self.game_counter += 1
        self.big_blind = big_blind
        self.results_buffer.append((is_winner, net_profit, pot_size))

    def get_training_stats(self):
        """
        Возвращает текущую статистику обучения.
        Unit Conversion: AvgPot конвертируется в Big Blinds.
        """
        if not self.results_buffer:
            return None

        total = len(self.results_buffer)
        wins = sum(1 for r in self.results_buffer if r[0])
        winrate = wins / total * 100
        avg_profit = sum(r[1] for r in self.results_buffer) / total

        # Unit Conversion: фишки → Big Blinds
        avg_pot_bb = sum(r[2] for r in self.results_buffer) / total / self.big_blind

        if self.loss_log:
            avg_a = sum(l[0] for l in self.loss_log) / len(self.loss_log)
            avg_c = sum(l[1] for l in self.loss_log) / len(self.loss_log)
            avg_e = sum(l[2] for l in self.loss_log) / len(self.loss_log)
        else:
            avg_a, avg_c, avg_e = 0, 0, 0

        return winrate, avg_a, avg_c, avg_e, avg_pot_bb

    def print_stats_if_needed(self, epoch):
        """
        Выводит статистику каждые epoch_interval игр.
        Формат: [Epoch X] Winrate(Last 500): 54% | Loss(A/C): 0.021/0.14 | Entropy: 0.65 | AvgPot: 150BB
        """
        games_since_last = self.game_counter - self.last_update_games
        if games_since_last >= self.epoch_interval:
            stats = self.get_training_stats()
            if stats:
                winrate, avg_a, avg_c, avg_e, avg_pot_bb = stats
                msg = (f"[Epoch {epoch}] Winrate(Last 500): {winrate:.0f}% | "
                       f"Loss(A/C): {avg_a:.3f}/{avg_c:.2f} | "
                       f"Entropy: {avg_e:.2f} | AvgPot: {avg_pot_bb:.0f}BB")
                print(f"\n\U0001f4ca {msg}")
                StaticLogger.print_to("training", msg)
            self.last_update_games = self.game_counter

    def run_validation(self, num_games_val=100):
        """
        Контрольный срез: проверка качества модели без обучения.
        Validation Safety: eval() + no_grad() гарантируют, что тестовые игры
        НЕ подмешиваются в градиенты основного обучения.
        """
        from .Deck import Deck
        from .HandCalculator import HandCalculator

        # === VALIDATION SAFETY ===
        self.player.ac_net.eval()

        wins = 0
        with torch.no_grad():
            for g in range(num_games_val):
                deck = Deck()
                deck.shuffle()
                hero_cards = [deck.dealcard(), deck.dealcard()]

                # Оценка: hero vs случайный оппонент (префлоп hand_strength)
                hero_strength = HandCalculator.evaluate_hand_strength(hero_cards, [])
                opp_cards = [deck.dealcard(), deck.dealcard()]
                opp_strength = HandCalculator.evaluate_hand_strength(opp_cards, [])

                if hero_strength >= opp_strength:
                    wins += 1

        # === Возвращаем в режим обучения ===
        self.player.ac_net.train()

        winrate = wins / num_games_val * 100
        val_msg = f"\U0001f9ea [Validation] Preflop strength check: {winrate:.1f}% ({wins}/{num_games_val})"
        print(val_msg)
        StaticLogger.print_to("validation", val_msg)
        return winrate

    def act(self, s_actor: list, s_critic: list, can_check=False, training_mode=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.
        """

        # Сохраняем hidden states НАЧАЛА шага (для обучения)
        init_actor_h = self.player.actor_hidden
        init_critic_h = self.player.critic_hidden

        s_actor_tensor = (torch.tensor(s_actor, dtype=torch.float32).unsqueeze(0)
                          .to(self.device))
        s_critic_tensor = (torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0)
                           .to(self.device))

        history_tensor = self.get_history_tensor()

        self.player.ac_net.eval()
        with torch.no_grad():
            action_logits, value, next_actor_h, next_critic_h = self.player.ac_net(
                s_actor_tensor, s_critic_tensor, history_tensor,
                actor_hidden=self.player.actor_hidden,
                critic_hidden=self.player.critic_hidden
            )
        self.player.ac_net.train()

        if next_actor_h is not None:
            self.player.actor_hidden = (next_actor_h[0].detach(), next_actor_h[1].detach())
            self.player.critic_hidden = (next_critic_h[0].detach(), next_critic_h[1].detach())

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

        self.episode_data.append((s_actor, s_critic, current_history_snapshot,
                                  action_idx, log_prob, value_estimate,
                                  init_actor_h, init_critic_h))

        self.last_s_actor = s_actor
        self.last_s_critic = s_critic
        self.last_action_idx = action_idx

        return action_idx

    def train_actor_critic(self, final_reward):
        """
        Обрабатывает завершенный эпизод (одну раздачу):
        1. Считает дисконтированные награды (Returns).
        2. Формирует траекторию (trajectory dict).
        3. Складывает траекторию в общий буфер (NNData).
        4. Если буфер полон — запускает обучение.
        """
        if not self.episode_data:
            return

        T = len(self.episode_data)

        # Считаем returns (обратно от конца к началу)
        returns = []
        R = final_reward
        for _ in reversed(range(T)):
            returns.insert(0, R)
            R *= self.player.gamma

        # Распаковываем все поля из episode_data
        s_actors = [d[0] for d in self.episode_data]
        s_critics = [d[1] for d in self.episode_data]
        histories = [d[2] for d in self.episode_data]
        actions = [d[3] for d in self.episode_data]
        init_actor_hs = [d[6] for d in self.episode_data]
        init_critic_hs = [d[7] for d in self.episode_data]

        # Формируем траекторию — одну полную раздачу
        trajectory = {
            's_actors': s_actors,
            's_critics': s_critics,
            'histories': histories,
            'actions': actions,
            'returns': returns,
            'init_actor_hs': init_actor_hs,
            'init_critic_hs': init_critic_hs,
        }

        NNData.add_episode(trajectory)

        self.episode_data.clear()

        if NNData.is_full():
            total_steps = sum(len(traj['actions']) for traj in NNData.episode_buffer)
            StaticLogger.print_to("training", f"Target Network updated!!!, Hands: {len(NNData.episode_buffer)}, Steps: {total_steps}")
            self._update_network()


    def _update_network(self):
        """
        Обучает сеть на накопленных траекториях (целых раздачах).
        - Каждая раздача обрабатывается ОДНИМ forward() вызовом
        - Градиенты суммируются по всем раздачам
        - ОДИН optimizer.step() в конце
        """
        trajectories = NNData.get_buffer()

        # Guard: проверяем, что есть хотя бы одна траектория
        if not trajectories or len(trajectories) == 0:
            StaticLogger.print_to("training", "[WARN] _update_network called with empty buffer, skipping.")
            return

        self.player.ac_net.train()

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_trajectories = 0

        for traj_idx, traj in enumerate(trajectories):
            s_actors = traj['s_actors']
            s_critics = traj['s_critics']
            histories = traj['histories']
            actions = traj['actions']
            returns = traj['returns']

            # Guard: пропускаем битые траектории с логированием
            if not actions:
                StaticLogger.print_to(
                    "training",
                    f"[WARN] Trajectory #{traj_idx} has no actions (len=0), skipping. "
                    f"s_actors={len(s_actors)}, returns={len(returns)}"
                )
                continue

            # --- Конвертация в тензоры [Seq_Len, ...] ---
            s_actors_t = torch.tensor(np.array(s_actors), dtype=torch.float32).to(self.device)
            s_critics_t = torch.tensor(np.array(s_critics), dtype=torch.float32).to(self.device)
            histories_t = torch.tensor(np.array(histories), dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

            # Нормализация returns внутри траектории
            if returns_t.numel() > 1 and returns_t.std() > 1e-7:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # --- ОДИН forward() на ВСЮ раздачу ---
            # hidden=None → LSTM стартует с нуля (корректно: новая раздача начинается с reset)
            action_logits, values, _, _ = self.player.ac_net(
                s_actors_t, s_critics_t, histories_t,
                actor_hidden=None, critic_hidden=None
            )

            values = values.squeeze(-1)  # [Seq_Len]

            # --- Вычисление loss ---
            dist = distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions_t)    # [Seq_Len]
            entropy = dist.entropy().mean()

            advantage = returns_t - values.detach()  # [Seq_Len]

            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.huber_loss(values, returns_t, delta=1.0)

            # Накапливаем для логирования
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            num_trajectories += 1

            # Записываем loss в лог для статистики
            self.loss_log.append((actor_loss.item(), critic_loss.item(), entropy.item()))

            # Накапливаем градиенты (не вызываем zero_grad между траекториями!)
            total_loss += actor_loss + 0.5 * critic_loss - self.epsilon * entropy

        # Guard: защита от ZeroDivisionError
        if num_trajectories == 0:
            StaticLogger.print_to("training", "[WARN] No valid trajectories to train on, skipping update.")
            NNData.clear()
            return

        # --- ОДИН optimizer.step() для ВСЕГО батча ---
        total_loss = total_loss / num_trajectories

        self.player.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=0.5)
        self.player.optimizer.step()

        # Очищаем буфер ПОСЛЕ успешного обновления
        NNData.clear()

        # Логирование (средние значения на одну раздачу)
        StaticLogger.print_to(
            "loss",
            f"Update ({num_trajectories} hands): "
            f"Loss={total_loss.item():.4f}, "
            f"Actor={total_actor_loss / num_trajectories:.4f}, "
            f"Critic={total_critic_loss / num_trajectories:.4f}"
        )
        NNData.add_loss_actor(total_actor_loss / num_trajectories)
        NNData.add_loss_critic(total_critic_loss / num_trajectories)
        NNData.add_loss_buffer(total_loss.item())

        print(f"✅ [NETWORK UPDATED] {num_trajectories} hands processed, Loss={total_loss.item():.4f}")


    def ask_decision(self, s_actor: list, s_critic: list, can_check=False):
        """
        Интерфейс с GameManager. Принимает векторы состояния, вызывает act,
        устанавливает решение игрока и логирует ситуацию.
        """

        action_idx = self.act(s_actor, s_critic, can_check)

        action = ACTIONS[action_idx]
        self.player.set_decision(action)

        StaticLogger.print_to("decisions", f"\nИгрок {self.player.name}")
        StaticLogger.print_to("decisions", f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print_to("decisions", f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")

        StaticLogger.print_to("decisions", f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print_to("decisions", f"Ваш выбор: {self.player.decision}")

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
                # Прямые параметры из агента (не вычисляем обратно из слоёв!)
                'actor_size': self.player.actor_size if hasattr(self.player, 'actor_size') else 10,
                'critic_size': self.player.critic_size if hasattr(self.player, 'critic_size') else 11,
                'action_size': self.player.ac_net.actor_net[-1].out_features if hasattr(self.player, 'ac_net') else 3,

                # Сохраняем параметры LSTM
                'lstm_hidden': lstm_hidden,
                'history_len': self.history_len,
                'action_vector_size': self.action_dim,

                'name': self.player.name,
                'stack': self.player.stack,
                'model_type': 'ActorCritic_DualLSTM',
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
        Загружает состояние NeuralACAgent с ДЕТАЛЬНОЙ ПРОВЕРКОЙ ключей.

        Args:
            self: экземпляр NeuralACAgent для загрузки данных
            filename: имя файла для загрузки
            save_dir: директория с файлами
            load_memory: загружать ли память
            strict: строгая загрузка весов модели (по умолчанию True)
        """
        try:
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                print(f"[❌] Файл {filepath} не найден!")
                return False

            checkpoint = torch.load(filepath, map_location=torch.device('cuda'))

            if checkpoint.get('model_type') != 'ActorCritic_DualLSTM':
                print(f"[⚠️] Внимание: model_type='{checkpoint.get('model_type')}', ожидается 'ActorCritic_DualLSTM'")

            # === ДЕТАЛЬНАЯ ПРОВЕРКА КЛЮЧЕЙ ПЕРЕД ЗАГРУЗКОЙ ===
            file_state_dict = checkpoint['ac_net_state_dict']
            model_state_dict = self.player.ac_net.state_dict()

            print("\n" + "="*70)
            print("=== ПРОВЕРКА СОВПАДЕНИЯ КЛЮЧЕЙ STATE_DICT ===")
            print("="*70)

            print("\n📄 Ключи state_dict в ФАЙЛЕ:")
            for k in sorted(file_state_dict.keys()):
                shape = list(file_state_dict[k].shape)
                print(f"   {k}: {shape}")

            print("\n🔧 Ключи state_dict в ТЕКУЩЕЙ МОДЕЛИ:")
            for k in sorted(model_state_dict.keys()):
                shape = list(model_state_dict[k].shape)
                print(f"   {k}: {shape}")

            # Находим несовпадения
            file_keys = set(file_state_dict.keys())
            model_keys = set(model_state_dict.keys())

            missing_keys = model_keys - file_keys      # Есть в модели, нет в файле
            unexpected_keys = file_keys - model_keys   # Есть в файле, нет в модели
            common_keys = file_keys & model_keys

            # Проверка размеров совпадающих ключей
            shape_mismatches = []
            for key in sorted(common_keys):
                file_shape = list(file_state_dict[key].shape)
                model_shape = list(model_state_dict[key].shape)
                if file_shape != model_shape:
                    shape_mismatches.append((key, file_shape, model_shape))

            # Вывод результатов
            has_errors = False

            if missing_keys:
                print(f"\n❌ MISSING KEYS (есть в модели, НЕТ в файле) — {len(missing_keys)}:")
                for k in sorted(missing_keys):
                    print(f"   - {k}")
                has_errors = True

            if unexpected_keys:
                print(f"\n❌ UNEXPECTED KEYS (есть в файле, НЕТ в модели) — {len(unexpected_keys)}:")
                for k in sorted(unexpected_keys):
                    print(f"   - {k}")
                has_errors = True

            if shape_mismatches:
                print(f"\n❌ SHAPE MISMATCHES ({len(shape_mismatches)}):")
                for key, file_shape, model_shape in shape_mismatches:
                    print(f"   - {key}: файл={file_shape} != модель={model_shape}")
                has_errors = True

            if has_errors:
                print("\n" + "="*70)
                raise RuntimeError(
                    f"❌ ЗАГРУЗКА МОДЕЛИ ПРЕРВАНА! Найдены несовпадения ключей/размеров.\n"
                    f"   Missing keys: {len(missing_keys)}\n"
                    f"   Unexpected keys: {len(unexpected_keys)}\n"
                    f"   Shape mismatches: {len(shape_mismatches)}\n"
                    f"Проверьте архитектуру модели и параметры сохранения."
                )

            print(f"\n✅ Все {len(common_keys)} ключей совпали!")
            print("="*70 + "\n")

            # === ЗАГРУЗКА ВЕСОВ (strict=True) ===
            self.player.ac_net.load_state_dict(checkpoint['ac_net_state_dict'], strict=True)
            print("[✅] Веса модели загружены успешно (strict=True)")

            if 'optimizer_state_dict' in checkpoint:
                self.player.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("[✅] Оптимизатор загружен")

            if 'gamma' in checkpoint:
                self.player.gamma = checkpoint['gamma']
                print(f"    Gamma: {self.player.gamma}")

            if load_memory and 'memory' in checkpoint and hasattr(self.player, 'memory'):
                self.player.memory = deque(checkpoint['memory'], maxlen=self.player.memory.maxlen)
                print(f"    Память: загружено {len(self.player.memory)} записей")

            if 'stack' in checkpoint:
                self.player.stack = checkpoint['stack']

            print(f"\n[✅] NeuralACAgent загружен из {filepath}")
            print(f"    Имя: {checkpoint.get('name', 'Unknown')}")
            print(f"    actor_size: {checkpoint.get('actor_size', 'N/A')}")
            print(f"    critic_size: {checkpoint.get('critic_size', 'N/A')}")
            print(f"    lstm_hidden: {checkpoint.get('lstm_hidden', 'N/A')}")

            return True

        except RuntimeError as e:
            # Пробрасываем ошибки несовпадения ключей
            raise
        except Exception as e:
            print(f"[❌] Ошибка при загрузке агента: {e}")
            import traceback
            traceback.print_exc()
            return False