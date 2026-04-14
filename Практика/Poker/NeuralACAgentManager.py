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

        # --- Entropy Decay: от exploration к exploitation ---
        # УСКОРЕННОЕ ЗАТУХАНИЕ: бот должен начать «наглеть», а не играть случайно
        self.entropy_coef = 0.001          # Стартовое (уже маленькое)
        self.entropy_coef_min = 0.0001     # Минимальное (было 0.0005)
        self.entropy_decay = 0.995         # Быстрое затухание (было 0.9995)
        self.entropy_update_interval = 200  # Чаще обновлять (было 500)
        self.last_entropy_update = 0

        # --- Rolling статистика обучения ---
        self.results_buffer = deque(maxlen=500)    # (is_winner, net_profit, pot_size)
        self.loss_log = deque(maxlen=100)          # (actor_loss, critic_loss, entropy)
        self.game_counter = 0
        self.epoch_interval = 200  # Уменьшено с 500 для более частого вывода энтропии
        self.last_update_games = 0
        self.big_blind = 10

        self.history_buffer = deque(maxlen=self.history_len)
        self.reset_history()

        # === ЛОГИРОВАНИЕ РАСПРЕДЕЛЕНИЯ ДЕЙСТВИЙ (Шаг 1) ===
        self.action_counter = {0: 0, 1: 0, 2: 0}  # fold, raise, call
        self.games_since_log = 0
        self.action_log_interval = 1000


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
            avg_a, avg_c, avg_e = 0, 0, 0  # Ещё не было обновлений сети

        return winrate, avg_a, avg_c, avg_e, avg_pot_bb

    def log_action_frequencies(self):
        """
        Выводит процентное соотношение Fold/Call/Raise за последние N игр.
        Сбрасывает счётчики после логирования.
        """
        total = sum(self.action_counter.values())
        if total == 0:
            return
        
        fold_pct = self.action_counter[0] / total * 100
        call_pct = self.action_counter[2] / total * 100
        raise_pct = self.action_counter[1] / total * 100
        
        msg = (f"[ACTION FREQ over {total} games] "
               f"Fold: {fold_pct:.1f}%, Call: {call_pct:.1f}%, Raise: {raise_pct:.1f}%")
        StaticLogger.print_to('summary', msg)
        print(f"\n📊 {msg}")
        
        # Сброс счётчиков
        self.action_counter = {0: 0, 1: 0, 2: 0}
        self.games_since_log = 0

    def print_stats_if_needed(self, epoch):
        """
        Выводит статистику каждые epoch_interval игр.
        Формат: [Epoch X] Winrate(Last 500): 54% | Loss(A/C): 0.021/0.14 | AvgPot: 150BB
        Энтропия пишется отдельно в entropy.log при каждом update сети.
        """
        games_since_last = self.game_counter - self.last_update_games
        if games_since_last >= self.epoch_interval:
            stats = self.get_training_stats()
            if stats:
                winrate, avg_a, avg_c, avg_e, avg_pot_bb = stats

                # Если loss ещё не было — добавим пояснение
                if avg_a == 0 and avg_c == 0:
                    loss_str = "pending (no updates yet)"
                else:
                    loss_str = f"{avg_a:.3f}/{avg_c:.2f}"

                msg = (f"[Epoch {epoch}] Winrate(Last 500): {winrate:.0f}% | "
                       f"Loss(A/C): {loss_str} | AvgPot: {avg_pot_bb:.0f}BB | "
                       f"EntCoef: {self.entropy_coef:.4f}")
                print(f"\n\U0001f4ca {msg}")
                StaticLogger.print_to("training", msg)
            self.last_update_games = self.game_counter

            # Логируем распределение действий каждые epoch_interval игр
            self.log_action_frequencies()

            # Комплексная диагностика каждые 2000 игр
            if self.game_counter % 2000 == 0:
                self.diagnose_agent()

    def update_entropy_coef(self):
        """
        Плавное уменьшение коэффициента энтропии — от exploration к exploitation.
        Вызывать после каждой раздачи (из GameManager.winners_distribution).
        """
        games_since_update = self.game_counter - self.last_entropy_update
        if games_since_update >= self.entropy_update_interval:
            if self.entropy_coef > self.entropy_coef_min:
                self.entropy_coef *= self.entropy_decay
                self.last_entropy_update = self.game_counter

        # Принудительное ограничение: энтропия не должна мешать обучению
        if self.entropy_coef > 0.005 and self.game_counter > 2000:
            self.entropy_coef = 0.001  # Жёсткий сброс после 2000 игр

        # Экстренный сброс: если модель "замерла" в случайных действиях
        if self.game_counter > 500 and self.entropy_coef > 0.0005:
            self.entropy_coef = 0.0001  # Минимальный коэффициент

    def validate_hand_values(self):
        """
        Проверяет, различает ли Критик сильные и слабые руки.
        Тестирует:
          - AA (сильная рука): hand_strength ≈ 0.85, avg_opp ≈ 0.15
          - 72o (слабая рука): hand_strength ≈ 0.15, avg_opp ≈ 0.85

        Выводит predicted Value для каждой руки в validation.log.
        Разница должна быть колоссальной.
        """
        self.player.ac_net.eval()

        # Тестовые кейсы: [hand_name, hand_strength, avg_opp_strength]
        test_hands = [
            ('AA (тузы)',       0.85, 0.15),
            ('KK (короли)',     0.80, 0.20),
            ('72o (мусор)',     0.15, 0.85),
            ('random (средняя)', 0.50, 0.50),
        ]

        StaticLogger.print_to('validation', "\n=== HAND VALUE VALIDATION ===")
        print("\n🧪 === HAND VALUE VALIDATION ===")

        values_dict = {}

        with torch.no_grad():
            for hand_name, hand_str, opp_str in test_hands:
                # Создаём critic state vector (как в build_state_vectors)
                # s_critic = [hand_strength, bet, stack, pot, preflop, flop, turn, river, decision, ..., avg_opp]
                s_critic = np.zeros(self.player.critic_size, dtype=np.float32)
                s_critic[0] = hand_str       # hand_strength
                s_critic[-1] = opp_str       # avg_opp_strength (последний элемент)

                # History (фиктивная, заполнена нулями)
                history = torch.zeros(1, self.history_len, self.action_dim,
                                      dtype=torch.float32).to(self.device)

                s_critic_tensor = torch.tensor(s_critic, dtype=torch.float32).unsqueeze(0).to(self.device)

                value = self.player.ac_net.get_critic_value(s_critic_tensor, history)
                values_dict[hand_name] = value

                msg = f"  {hand_name:20s} → Value = {value:+.4f}"
                StaticLogger.print_to('validation', msg)
                print(msg)

        # === ДИАГНОСТИКА: проверяем работу Skip Connection ===
        aa_value = values_dict.get('AA (тузы)', 0)
        junk_value = values_dict.get('72o (мусор)', 0)
        value_gap = aa_value - junk_value

        if value_gap < 0.1:
            warning = f"⚠️ WARNING: Value gap = {value_gap:.4f} < 0.1! Skip Connection может не работать!"
            StaticLogger.print_to('validation', warning)
            print(warning)
        elif value_gap < 0.5:
            warning = f"⚡ Value gap = {value_gap:.4f} — слабое различие. Критик сомневается."
            StaticLogger.print_to('validation', warning)
            print(warning)
        else:
            ok = f"✅ Value gap = {value_gap:.4f} — Критик хорошо различает руки!"
            StaticLogger.print_to('validation', ok)
            print(ok)

        StaticLogger.print_to('validation', "=== END HAND VALUE VALIDATION ===\n")
        print("=== END HAND VALUE VALIDATION ===\n")

        self.player.ac_net.train()

    def diagnose_agent(self):
        """
        Комплексная диагностика агента:
        1. validate_hand_values() — проверяет Критика
        2. action_frequencies — проверяет Актора
        3. Сравнение: если Критик знает AA>72o, но Актор fold'ит → проблема в LR/награде
        """
        StaticLogger.print_to('summary', "\n" + "="*60)
        StaticLogger.print_to('summary', "🔍 AGENT DIAGNOSTICS")
        StaticLogger.print_to('summary', "="*60)

        # 1. Проверка Критика
        self.validate_hand_values()

        # 2. Проверка распределения действий
        self.log_action_frequencies()

        # 3. Текущие гиперпараметры
        stats_msg = (
            f"\n📊 HYPERPARAMS:\n"
            f"   entropy_coef = {self.entropy_coef:.6f}\n"
            f"   Actor LR = 2e-4, Critic LR = 1e-4\n"
            f"   games_played = {self.game_counter}\n"
            f"   entropy_update_interval = {self.entropy_update_interval}"
        )
        StaticLogger.print_to('summary', stats_msg)
        print(stats_msg)

        # 4. Интерпретация
        StaticLogger.print_to('summary', "\n📋 INTERPRETATION:")
        StaticLogger.print_to('summary', "   Если Value(AA) >> Value(72o), но Актор часто делает Fold:")
        StaticLogger.print_to('summary', "   → Проблема: Actor LR слишком маленький или награда слабая")
        StaticLogger.print_to('summary', "   Если Value(AA) ≈ Value(72o):")
        StaticLogger.print_to('summary', "   → Проблема: Skip Connection / LayerNorm не работают")
        StaticLogger.print_to('summary', "   Если энтропия ≈ 1.1:")
        StaticLogger.print_to('summary', "   → Проблема: Бот играет случайно (entropy_coef слишком большой)")
        StaticLogger.print_to('summary', "="*60 + "\n")

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
        
        # Дополнительно: проверяем, как Критик оценивает конкретные руки
        self.validate_hand_values()
        
        return winrate

    def act(self, s_actor: list, s_critic: list, can_check=False, can_raise=True, training_mode=False):
        """
        Выбирает действие, используя Actor (на основе S_actor),
        и получает оценку состояния V(s) от Critic (на основе S_critic).
        Сохраняет данные для on-policy обучения.

        Args:
            can_check: Можно ли проверить (если current_bet == player.bet)
            can_raise: Можно ли сделать рейз (хватает ли фишек)
            training_mode: Если True — сэмплировать из политики, иначе — argmax
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

        # === ACTION MASKING: маскируем нелегальные действия ===
        # action_idx: 0=fold, 1=raise, 2=call
        # Создаём маску легальных действий: [fold, raise, call]
        legal_mask = torch.tensor([True, can_raise, True], device=self.device)

        # Check = call с 0 ставкой. В нашей системе check и call — это одно действие (idx=2).
        # can_check=False означает: current_bet > player.bet → нужно платить → call легален
        # Значит check отдельно НЕ маскируем — он слит с call

        # Применяем маску к логитам перед сэмплированием/выбором
        action_logits = action_logits.masked_fill(~legal_mask, -1e9)

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

        # Сохраняем маску в episode_data для последующего обучения
        self.episode_data.append((s_actor, s_critic, current_history_snapshot,
                                  action_idx, log_prob, value_estimate,
                                  init_actor_h, init_critic_h, legal_mask))

        self.last_s_actor = s_actor
        self.last_s_critic = s_critic
        self.last_action_idx = action_idx

        # === ЛОГИРОВАНИЕ ДЕЙСТВИЙ (Шаг 1) ===
        self.action_counter[action_idx] += 1

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

        # Распаковываем все поля из episode_data (индекс 8 = legal_mask)
        s_actors = [d[0] for d in self.episode_data]
        s_critics = [d[1] for d in self.episode_data]
        histories = [d[2] for d in self.episode_data]
        actions = [d[3] for d in self.episode_data]
        init_actor_hs = [d[6] for d in self.episode_data]
        init_critic_hs = [d[7] for d in self.episode_data]
        legal_masks = [d[8] for d in self.episode_data]  # Маски легальных действий

        # Формируем траекторию — одну полную раздачу
        trajectory = {
            's_actors': s_actors,
            's_critics': s_critics,
            'histories': histories,
            'actions': actions,
            'returns': returns,
            'init_actor_hs': init_actor_hs,
            'init_critic_hs': init_critic_hs,
            'legal_masks': legal_masks,  # Сохраняем маски для обучения
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
        РЕФАКТОРИНГ: Глобальная нормализация Advantage по всему батчу.
        1. Собираем все log_probs, values, returns со всех траекторий
        2. Склеиваем в глобальные тензоры
        3. Нормализуем глобальный Advantage
        4. Считаем итоговые лоссы
        """
        trajectories = NNData.get_buffer()

        # Guard: проверяем, что есть хотя бы одна траектория
        if not trajectories or len(trajectories) == 0:
            StaticLogger.print_to("training", "[WARN] _update_network called with empty buffer, skipping.")
            return

        self.player.ac_net.train()

        # === ГЛОБАЛЬНЫЕ КОЛЛЕКЦИИ ДЛЯ СБОРА ДАННЫХ ===
        all_log_probs = []
        all_values = []
        all_returns = []
        all_entropies = []
        num_trajectories = 0

        # === 1. СБОР ДАННЫХ СО ВСЕХ ТРАЕКТОРИЙ ===
        for traj_idx, traj in enumerate(trajectories):
            s_actors = traj['s_actors']
            s_critics = traj['s_critics']
            histories = traj['histories']
            actions = traj['actions']
            returns = traj['returns']
            legal_masks = traj.get('legal_masks', None)

            # Guard: пропускаем битые траектории
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

            # Награды уже в Big Blinds (из GameManager), дополнительное масштабирование не нужно

            # --- Backpropagation Through Time: пошаговый прогон раздачи ---
            action_logits_list = []
            values_list = []
            curr_actor_h = None
            curr_critic_h = None

            for i in range(len(s_actors_t)):
                logits, val, curr_actor_h, curr_critic_h = self.player.ac_net(
                    s_actors_t[i].unsqueeze(0),
                    s_critics_t[i].unsqueeze(0),
                    histories_t[i].unsqueeze(0),
                    actor_hidden=curr_actor_h,
                    critic_hidden=curr_critic_h
                )
                action_logits_list.append(logits)
                values_list.append(val)

            # Склеиваем результаты [Seq_Len, ...]
            action_logits = torch.cat(action_logits_list, dim=0)
            values = torch.cat(values_list, dim=0).squeeze(-1)

            # === ACTION MASKING ===
            if legal_masks is not None:
                masks_t = torch.stack(legal_masks).to(self.device)
                action_logits = action_logits.masked_fill(~masks_t, -1e9)

            # --- Сохраняем в глобальные коллекции (НЕ считая loss!) ---
            dist = distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions_t)

            all_log_probs.append(log_probs)
            all_values.append(values)
            all_returns.append(returns_t)
            all_entropies.append(dist.entropy().mean())

            num_trajectories += 1

        # Guard: нет валидных траекторий
        if num_trajectories == 0:
            StaticLogger.print_to("training", "[WARN] No valid trajectories to train on, skipping update.")
            NNData.clear()
            return

        # === 2. СКЛЕИВАЕМ В ГЛОБАЛЬНЫЕ ТЕНЗОРЫ ===
        global_log_probs = torch.cat(all_log_probs)
        global_values = torch.cat(all_values)
        global_returns = torch.cat(all_returns)
        global_entropy = torch.stack(all_entropies).mean()

        # === 3. ГЛОБАЛЬНЫЙ ADVANTAGE С НАДЁЖНОЙ НОРМАЛИЗАЦИЕЙ ===
        global_advantage = global_returns - global_values.detach()

        # Отладка: печатаем сырые значения ПЕРЕД нормализацией
        adv_mean_raw = global_advantage.mean().item()
        adv_std_raw = global_advantage.std().item()
        returns_mean = global_returns.mean().item()
        values_mean = global_values.mean().item()

        # ИСПРАВЛЕНИЕ: повышаем порог с 1e-5 до 1e-3 и ставим минимум std=1.0
        # Чтобы нормализация не производила огромные числа из микроскопического std
        if global_advantage.numel() > 1 and adv_std_raw > 1e-3:
            global_advantage = (global_advantage - adv_mean_raw) / max(adv_std_raw, 1.0)

        # === 4. ИТОГОВЫЕ ЛОССЫ ===
        actor_loss = -(global_log_probs * global_advantage).mean()
        critic_loss = F.huber_loss(global_values, global_returns, delta=1.0)

        # Снижаем влияние энтропии (entropy_coef <= 0.001)
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * global_entropy

        # === 5. BACKPROP ===
        self.player.optimizer.zero_grad()
        total_loss.backward()

        # Отладка: проверяем что градиенты текут через Актор
        actor_grad_norm = 0.0
        for p in self.player.ac_net.actor_net.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.norm().item() ** 2
        actor_grad_norm = actor_grad_norm ** 0.5

        # Проверяем requires_grad на логитах
        # action_logits должен быть результатом torch.cat → gradFn = CatBackward
        logits_requires_grad = action_logits.requires_grad

        torch.nn.utils.clip_grad_norm_(self.player.ac_net.parameters(), max_norm=0.5)
        self.player.optimizer.step()

        # Очищаем буфер ПОСЛЕ успешного обновления
        NNData.clear()

        # === ЛОГИРОВАНИЕ С ОТЛАДКОЙ ===
        self.loss_log.append((actor_loss.item(), critic_loss.item(), global_entropy.item()))

        StaticLogger.print_to("entropy",
            f"Entropy: {global_entropy.item():.4f} | "
            f"Adv(raw): {adv_mean_raw:.4f}±{adv_std_raw:.4f} | "
            f"Returns: {returns_mean:.4f} | Values: {values_mean:.4f} | "
            f"Actor grad: {actor_grad_norm:.6f} | logits_grad: {logits_requires_grad} | "
            f"Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | "
            f"EntCoef: {self.entropy_coef:.6f}"
        )

        StaticLogger.print_to(
            "loss",
            f"Update ({num_trajectories} hands, GLOBAL adv): "
            f"Loss={total_loss.item():.4f}, "
            f"Actor={actor_loss.item():.4f}, "
            f"Critic={critic_loss.item():.4f}, "
            f"AdvStd={adv_std_raw:.4f}, ActorGrad={actor_grad_norm:.6f}"
        )

        # КОНСОЛЬНЫЙ ВЫВОД для быстрого контроля
        if adv_std_raw < 0.01:
            print(f"\n⚠️ [ALERT] Advantage std={adv_std_raw:.6f} — КРИТИЧЕСКИ МАЛЕНЬКИЙ! Градиенты могут исчезать.")
        if global_entropy.item() > 1.09:
            print(f"\n⚠️ [ALERT] Entropy={global_entropy.item():.4f} ≈ ln(3) — модель выдаёт случайные действия!")
        if actor_grad_norm < 1e-6:
            print(f"\n⚠️ [ALERT] Actor gradient norm={actor_grad_norm:.8f} ≈ 0 — Актор НЕ ОБУЧАЕТСЯ!")

        print(f"✅ [UPDATE] {num_trajectories} hands | Loss={total_loss.item():.4f} | "
              f"A={actor_loss.item():.4f} | C={critic_loss.item():.4f} | "
              f"Ent={global_entropy.item():.4f} | AdvStd={adv_std_raw:.4f} | "
              f"ActorGrad={actor_grad_norm:.6f} | R={returns_mean:.2f} | V={values_mean:.2f}")


    def ask_decision(self, s_actor: list, s_critic: list, can_check=False, can_raise=True):
        """
        Интерфейс с GameManager. Принимает векторы состояния, вызывает act,
        устанавливает решение игрока и логирует ситуацию.

        Args:
            can_check: Можно ли проверить (если current_bet == player.bet)
            can_raise: Можно ли сделать рейз (хватает ли фишек)
        """

        action_idx = self.act(s_actor, s_critic, can_check, can_raise)

        action = ACTIONS[action_idx]
        self.player.set_decision(action)

        StaticLogger.print_to("game", f"\nИгрок {self.player.name}")
        StaticLogger.print_to("game", f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print_to("game", f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")

        StaticLogger.print_to("game", f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print_to("game", f"Ваш выбор: {self.player.decision}")

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



    def save_ac_agent(self, filename="Big_Experiment_better_NN.pth", save_dir="models"):
        """
        Сохраняет состояние NeuralACAgent (Actor-Critic)

        Args:
            filename: имя файла для сохранения
            save_dir: директория для сохранения
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

            torch.save(checkpoint, filepath)
            print(f"[✅] NeuralACAgent '{self.player.name}' сохранен в {filepath}")

            return filepath

        except Exception as e:
            print(f"[❌] Ошибка при сохранении агента: {e}")
            return None

    def load_ac_agent(self, filename="neural_ac_agent.pth", save_dir="models", strict=True):
        """
        Загружает состояние NeuralACAgent с ДЕТАЛЬНОЙ ПРОВЕРКОЙ ключей.

        Args:
            filename: имя файла для загрузки
            save_dir: директория с файлами
            strict: строгая загрузка весов модели (по умолчанию True)
        """
        try:
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                print(f"[❌] Файл {filepath} не найден!")
                return False

            expected_missing = False  # Флаг несовместимости архитектуры

            checkpoint = torch.load(filepath, map_location=torch.device('cuda'))

            if checkpoint.get('model_type') != 'ActorCritic_DualLSTM':
                print(f"[⚠️] Внимание: model_type='{checkpoint.get('model_type')}', ожидается 'ActorCritic_DualLSTM'")

            # Проверка совместимости архитектуры (Skip Connections + LayerNorm)
            file_state_dict = checkpoint['ac_net_state_dict']
            has_old_critic = any('critic_net' in k for k in file_state_dict.keys())
            has_new_critic = any('critic_fc' in k for k in file_state_dict.keys())
            has_layer_norm = any('critic_layer_norm' in k for k in file_state_dict.keys())

            model_state_dict = self.player.ac_net.state_dict()

            if has_old_critic and not has_new_critic:
                print("[⚠️] ВНИМАНИЕ: НЕСОВМЕСТИМАЯ АРХИТЕКТУРА!")
                print("   Загружаемый чекпоинт использует старую архитектуру Critic (Sequential).")
                print("   Текущая модель использует Skip Connections (critic_fc1/fc2/fc3).")
                print("   Веса Критика НЕ будут загружены — Critic будет обучаться с нуля.")
                print("   Веса Актора и LSTM будут загружены (если совместимы).")
                print()

                # Фильтруем: убираем старые веса Critic, оставляем Actor + LSTM
                filtered_state_dict = {}
                for k, v in file_state_dict.items():
                    if 'critic_net' in k:
                        continue  # Пропускаем старый Critic
                    elif k in model_state_dict and model_state_dict[k].shape == v.shape:
                        filtered_state_dict[k] = v

                file_state_dict = filtered_state_dict
            elif has_new_critic and not has_layer_norm:
                print("[⚠️] ВНИМАНИЕ: Чекпоинт ДО добавления LayerNorm.")
                print("   critic_layer_norm будет инициализирован с нуля.")
                print("   Остальные веса будут загружены корректно.")
                print()
            else:
                pass  # file_state_dict и model_state_dict уже определены

            # === ДЕТАЛЬНАЯ ПРОВЕРКА КЛЮЧЕЙ ПЕРЕД ЗАГРУЗКОЙ ===

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

            # Ожидаемые missing keys:
            # 1. Старый чекпоинт (Sequential Critic) → critic_fc1/fc2/fc3 отсутствуют
            # 2. Чекпоинт до LayerNorm → critic_layer_norm отсутствует
            expected_missing = (has_old_critic and not has_new_critic) or \
                               (has_new_critic and not has_layer_norm)

            if missing_keys:
                if expected_missing:
                    reason = "Critic обучается с нуля" if (has_old_critic and not has_new_critic) else "LayerNorm инициализируется с нуля"
                    print(f"\n⚠️ MISSING KEYS (ожидаемо — {reason}) — {len(missing_keys)}:")
                else:
                    print(f"\n❌ MISSING KEYS (есть в модели, НЕТ в файле) — {len(missing_keys)}:")
                for k in sorted(missing_keys):
                    print(f"   - {k}")
                if not expected_missing:
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

            if expected_missing:
                if has_old_critic and not has_new_critic:
                    print(f"\n⚠️ Загружено {len(common_keys)} ключей (Critic fc1/fc2/fc3 — с нуля)")
                else:
                    print(f"\n⚠️ Загружено {len(common_keys)} ключей (critic_layer_norm — с нуля)")
            else:
                print(f"\n✅ Все {len(common_keys)} ключей совпали!")
            print("="*70 + "\n")

            # === ЗАГРУЗКА ВЕСОВ (strict=False для старых чекпоинтов) ===
            strict_mode = not expected_missing
            self.player.ac_net.load_state_dict(file_state_dict, strict=strict_mode)
            print(f"[✅] Веса модели загружены успешно (strict={strict_mode})")

            if 'optimizer_state_dict' in checkpoint:
                if expected_missing:
                    print("[⚠️] Оптимизатор НЕ загружен (архитектура изменилась — начнётся с нуля)")
                else:
                    self.player.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("[✅] Оптимизатор загружен")

            if 'gamma' in checkpoint:
                self.player.gamma = checkpoint['gamma']
                print(f"    Gamma: {self.player.gamma}")

            if 'stack' in checkpoint:
                self.player.stack = checkpoint['stack']

            print(f"\n[✅] NeuralACAgent загружен из {filepath}")
            print(f"    Имя: {checkpoint.get('name', 'Unknown')}")
            print(f"    actor_size: {checkpoint.get('actor_size', 'N/A')}")
            print(f"    critic_size: {checkpoint.get('critic_size', 'N/A')}")
            print(f"    lstm_hidden: {checkpoint.get('lstm_hidden', 'N/A')}")

            if expected_missing:
                if has_old_critic and not has_new_critic:
                    print(f"\n⚠️ Critic (fc1/fc2/fc3) будет обучаться с нуля (Skip Connections)")
                else:
                    print(f"\n⚠️ critic_layer_norm будет обучаться с нуля")

            # После загрузки модели — сразу валидация
            print("\n🧪 Running hand value validation after model load...")
            self.validate_hand_values()

            return True

        except RuntimeError as e:
            # Пробрасываем ошибки несовпадения ключей
            raise
        except Exception as e:
            print(f"[❌] Ошибка при загрузке агента: {e}")
            import traceback
            traceback.print_exc()
            return False