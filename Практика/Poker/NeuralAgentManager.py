import torch
import torch.nn as nn
import torch.optim as optim
import random
from . import *  # Твои импорты
from .PokerStageNetwork import *
from .HandCalculator import *
# --- Исправленный Класс Агента (Мозг) ---
class NeuralAgent(Player):
    def __init__(self, name="Agent", stack=100, learning_rate=0.001, gamma=0.99):
        super().__init__(name, stack)
        self.gamma = gamma

        # Инициализируем сети
        self.networks = {
            'preflop': PokerStageNetwork(),
            'flop': PokerStageNetwork(),
            'turn': PokerStageNetwork(),
            'river': PokerStageNetwork()
        }

        # Оптимизаторы
        self.optimizers = {
            key: optim.Adam(net.parameters(), lr=learning_rate)
            for key, net in self.networks.items()
        }
        self.criterion = nn.MSELoss()


# --- Исправленный Менеджер (Логика обучения) ---
class NeuralAgentManager(PlayerManager):
    def __init__(self, agent: NeuralAgent):
        super().__init__(agent)

        # === ПАМЯТЬ ДЛЯ ОБУЧЕНИЯ ===
        # Здесь мы храним то, что произошло В ПРОШЛЫЙ РАЗ,
        # чтобы обучиться, когда узнаем результат.
        self.last_state = None
        self.last_action_idx = None
        self.last_stage = None

        # Стэк в начале раздачи (чтобы считать Profit/Loss)
        self.starting_stack = 0.0

    def start_new_hand(self):
        """Вызывать в начале каждой раздачи"""
        self.starting_stack = self.player.stack
        self.last_state = None
        self.last_action_idx = None
        self.last_stage = None

    def ask_decision(self, current_bet, min_raise, community_cards, opponent: Player, stage='preflop'):
        # 1. Собираем ТЕКУЩЕЕ состояние
        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)

        current_state = self.get_state_vector(
            hand_strength,
            self.player.get_bet(),
            self.player.get_stack(),
            opponent.get_bet(),
            opponent.get_stack()
        )

        # 2. ОБУЧЕНИЕ (Шаг 1: Промежуточное)
        # Если у нас есть память о прошлом ходе, мы должны "закрыть" его,
        # сказав: "Прошлый ход привел нас в текущее состояние".
        if self.last_state is not None:
            # Награда 0, так как игра еще не закончилась.
            # Мы просто перешли в следующее состояние.
            self.train_step(
                stage=self.last_stage,
                state=self.last_state,
                action=self.last_action_idx,
                reward=0,
                next_state=current_state,
                done=False
            )

        # 3. Принятие НОВОГО решения
        action_idx = self.select_action(current_state, stage)
        self.player.decision = self.DECISIONS[action_idx + 1]

        # 4. ЗАПОМИНАНИЕ (Для следующего раза)
        self.last_state = current_state
        self.last_action_idx = action_idx
        self.last_stage = stage

        return self.player.decision

    def finish_hand(self, won_amount):
        """
        Вызывается в конце игры, когда известны победители.
        """
        if self.last_state is None:
            return

        # Рассчитываем реальную награду (изменение стэка)
        # Если выиграл: reward положительный. Если проиграл: отрицательный (размер ставок).
        final_stack = self.player.stack + won_amount  # won_amount - это доля из банка, которую нам только что дали
        # Внимание: логика ниже зависит от того, когда ты обновляешь stack игрока.
        # Допустим, self.player.stack уже обновлен в winners_distribution

        actual_reward = self.player.stack - self.starting_stack

        # Обучаем ПОСЛЕДНЕЕ действие (на ривере или когда был фолд)
        self.train_step(
            stage=self.last_stage,
            state=self.last_state,
            action=self.last_action_idx,
            reward=actual_reward,
            next_state=None,  # Состояния больше нет
            done=True  # Игра окончена
        )

        # Сбрасываем память
        self.last_state = None

    # ... методы get_state_vector и select_action остаются без изменений ...

    def train_step(self, stage, state, action, reward, next_state, done):
        # Исправлено обращение к сетям: self.player.networks
        network = self.player.networks[stage]
        optimizer = self.player.optimizers[stage]

        network.train()

        q_values = network(state)
        current_q = q_values[action]

        with torch.no_grad():
            if done:
                target_q = torch.tensor(float(reward))
            else:
                # Если перешли на другую улицу (state другой длины или типа),
                # нужно убедиться, что next_state подается в правильную сеть.
                # Но в DQN обычно next_q берется из той же сети (Target Network),
                # либо из сети следующей улицы.
                # Для упрощения пока берем ТУ ЖЕ сеть (stage), так как это стандартный Q-learning
                next_q_values = network(next_state)
                max_next_q = torch.max(next_q_values)
                target_q = reward + self.player.gamma * max_next_q

        loss = self.player.criterion(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def get_state_vector(self, hand_strength, hero_bet, hero_stack, villain_bet, villain_stack):
        """
        Формирует тензор состояния из 5 параметров.
        Важно: Желательно нормализовать данные (привести стэки к диапазону 0-1),
        чтобы нейросеть обучалась быстрее.
        """
        # Пример простой нормализации (делим на начальный стек, допустим 1000)
        MAX_STACK = 100.0

        state = [
            hand_strength,  # 1. Hand Strength (0.0 - 1.0)
            hero_bet / MAX_STACK,  # 2. Hero Current Bet
            hero_stack / MAX_STACK,  # 3. Hero Stack
            villain_stack / MAX_STACK,  # 4. Villain Stack
            villain_bet / MAX_STACK  # 5. Villain Bet
        ]
        return torch.FloatTensor(state)

    def select_action(self, state_vector, stage, epsilon=0.1):
        """
        Выбор действия: Epsilon-Greedy стратегия.
        Иногда делаем случайный ход (exploration), иногда лучший (exploitation).
        """
        if random.random() < epsilon:
            return random.randint(0, 2)  # 0=Fold, 1=Call, 2=Raise

        network = self.player.networks[stage]
        network.eval()  # Режим оценки
        with torch.no_grad():
            q_values = network(state_vector)
            action = torch.argmax(q_values).item()
        return action