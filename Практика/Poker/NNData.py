from Практика.Poker import Game
from matplotlib import pyplot as plt

class NNData:

    _instance = None
    episode_buffer = []
    BATCH_SIZE = 2048
    loss_critic_buffer = []
    loss_actor_buffer = []
    loss_buffer = []
    entropy_buffer = []
    win_history = []  # Список из 0 и 1 для каждой игры
    rolling_win_rate_history = []

    action_freq_buffer = []
    value_gap_buffer = []
    _tmp_action_counter = {0: 0, 1: 0, 2: 0}

    @staticmethod
    def record_game_result(is_win, window_size=100):
        """
        Записывает результат игры и вычисляет скользящий винрейт.
        is_win: True если агент выиграл, False если нет.
        """
        NNData.win_history.append(1 if is_win else 0)

        last_games = NNData.win_history[-window_size:]


        current_rolling_rate = (sum(last_games) / len(last_games)) * 100
        NNData.rolling_win_rate_history.append(current_rolling_rate)

    @staticmethod
    def record_action(action_idx):
        """Записывает каждое действие бота в реальном времени"""
        NNData._tmp_action_counter[action_idx] += 1

    @staticmethod
    def commit_action_freq():
        """Вычисляет проценты и сохраняет в историю (вызывать в конце игры или эпохи)"""
        total = sum(NNData._tmp_action_counter.values())
        if total > 0:
            freqs = {
                'fold': (NNData._tmp_action_counter[0] / total) * 100,
                'raise': (NNData._tmp_action_counter[1] / total) * 100,
                'call': (NNData._tmp_action_counter[2] / total) * 100
            }
            NNData.action_freq_buffer.append(freqs)
            # Сбрасываем счетчик для следующего окна
            NNData._tmp_action_counter = {0: 0, 1: 0, 2: 0}


    @staticmethod
    def add_value_gap(gap_value):
        NNData.value_gap_buffer.append(gap_value)

    @staticmethod
    def get_action_freq():
        return NNData._tmp_action_counter

    @staticmethod
    def add_buffer(args):
        NNData.episode_buffer.append(args)

    @staticmethod
    def is_full():
        return len(NNData.episode_buffer) >= NNData.BATCH_SIZE
    @staticmethod
    def get_buffer():
        return NNData.episode_buffer
    @staticmethod
    def get_sleazy_win():
        return NNData.rolling_win_rate_history[-1]

    @staticmethod
    def clear():
        NNData.episode_buffer.clear()

    @staticmethod
    def add_loss_critic(args):
        NNData.loss_critic_buffer.append(args)
    @staticmethod
    def add_loss_actor(args):
        NNData.loss_actor_buffer.append(args)

    @staticmethod
    def add_loss_buffer(args):
        NNData.loss_buffer.append(args)
    @staticmethod
    def add_entropy(val):
        NNData.entropy_buffer.append(val)
    @staticmethod
    def show_losses():
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_buffer) + 1), NNData.loss_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Total Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(min(NNData.loss_buffer), max(NNData.loss_buffer))
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_actor_buffer) + 1), NNData.loss_actor_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Actor Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(min(NNData.loss_actor_buffer), max(NNData.loss_actor_buffer))
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_critic_buffer) + 1), NNData.loss_critic_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Critic Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(min(NNData.loss_critic_buffer), max(NNData.loss_critic_buffer))
        plt.show()

    @staticmethod
    def show_all_stats():
        # Вызываем твои старые графики (show_losses и т.д.)
        NNData.show_losses()

        # Новый график энтропии
        if len(NNData.entropy_buffer) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(NNData.entropy_buffer) + 1), NNData.entropy_buffer, 'g-', linewidth=2)
            plt.xlabel('Шаг обновления (Update)')
            plt.ylabel('Entropy')
            plt.title('Policy Entropy (Uncertainty)')
            plt.grid(True, alpha=0.3)
            # Для 2-3 действий энтропия обычно лежит в пределах 0.0 - 1.1
            plt.show()


        # График Value Gap (показывает, умнеет ли Критик)
        if NNData.value_gap_buffer:
            plt.figure(figsize=(10, 4))
            plt.plot(NNData.value_gap_buffer, 'r-', label='Value Gap (AA vs 72o)')
            plt.axhline(y=0.5, color='gray', linestyle='--') # Целевой порог
            plt.title("Critic Intelligence (Value Gap)")
            plt.ylabel("Difference in predicted Value")
            plt.legend()
            plt.show()

        # График Action Frequencies (показывает стиль игры)
        if NNData.action_freq_buffer:
            folds = [f['fold'] for f in NNData.action_freq_buffer]
            raises = [f['raise'] for f in NNData.action_freq_buffer]
            calls = [f['call'] for f in NNData.action_freq_buffer]

            plt.figure(figsize=(10, 4))
            plt.stackplot(range(len(folds)), folds, calls, raises,
                          labels=['Fold', 'Call', 'Raise'], colors=['#ff9999','#66b3ff','#99ff99'])
            plt.title("Evolution of Playing Style")
            plt.ylabel("Percentage %")
            plt.legend(loc='upper right')
            plt.show()

        # Новый график: Скользящий Винрейт
        if NNData.rolling_win_rate_history:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(NNData.rolling_win_rate_history) + 1),
                     NNData.rolling_win_rate_history, 'b-', label='Rolling Win Rate')

            # Рисуем горизонтальную линию 50% для ориентира
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.3)

            plt.xlabel('Номер игры')
            plt.ylabel('Win Rate (%)')
            plt.title(f'Moving Average Win Rate (Window: 100 games)')
            plt.grid(True, alpha=0.3)
            plt.ylim(-5, 105) # Чуть шире 100, чтобы видеть края
            plt.legend()
            plt.show()