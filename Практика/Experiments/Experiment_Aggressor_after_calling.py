from Практика.Poker import *
import time
import numpy as np
import matplotlib.pyplot as plt

StaticLogger.configure("Experiment_with_Aggressor_course_after_calling.log", 1000, is_needed=False)

learning_num_games = 50
learning_num_rounds = 50

interval_games = 100
start_time = time.time()
interval_start_time = start_time
checkpoint_interval = 10_000


num_rounds = 30
num_games = 1_000


game_winners = []

players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        #CallingPlayer(),
        NeuralACAgent()
    ]

pm = NeuralACAgentManager(players[1])
pm.load_ac_agent(filename="neural_ac_agent_for_course.pth")
num_wins = { p:0 for p in players}

win_rate_history = []  # История изменения винрейта
games_counter = 0  # Счетчик игр

for i in range(num_games):

    game = Game()

    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds, i)

    if (i + 1) % interval_games == 0:
        current_time = time.time()

        elapsed_interval = current_time - interval_start_time
        gps_current = interval_games / elapsed_interval

        elapsed_total = current_time - start_time
        gps_avg = (i + 1) / elapsed_total

        print(f"[{i+1}/{num_games}] Speed: {gps_current:.2f} games/sec (Avg: {gps_avg:.2f})")
        StaticLogger.print(f"[{i+1}/{num_games}] Speed: {gps_current:.2f} games/sec (Avg: {gps_avg:.2f})")

        interval_start_time = current_time

        if (i + 1) % checkpoint_interval == 0:
            checkpoint_name = f"ac_model_checkpoint_{i+1}.pth"

            # Рекомендую отключить сохранение памяти для промежуточных чекпоинтов,
            # чтобы сэкономить место на диске.
            pm.save_ac_agent(filename=checkpoint_name, save_memory=False)
            StaticLogger.print(f"[*] Промежуточный чекпоинт создан: {checkpoint_name}")

    StaticLogger.print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')



    game_winners.append(winners)

    best_stack = winners[0].get_stack()

    agent = players[1]
    is_agent_winner = (agent.get_stack() == best_stack)
    NNData.record_game_result(is_agent_winner, window_size=100)



    for winner in winners:
        if (winner.stack == best_stack):
            num_wins[winner] += 1

    win_rate = num_wins[players[1]] / (i+1) * 100  # В процентах
    win_rate_history.append(win_rate)



StaticLogger.print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')
StaticLogger.print(f'\033[32mКоличество выигрышей: {num_wins}\033[0m\n')
StaticLogger.print(f"Win rate: {win_rate_history}")
StaticLogger.flush()

# ========== ПРОСТОЙ ГРАФИК WIN RATE ==========
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_games + 1), win_rate_history, 'b-', linewidth=2)
plt.xlabel('Номер игры')
plt.ylabel('Win Rate, %')
plt.title(f'Динамика Win Rate нейросетевого агента ({num_games} игр)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Показываем финальное значение
final_rate = win_rate_history[-1]
plt.axhline(y=final_rate, color='r', linestyle='--', alpha=0.7)

plt.show()

# Только основная статистика
print(f"\nФинальная статистика:")
print(f"Всего игр: {num_games}")
print(f"Побед: {num_wins[players[1]]}")
print(f"Win Rate: {final_rate:.2f}%")

NNData.show_all_stats()