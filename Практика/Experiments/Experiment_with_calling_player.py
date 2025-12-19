from Практика.Poker import *
import numpy as np
import matplotlib.pyplot as plt

StaticLogger.configure("Experiment_with_calling_player_new.log", 1000)

learning_num_games = 50
learning_num_rounds = 50


num_rounds = 30
num_games = 30000


game_winners = []

# bot_fabric = BotFabric()
# bot_fabric.fit(learning_num_games, learning_num_rounds)
players = [
        CallingPlayer(),
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

    StaticLogger.print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')

    game_winners.append(winners)
    best_stack = winners[0].get_stack()
    for winner in winners:
        if (winner.stack == best_stack):
            num_wins[winner] += 1

    win_rate = num_wins[players[1]] / (i+1) * 100  # В процентах
    win_rate_history.append(win_rate)

StaticLogger.print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')
StaticLogger.print(f'\033[32mКоличество выигрышей: {num_wins}\033[0m\n')
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