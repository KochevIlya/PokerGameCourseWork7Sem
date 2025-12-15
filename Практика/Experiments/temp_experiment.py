import sys
import os

from sympy import false

# 1. СНАЧАЛА настройте логгер
from Практика.Poker.Logger import StaticLogger  # Импортируем только логгер
StaticLogger.configure("Genetic_bots_experiment.log", 1000, False)

# 2. Проверьте настройки
print(f"Логгер настроен: файл={StaticLogger._filename}, буфер={StaticLogger._buffer_size}")

# 3. ПОТОМ импортируйте всё остальное
from Практика.Poker import *

# player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
# player_names = ["Ilya", "Stas"]
# player_managers = []

StaticLogger.configure("Genetic_bots_experiment.log", 1000)



learning_num_games = 1
learning_num_rounds = 30


num_rounds = 30
num_games = 1

aggressor = SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor")
tight = SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight")
bluff = SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff")
balanced = SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced")
maniac = SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac")
tactics = [
    aggressor,
    tight,
    bluff,
    balanced,
    maniac,
]

players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
        SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
        SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
        SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
    ]



game_winners = []
num_wins = { p:0 for p in tactics }


bot_fabric = BotFabric()
bot_fabric.fit(learning_num_games, learning_num_rounds)


for i in range(num_games):

    game = Game()

    # players = [
    #     SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
    #     SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
    #     SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
    #     SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
    #     SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
    #     bot_fabric.get_result_bot()
    # ]

    # players = [
    #     SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
    #     NeuralAgent()
    # ]

    for player in tactics:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds)

    StaticLogger.print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')

    game_winners.append(winners)
    best_stack = winners[0].get_stack()
    for winner in winners:
        if(winner.stack == best_stack):
            num_wins[winner] += 1



StaticLogger.print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')
StaticLogger.print(f'\033[32mКоличество выигрышей: {num_wins}\033[0m\n')
StaticLogger.flush()