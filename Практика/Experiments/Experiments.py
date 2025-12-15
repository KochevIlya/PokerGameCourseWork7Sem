from Практика.Poker import *

# player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
# player_names = ["Ilya", "Stas"]
# player_managers = []

StaticLogger.configure("Experiments.log", 1000)


learning_num_games = 50
learning_num_rounds = 50


num_rounds = 30
num_games = 20000



# players = [
#         SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
#         SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
#         SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
#         SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
#         SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
#     ]

game_winners = []

# bot_fabric = BotFabric()
# bot_fabric.fit(learning_num_games, learning_num_rounds)

players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        NeuralAgent()
    ]
num_wins = { p:0 for p in players }

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


    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds)

    StaticLogger.print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')
    game_winners.append(winners)
    game_winners.append(winners)
    best_stack = winners[0].get_stack()
    for winner in winners:
        if (winner.stack == best_stack):
            num_wins[winner] += 1


StaticLogger.print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')
StaticLogger.print(f'\033[32mКоличество выигрышей: {num_wins}\033[0m\n')
StaticLogger.flush()