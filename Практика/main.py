from Poker import *

player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
# player_names = ["Ilya", "Stas"]
player_managers = []

num_rounds = 10
num_games = 10

players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
        SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
        SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
        SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
    ]


game_winners = []



for i in range(num_games):

    game = Game()

    players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
        SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
        SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
        SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
    ]
    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds)

    print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')
    game_winners.append(winners)

print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')