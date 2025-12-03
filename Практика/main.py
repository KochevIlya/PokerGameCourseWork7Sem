from Poker import *

player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
# player_names = ["Ilya", "Stas"]
player_managers = []

num_rounds = 100
num_games = 10

players = [
    SimpleGeneticBot([0.9, 0.1, 0.2], name="Aggressor"),
    SimpleGeneticBot([0.2, 0.1, 0.9], name="Tight"),
    SimpleGeneticBot([0.4, 0.9, 0.4], name="Bluff"),
    SimpleGeneticBot([0.5, 0.5, 0.5], name="Balanced"),
    SimpleGeneticBot([0.9, 0.9, 0.2], name="Maniac"),
]


game_winners = []



for i in range(num_games):

    game = Game()

    players = [
        SimpleGeneticBot([0.9, 0.1, 0.2], name="Aggressor"),
        SimpleGeneticBot([0.2, 0.1, 0.9], name="Tight"),
        SimpleGeneticBot([0.4, 0.9, 0.4], name="Bluff"),
        SimpleGeneticBot([0.5, 0.5, 0.5], name="Balanced"),
        SimpleGeneticBot([0.9, 0.9, 0.2], name="Maniac"),
    ]
    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds)

    print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')
    game_winners.append(winners)

print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')