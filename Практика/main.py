from Poker import *

player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
# player_names = ["Ilya", "Stas"]
player_managers = []
num_games = 10

game = Game()
for name in player_names:
    player = SimpleGeneticBot((0.8, 0.2, 0.4), name)
    game.add_player(player)
    #player_managers.append(PlayerManager(player))

gameManager = GameManager(game)

for _ in range(num_games):
    print(f'\033[32mВыигрывает(ют): {gameManager.start_round()}\033[0m\n')