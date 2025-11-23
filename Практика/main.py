from Poker import *

# player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]
player_names = ["Ilya", "Stas"]
player_managers = []

game = Game()
for name in player_names:
    player = Player(name)
    game.add_player(player)
    #player_managers.append(PlayerManager(player))

gameManager = GameManager(game)
gameManager.start_round()