from Poker import *

player_names = ["Ilya", "Stas", "Matvei", "Anton", "Artem", "Alex", "Nikita", "Semen"]


game = Game()
for name in player_names:
    game.add_player(Player(name))

print(game)