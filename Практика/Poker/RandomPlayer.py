from .Player import *

class RandomPlayer(Player):

    def __init__(self, name="RandomPlayer", stack=100):
        super().__init__(name, stack)