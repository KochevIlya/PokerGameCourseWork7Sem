from .Player import *

class CallingPlayer(Player):

    def __init__(self, name="CallingPlayer", stack=100):
        super().__init__(name, stack)