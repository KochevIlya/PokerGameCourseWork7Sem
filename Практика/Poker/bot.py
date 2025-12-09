import random
from .Player import Player
from .HandCalculator import *

class SimpleGeneticBot(Player):
    


    def __init__(self, genome, name="Bot",  stack=100,):
        
        super().__init__(name=name, stack=stack)
        self.genome = genome

    def __str__(self):
        return(f"{super().__str__()} genome: {self.genome}\n")
        
        
    def __repr__(self):
        return str(self)


    def make_decision(self, hand, community_cards, min_call):
        hand_strength = HandCalculator.evaluate_hand_strength(hand, community_cards)
        bluff_rand = random.random()
        
        score = self.genome[0] * hand_strength + self.genome[1] * bluff_rand + (1 - self.genome[2] * self.get_bet() / (self.get_bet() + self.stack))
     
        if score > 0.6:
            return 'raise'
        elif score > 0.3:
            return 'call'
        else:
            return 'fold'

    def get_genome(self):
        return self.genome