import random
from .Player import Player
from .ranking_cards import poker_strength

class SimpleGeneticBot(Player):
    


    def __init__(self, genome, name="Bot",  stack=100,):
        
        super().__init__(name=name, stack=stack)
        self.genome = genome

    def __str__(self):
        return(f"{super().__str__()}\n genome: {self.genome}")
        
        
    def __repr__(self):
        return str(self)

    def evaluate_hand_strength(self, hand, community_cards):
        return poker_strength(hand, community_cards, iters=1000 )

    def make_decision(self, hand, community_cards, min_call):
        hand_strength = self.evaluate_hand_strength(hand, community_cards)
        bluff_rand = random.random()
        
        score = self.genome[0] * hand_strength + self.genome[1] * bluff_rand - self.genome[2] * self.game_bet / (self.game_bet + self.stack)
     
        if score > 0.8:
            return 'raise'
        elif score > 0.4:
            return 'call'
        else:
            return 'fold'

    def get_desicion(self):
        return self.make_desicion()