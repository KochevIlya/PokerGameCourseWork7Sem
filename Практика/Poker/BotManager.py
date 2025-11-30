from Poker import *

class BotManager(PlayerManager):
    
    def __init__(self, bot: SimpleGeneticBot):
        self.bot = bot

    def ask_decision(self, current_bet, min_raise, community_cards):

        hand_strength = self.bot.evaluate_hand_strength(self.bot.hand, community_cards)
        bluff_rand = random.random()
        
        score = self.bot.genome[0] * hand_strength + self.bot.genome[1] * bluff_rand - self.bot.genome[2] * current_bet / (current_bet + self.stack)
     
        if score > 0.8:
            return 'raise'
        elif score > 0.4:
            return 'call'
        else:
            return 'fold'