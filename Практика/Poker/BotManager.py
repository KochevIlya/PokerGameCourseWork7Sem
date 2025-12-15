from .PlayerManager import PlayerManager
from .bot import SimpleGeneticBot
from .HandCalculator import *
import random
from .Logger import *
class BotManager(PlayerManager):
    
    def __init__(self, bot: SimpleGeneticBot):
        super().__init__(bot)

    def ask_decision(self, current_bet, min_raise, community_cards):

        hand_strength = HandCalculator.evaluate_hand_strength(self.player.hole_cards, community_cards)
        bluff_rand = random.random()
        
        score = self.player.genome[0] * hand_strength + self.player.genome[1] * bluff_rand + (1/len(self.player.genome) - self.player.genome[2] * current_bet / (current_bet + self.player.get_stack()))

        StaticLogger.print(f"\nИгрок {self.player.name}")
        StaticLogger.print(f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        StaticLogger.print(f"На столе ставка {current_bet}. Минимальный рейз: {min_raise}")
        StaticLogger.print(f"Ваша лучшая комбинация: {self.player.best_hand}")

        if score > 0.8:
            self.player.decision = "raise"
        elif score > 0.3:
            self.player.decision = "call"
        else:
            self.player.decision = "fold"
        StaticLogger.print(f"Ваш выбор: {self.player.decision}")
        return self.player.decision
