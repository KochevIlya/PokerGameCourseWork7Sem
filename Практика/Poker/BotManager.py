from .PlayerManager import PlayerManager
from .bot import SimpleGeneticBot
import random
class BotManager(PlayerManager):
    
    def __init__(self, bot: SimpleGeneticBot):
        super().__init__(bot)

    def ask_decision(self, current_bet, min_raise, community_cards):

        hand_strength = self.player.evaluate_hand_strength(self.player.hole_cards, community_cards)
        bluff_rand = random.random()
        
        score = self.player.genome[0] * hand_strength + self.player.genome[1] * bluff_rand - self.player.genome[2] * current_bet / (current_bet + self.player.get_stack())

        print(f"\nИгрок {self.player.name}")
        print(f"Ваши карты: {self.player.hole_cards}")
        print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        print(f"На столе ставка {current_bet}. Минимальный рейз: {min_raise}")
        print(f"Ваша лучшая комбинация: {self.player.best_hand}")

        if score > 0.8:
            self.player.decision = "raise"
        elif score > 0.4:
            self.player.decision = "call"
        else:
            self.player.decision = "fold"
        print(f"Ваш выбор: {self.player.decision}")
        return self.player.decision
