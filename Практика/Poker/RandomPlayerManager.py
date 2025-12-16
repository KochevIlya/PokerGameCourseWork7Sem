from .PlayerManager import *
import random


class RandomPlayerManager(PlayerManager):

    def __init__(self, player):
        super().__init__(player)

    def ask_decision(self, current_bet, min_raise):
        decision = self.DECISIONS[random.randint(1, len(self.DECISIONS))]
        self.player.set_decision(decision)

        StaticLogger.print(f"\nИгрок {self.player.name}")
        StaticLogger.print(f"Ваши карты: {self.player.hole_cards}")
        StaticLogger.print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        StaticLogger.print(f"На столе ставка {current_bet}. Минимальный рейз: {min_raise}")
        StaticLogger.print(f"Ваша лучшая комбинация: {self.player.best_hand}")
        StaticLogger.print(f"Ваш выбор: {self.player.decision}")


