from itertools import combinations
from .poker_rules import best_hand, categorize_hand
from .Player import Player


DECISIONS = {
    1: "fold",
    2: "call",
    3: "raise"
}

class PlayerManager:
    """
    Управляет состоянием Player:
    - спрашивает решение у игрока
    - обновляет лучшую руку
    - изменяет состояние (ставка, участие, решение)
    - предоставляет информацию GameManager
    """

    def __init__(self, player: Player):
        self.player = player


    def ask_decision(self, current_bet, min_raise):
        """
        Спрашивает решение у пользователя.
        Вызывается из GameManager.
        """

        print(f"\nИгрок {self.player.name}")
        print(f"Ваши карты: {self.player.hole_cards}")
        print(f"Текущая ставка: {self.player.bet}, стек: {self.player.stack}")
        print(f"На столе ставка {current_bet}. Минимальный рейз: {min_raise}")
        print(f"Ваша лучшая комбинация: {self.player.best_hand}")

        print("Ваш выбор:")
        for code, name in DECISIONS.items():
            print(f"{code} — {name}")

        while True:
            try:
                choice = int(input("Введите номер действия: "))
                if choice in DECISIONS:
                    break
                print("Неверный ввод.")
            except ValueError:
                print("Введите число.")

        decision = DECISIONS[choice]
        self.player.set_decision(decision)
    

    def update_best_hand(self, table_cards):
        """
        Обновляет лучшую комбинацию игрока исходя из table_cards.
        """
        if len(table_cards) < 3:
            self.player.set_best_hand("HighCard")
            return "HighCard"

        all_cards = self.player.hole_cards + table_cards
        combos = [list(combo) for combo in combinations(all_cards, 5)]
        best = best_hand(combos)

        best.sort(reverse=True)
        self.player.set_best_hand(best)
        return best

    def apply_bet(self, amount):
        """
        Делает ставку: уменьшает стек, увеличивает bet.
"""
        amount = min(amount, self.player.stack)
        self.player.stack -= amount
        self.player.bet += amount
        return amount

    def fold(self):
        """Игрок пасует."""
        self.player.in_hand = False
        self.player.set_decision("fold")

    def all_in(self):
        amount = self.player.stack
        amount = self.apply_bet(amount)
        return amount

    def call(self, amount):
        """Игрок уравнивает ставку."""
        to_call = amount - self.player.bet
        to_call = self.apply_bet(to_call)
        self.player.set_decision("call")
        return to_call

    def raise_bet(self, amount):
        """Игрок делает рейз."""
        to_raise = amount - self.player.bet
        to_raise = self.apply_bet(to_raise)
        self.player.set_decision("raise")
        return to_raise

    def can_apply(self, amount):
        if self.player.get_stack()  + self.player.get_bet() <  amount:
            return False
        return True




    def is_active(self):
        return self.player.in_hand

    def get_bet(self):
        return self.player.bet

    def get_best_hand(self):
        return self.player.best_hand

    def get_decision(self):
        return self.player.decision

    def get_stack(self):
        return self.player.get_stack()

    def get_player(self):
        return self.player
    
    def add_stack(self, stack):
        self.player.add_stack(stack)
