from itertools import combinations
from .poker_rules import best_hand, categorize_hand
desicions = {
            1 : "fold",
            2 : "call",
            3 : "raise"
        }

class Player:
    def __init__(self, name="Player", stack=100, min_bet=10):
        self.name = str(name)
        self._holeCards = []
        self.stack = stack
        self.in_hand = False
        self.bet = 0
        self.desicion = ""
        self._bestHand = None

    def __str__(self):
        if not self._bestHand:
            return self.name + ":" + str(self._holeCards)
        else:
            return self.name + ":" + str(self._bestHand) + "," + categorize_hand(self._bestHand)
        
    def __repr__(self):
        return str(self)

    def ask_player(self):
        print(f'Ваши карты: {self._holeCards}, лучшая комбинация: {self._bestHand} вы поставили {self.bet}')
        
        print(f'Сделайте выбор:')
        for desicion in desicions:
            print(desicion)
        desicion = int(input())
        self.make_decision(desicion)

    def reset_for_new_hand(self):
        self._holeCards = []
        self._bestHand = None
        self.in_hand = False
        self.bet = 0
        self.desicion = ""

    def add_card(self, c):
        if len(self._holeCards) < 2:
            self._holeCards.append(c)
        else:
            raise ValueError("Player can only have two hole cards")
        if(len(self._holeCards == 2)):
            self._bestHand = "HighCard"
            self.in_hand = True

    def make_decision(self, desicion, desicions):
        self.desicion = desicions[desicion]


    def update_best_hand(self, table):
        
        if len(table) < 3:
            self._bestHand = "HighCard"
            return self._bestHand
        if len(table) >= 3:
            lst_hands = [list(combo) for combo in combinations(self._holeCards + table, 5)]
            self._bestHand = best_hand(lst_hands)
            self._bestHand.sort(reverse=True)
            return self._bestHand    

    def get_holecards(self):
        return self._holeCards

    def get_best_hand(self):
        if not self._bestHand:
            raise ValueError("Best hand undetermiend. Call update_best_hand")
        return self._bestHand

    def get_desicion(self):
        return self.desicion

    def get_holecards_pokernotation(self):
        
        self._holeCards.sort(reverse=True)
        poker_notation = self._holeCards[0].value + self._holeCards[1].value
        if poker_notation[0] == poker_notation[1]:
            return poker_notation
        else:
            if self._holeCards[0].suite == self._holeCards[1].suite:
                poker_notation = poker_notation + "s"
            else:
                poker_notation = poker_notation + "o"
            return poker_notation