from .Card import Card
import time
import random

class Deck:
    def __init__(self):
        self._deck = []
        for s in Card.static_suites:
            for c in Card.static_cardvalues:
                self._deck.append(Card(s, c))

    def shuffle(self):
        random.seed(time.time())
        for _ in range(random.randint(1, 5)):
            random.shuffle(self._deck)

    def get_cards(self):
        """return list of cards in deck"""
        return self._deck
    
    def dealcard(self):
        return self._deck.pop()            

    def __len__(self):
        return len(self._deck)
    
    def __str__(self):
        return " ".join([str(crd) for crd in self._deck])
    
    def burn(self):
        self.dealcard()
    