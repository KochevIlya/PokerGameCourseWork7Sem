class Player:
    """
    Игрок хранит только данные:
    - имя
    - стек
    - карты на руках
    - участие в раздаче
    - ставка
    - лучшая комбинация
    """

    def __init__(self, name="Player", stack=100):
        self.name = str(name)
        self.stack = stack

        self.hole_cards = []
        self.best_hand = None
        self.in_hand = False
        self.bet = 0.0
        self.decision = None


    def reset_for_new_hand(self):
        """Полное обновление состояния для новой раздачи."""
        self.hole_cards = []
        self.best_hand = None
        self.in_hand = False
        self.bet = 0.0
        self.decision = None

    def add_card(self, card):
        """Добавить карту игроку."""
        if len(self.hole_cards) >= 2:
            raise ValueError("Player can only have two hole cards")
        self.hole_cards.append(card)
        if len(self.hole_cards) == 2:
            self.in_hand = True

    def set_best_hand(self, hand):
        self.best_hand = hand

    def set_decision(self, decision):
        self.decision = decision

    def get_stack(self):
        return self.stack

    def get_holecards_notation(self):
        """Возвращает покерную нотацию ('AKs', 'TT', 'QJo')."""
        if len(self.hole_cards) < 2:
            return None

        cards = sorted(self.hole_cards, reverse=True)
        value_str = cards[0].value + cards[1].value

        if cards[0].value == cards[1].value:
            return value_str  

        suited = "s" if cards[0].suite == cards[1].suite else "o"
        return value_str + suited

    def __str__(self):
        return f"{self.name}: hole={self.hole_cards}, best={self.best_hand}"

    def __repr__(self):
        return str(self)
    
    def add_stack(self, stack):
        self.stack += stack;
    