class Card:
    suite = ""
    value = ""
    static_suites = ["Heart", "Club", "Diamond", "Spade"]
    static_cardvalues = [str(n) for n in range(2, 10)] + ["T", "J", "Q", "K", "A"]

    def __init__(self, suite, value):
        if suite not in Card.static_suites:
            raise ValueError("Invalid suite " + suite)
        if value not in Card.static_cardvalues:
            raise ValueError("Invalid card value " + value)
        self.suite = suite
        self.value = value

    def __sub__(self, other):
        """ subtract rank / positional value of cards, positive result means 
        left operand higher card than right """
        if self.value == "A" and other.value == "2":
            return -1
        if self.value == "2" and other.value == "A":
            return 1
        return Card.static_cardvalues.index(self.value) - Card.static_cardvalues.index(other.value)

    def __gt__(self, other):
        """ is this (self) a higher card than the other card """
        return Card.static_cardvalues.index(self.value) > Card.static_cardvalues.index(other.value)

    def __ge__(self, other):
        """ is this (self) card higher or equal to other card """
        return Card.static_cardvalues.index(self.value) >= Card.static_cardvalues.index(other.value)

    def __lt__(self, other):
        """ is this (self) a lower card than the other card """
        return Card.static_cardvalues.index(self.value) < Card.static_cardvalues.index(other.value)

    def __le__(self, other):
        """ is this (self) a lower or equal to the other card """
        return Card.static_cardvalues.index(self.value) <= Card.static_cardvalues.index(other.value)

    def __eq__(self, other):
        """ is this (self) card equal to the other card """
        return Card.static_cardvalues.index(self.value) == Card.static_cardvalues.index(other.value)

    def __str__(self):
        match self.suite:
            case "Heart":
                return '\u2665' + self.value
            case "Club":
                return '\u2663' + self.value
            case "Spade":
                return '\u2660' + self.value
            case "Diamond":
                return '\u2666' + self.value
            case _:
                raise ValueError("Invalid card suite")

    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash((self.value, self.suite))

    def compare(self, other):
        """
        return 1 if greater, 2 if smaller, 0 if same
        :param other: another Card object
        :return: 1, 2 or 0
        """
        if self > other:
            return 1
        elif self < other:
            return 2
        else:
            return 0
        