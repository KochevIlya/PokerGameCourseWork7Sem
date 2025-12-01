from .poker_hands import *

ACTIONS = ['fold', 'call', 'raise']
RAISE_AMOUNT = 10  



category_funcs = (is_royal_flush,
                 is_straight_flush,
                 is_fourkind,
                 is_fullhouse,
                 is_flush,
                 is_straight,
                 is_threekind,
                 is_twopair,
                 is_pair,
                 is_highcard)

category_func_dict = dict(zip(hand_categories, category_funcs))

def post_game(players):
    for player in players:
        player.in_hand = True
        player._holeCards = []
        player._bestHand = None
        player.bet = 0
        player.desicion = ""

def best_hand(lst_hands):
    """
    given a set of poker hands, return the hand with the highest value

    :param lst_hands: list of hands, a hand is a list of five Cards
    :return: best hand in list
    :rtype: list[Card]
    """
    if len(lst_hands) < 2:
        
        return lst_hands[0]
    elif len(lst_hands) == 2:
        match compare_hands(lst_hands[0], lst_hands[1]):
            case 0:
                
                return lst_hands[0]
            case 1:
                return lst_hands[0]
            case 2:
                return lst_hands[1]
    else:
        
        left = best_hand(lst_hands[:len(lst_hands) // 2])
        right = best_hand(lst_hands[len(lst_hands) // 2:])
        return best_hand([left, right])

def compare_hands(hand1, hand2):
    """
    a poker hand is a collection of 5 Cards
    compare two hands to decide which one is better
    :param hand1: list of five Cards, 1st hand
    :param hand2: list of five Cards, 2nd hand

    :return: which hand is better (first=1, second=2, tie=0)
    :rtype: int
    """
    global hand_categories
    h1_category = categorize_hand(hand1)
    h2_category = categorize_hand(hand2)
    if hand_categories.index(h1_category) < hand_categories.index(h2_category):
        return 1
    elif hand_categories.index(h1_category) > hand_categories.index(h2_category):
        return 2
    else:
        # both hands same category
        if h1_category == "RoyalFlush":
            return 0  # royal flush is the highest hand
        elif h1_category == "StraightFlush" or h1_category == "Straight" or h1_category == "Flush":
            # hands sorted largest to smallest card, therefore only need to compare
            # the first card in list to determine which hand is better for five card hands
            return hand1[0].compare(hand2[0])
        elif h1_category == "FourofaKind":  # not a five card hand, must compare the quad then kicker
            # first card could be kicker or part of the quad,
            # so compare the second card which is always part of the quad
            h1_cmp_h2 = hand1[1].compare(hand2[1])
            if h1_cmp_h2 == 0:
                if hand1[0] == hand1[1]:
                    return hand1[-1].compare(hand2[-1])  # the last card was the kicker
                else:
                    return hand1[0].compare(hand2[0])  # the first card was the kicker
            else:
                return h1_cmp_h2
        elif h1_category == "FullHouse":  # not a five card hand, must compare the triple then pair
            # third card is always part of the triple regardless of whether pair is larger or smaller
            h1_cmp_h2 = hand1[2].compare(hand2[2])
            if h1_cmp_h2 == 0:
                if hand1[1] == hand1[2]:
                    return hand1[-1].compare(hand2[-1])  # the last two are the pair
                else:
                    return hand1[0].compare(hand2[0])  # the first two are the pair
            else:
                return h1_cmp_h2
        elif h1_category == "ThreeofaKind":
            # third card is always part of the triple regardless of other two
            h1_cmp_h2 = hand1[2].compare(hand2[2])
            if h1_cmp_h2 == 0:
                if hand1[1] == hand1[2]:
                    # triple is last 3 cards
                    match hand1[0].compare(hand2[0]):
                        case 1: return 1
                        case 2: return 2
                        case 0: return hand1[1].compare(hand2[1])
                else:
                    # triple is first 3 cards
                    match hand1[-2].compare(hand2[-2]):
                        case 1:
                            return 1
                        case 2:
                            return 2
                        case 0:
                            return hand1[-1].compare(hand2[-1])
            else:
                return h1_cmp_h2
        elif h1_category == "TwoPair":
            b, h1_twop = is_twopair(hand1)
            b, h2_twop = is_twopair(hand2)

            h1_cmp_h2 = h1_twop[0].compare(h2_twop[0])  # first pair
            if h1_cmp_h2 == 0:
                cmp = h1_twop[2].compare(h2_twop[2])  # second pair
                if cmp != 0:
                    # compare kicker
                    k1, k2 = None, None
                    for c in hand1:
                        if c not in h1_twop:
                            k1 = c
                            break
                    for c in hand2:
                        if c not in h2_twop:
                            k2 = c
                            break
                    return k1.compare(k2)
                else:
                    return cmp
            else:
                return h1_cmp_h2
        elif h1_category == "Pair":
            b, h1_pair = is_pair(hand1)
            b, h2_pair = is_pair(hand2)
            h1_cmp_h2 = h1_pair[0].compare(h2_pair[0])  # compare pairs
            if h1_cmp_h2 == 0:
                for l, r in zip(hand1, hand2):
                    cmp = l.compare(r)
                    if cmp != 0:
                        return cmp
                return 0
            else:
                return h1_cmp_h2
        else:
            for l, r in zip(hand1, hand2):
                cmp = l.compare(r)
                if cmp != 0:
                    return cmp
            return 0

def categorize_hand(hand):
    """
    assign a category to a poker hand

    :param hand: a list of five Cards
    :return: category of poker hand, one of
    ["RoyalFlush", "StraightFlush", "FourofaKind", "FullHouse",
    "Flush", "Straight", "ThreeofaKind", "TwoPair", "Pair", "HighCard"]
    :rtype: str
    """
    global category_func_dict
    for category, func in category_func_dict.items():
        match, h = func(hand)
        if match:
            return category