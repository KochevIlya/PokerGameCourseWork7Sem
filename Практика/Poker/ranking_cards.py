import random
import itertools
from collections import Counter
from .Card import Card

def poker_strength(hero_cards, board_cards, iters=5000, seed=42):
    
    random.seed(seed)
    
    VALUE_MAP = {str(n): n for n in range(2, 10)}
    VALUE_MAP.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14})

    def create_deck():
        return [Card(suite, value) for value in Card.static_cardvalues for suite in Card.static_suites]

    def hand_rank(hand):
        ranks = sorted([VALUE_MAP[c.value] for c in hand], reverse=True)
        suits = [c.suite for c in hand]
        counts = Counter(ranks)
        rankcounts = counts.most_common()
        is_flush_ = len(set(suits)) == 1
        is_straight_ = False
        ranks_set = sorted(set(ranks), reverse=True)
        for i in range(len(ranks_set) - 4 + 1):
            window = ranks_set[i:i + 5]
            if len(window) == 5 and window[0] - window[4] == 4:
                is_straight_ = True
                break
        if not is_straight_ and set([14, 2, 3, 4, 5]).issubset(set(ranks)):
            is_straight_ = True
            ranks = [5, 4, 3, 2, 1]
        if is_straight_ and is_flush_:
            return (8, ranks)
        if rankcounts[0][1] == 4:
            return (7, [rankcounts[0][0], rankcounts[1][0]])
        if rankcounts[0][1] == 3 and rankcounts[1][1] == 2:
            return (6, [rankcounts[0][0], rankcounts[1][0]])
        if is_flush_:
            return (5, ranks)
        if is_straight_:
            return (4, ranks)
        if rankcounts[0][1] == 3:
            kickers = [r for r in ranks if r != rankcounts[0][0]]
            return (3, [rankcounts[0][0]] + kickers)
        if rankcounts[0][1] == 2 and rankcounts[1][1] == 2:
            kicker = [r for r in ranks if r != rankcounts[0][0] and r != rankcounts[1][0]][0]
            return (2, sorted([rankcounts[0][0], rankcounts[1][0]], reverse=True) + [kicker])
        if rankcounts[0][1] == 2:
            kickers = [r for r in ranks if r != rankcounts[0][0]]
            return (1, [rankcounts[0][0]] + kickers)
        return (0, ranks)

    def best_hand(cards):
        best = None
        for comb in itertools.combinations(cards, 5):
            r = hand_rank(list(comb))
            if best is None or r > best:
                best = r
        return best

    used = set(hero_cards + board_cards)
    deck = [c for c in create_deck() if c not in used]

    wins = 0
    ties = 0
    losses = 0

    board_count = len(board_cards)
    to_deal = 5 - board_count

    for _ in range(iters):
        deck_shuffled = deck[:]
        random.shuffle(deck_shuffled)
        villain = deck_shuffled[:2]
        left = deck_shuffled[2:]
        extra_board = left[:to_deal]
        full_board = board_cards + list(extra_board)
        hero_best = best_hand(hero_cards + full_board)
        villain_best = best_hand(villain + full_board)
        if hero_best > villain_best:
            wins += 1
        elif hero_best < villain_best:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties

    return (wins + ties/2) / total if total else 0.0

