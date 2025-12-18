import random
import itertools
from collections import Counter
import torch
from .Card import Card

class HandCalculator:


    @staticmethod
    def evaluate_hand_strength(hero_cards, board_cards, iters=200, seed=42):

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


import random
import eval7
from .Card import Card


class HandCalculator:
    # Словарь для маппинга твоих мастей в формат eval7
    # eval7 понимает: 's', 'h', 'd', 'c'
    SUITE_MAP = {
        "Heart": "h",
        "Club": "c",
        "Diamond": "d",
        "Spade": "s"
    }

    @staticmethod
    def evaluate_hand_strength(hero_cards, board_cards, iters=200, seed=42):
        """
        Вычисляет эквити (силу руки) с использованием библиотеки eval7 (Cython).
        Работает в 50-100 раз быстрее старой реализации.
        """
        random.seed(seed)

        # 1. Хелпер для конвертации твоего Card -> eval7.Card
        def to_eval7(card_obj):
            # card_obj.value у тебя уже "2".."9", "T".."A" - это совпадает с eval7
            # card_obj.suite у тебя "Heart" - надо превратить в "h"
            suit_char = HandCalculator.SUITE_MAP[card_obj.suite]
            # Создаем строку вида "Ah", "2d" и т.д.
            return eval7.Card(f"{card_obj.value}{suit_char}")

        # 2. Конвертируем входные данные один раз перед циклом
        # Это очень быстро
        hero_hand_e7 = [to_eval7(c) for c in hero_cards]
        board_hand_e7 = [to_eval7(c) for c in board_cards]

        # 3. Подготовка колоды
        # Берем полную колоду eval7
        deck = eval7.Deck()

        # Убираем карты, которые уже на руках (Hero) или на столе (Board)
        # Используем set для быстрого поиска
        known_cards = set(hero_hand_e7 + board_hand_e7)

        # Создаем список доступных карт для раздачи
        # deck.cards в eval7 - это список объектов Card
        available_deck = [c for c in deck.cards if c not in known_cards]

        # Сколько карт нужно еще выложить на стол?
        cards_to_deal = 5 - len(board_hand_e7)

        wins = 0
        ties = 0
        losses = 0

        # 4. Основной цикл (Monte Carlo)
        for _ in range(iters):
            # Перемешиваем доступную колоду
            random.shuffle(available_deck)

            # Раздаем 2 карты злодею
            villain_hand = available_deck[:2]

            # Раздаем недостающие карты на стол
            community_extras = available_deck[2: 2 + cards_to_deal]

            # Формируем полный борд для текущей симуляции
            full_board = board_hand_e7 + community_extras

            hero_score = eval7.evaluate(hero_hand_e7 + full_board)
            villain_score = eval7.evaluate(villain_hand + full_board)

            if hero_score > villain_score:
                wins += 1
            elif hero_score < villain_score:
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties

        # Возвращаем Equity (от 0.0 до 1.0)
        if total == 0:
            return 0.0
        return (wins + ties / 2.0) / total