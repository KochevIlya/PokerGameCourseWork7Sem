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


    # def evaluate_hand_strength_torch(hero_cards, board_cards, iters=10000, device='cuda'):
    #     """
    #     Args:
    #         hero_cards (list[Card]): Список из 2 объектов класса Card
    #         board_cards (list[Card]): Список из 0-5 объектов класса Card
    #         iters (int): Количество симуляций
    #         device (str): 'cuda' или 'cpu'
    #     """
    #
    #     # 1. Проверка устройства
    #     if device == 'cuda' and not torch.cuda.is_available():
    #         print("Warning: CUDA не найдена, вычисления будут на CPU (медленно).")
    #         device = 'cpu'
    #
    #     # 2. Маппинг твоего класса Card в числа (0..51)
    #     # Используем статические поля твоего класса для надежности
    #     # Ranks: 2=0, 3=1 ... A=12
    #     # Suites: Heart=0, Club=1, Diamond=2, Spade=3 (порядок важен только для уникальности)
    #
    #     # Создаем словари для быстрого поиска индексов
    #     try:
    #         RANK_TO_INT = {v: i for i, v in enumerate(Card.static_cardvalues)}
    #         SUITE_TO_INT = {s: i for i, s in enumerate(Card.static_suites)}
    #     except NameError:
    #         print("Ошибка: Класс Card не найден. Убедись, что он определен перед вызовом функции.")
    #         return 0.0
    #
    #     def encode_card(card_obj):
    #         r = RANK_TO_INT[card_obj.value]
    #         s = SUITE_TO_INT[card_obj.suite]
    #         # Формула: rank * 4 + suit. Это дает уникальный ID от 0 до 51
    #         return r * 4 + s
    #
    #     # Переводим объекты карт в тензоры
    #     try:
    #         h_list = [encode_card(c) for c in hero_cards]
    #         b_list = [encode_card(c) for c in board_cards]
    #     except KeyError as e:
    #         print(f"Ошибка в данных карты: {e}. Проверьте value/suite.")
    #         return 0.0
    #
    #     hero_tensor = torch.tensor(h_list, device=device, dtype=torch.long)
    #     board_tensor = torch.tensor(b_list, device=device, dtype=torch.long)
    #
    #     # --- Дальше идет векторизованная логика (та же, что и раньше) ---
    #
    #     # Полная колода
    #     full_deck = torch.arange(52, device=device, dtype=torch.long)
    #
    #     # Маска использованных карт
    #     used_mask = torch.zeros(52, device=device, dtype=torch.bool)
    #     used_mask[hero_tensor] = True
    #     used_mask[board_tensor] = True
    #
    #     available_deck = full_deck[~used_mask]
    #     n_available = available_deck.shape[0]
    #
    #     # Сколько карт нужно сдать (оппонент + остаток борда)
    #     cards_to_deal = 2 + (5 - len(b_list))
    #
    #     # Генерируем случайные индексы
    #     rand_vals = torch.rand((iters, n_available), device=device)
    #     perm_indices = torch.argsort(rand_vals, dim=1)[:, :cards_to_deal]
    #
    #     dealt_cards = available_deck[perm_indices]  # (iters, cards_to_deal)
    #
    #     villain_hole = dealt_cards[:, :2]  # Первые 2 - врагу
    #     runout = dealt_cards[:, 2:]  # Остальные - на стол
    #
    #     # Формируем 7 карт для каждого
    #     hero_exp = hero_tensor.unsqueeze(0).expand(iters, -1)
    #     board_exp = board_tensor.unsqueeze(0).expand(iters, -1)
    #
    #     # Собираем общий борд
    #     if len(b_list) > 0:
    #         if runout.shape[1] > 0:
    #             full_board = torch.cat([board_exp, runout], dim=1)
    #         else:
    #             full_board = board_exp
    #     else:
    #         full_board = runout
    #
    #     hero_7 = torch.cat([hero_exp, full_board], dim=1)
    #     villain_7 = torch.cat([villain_hole, full_board], dim=1)
    #
    #     # Индексы комбинаций (7 по 5)
    #     combos_idx = torch.tensor(list(itertools.combinations(range(7), 5)), device=device, dtype=torch.long)
    #
    #     def get_batch_scores(hands_7):
    #         # hands_7 shape: (Batch, 7)
    #         # Выбираем 5 карт из 7 всеми способами: (Batch, 21, 5)
    #         hands_5 = hands_7[:, combos_idx]
    #
    #         ranks = hands_5 // 4
    #         suits = hands_5 % 4
    #
    #         # Сортируем ранги для проверок
    #         ranks_sorted, _ = torch.sort(ranks, dim=2, descending=True)
    #
    #         # 1. Flush
    #         is_flush = (suits.max(dim=2).values == suits.min(dim=2).values)
    #
    #         # 2. Straight
    #         rank_diff = ranks_sorted[:, :, 0] - ranks_sorted[:, :, 4]
    #         # Проверка на обычный стрит (разница 4 и все уникальны)
    #         diffs = ranks_sorted[:, :, :-1] - ranks_sorted[:, :, 1:]
    #         is_unique = (diffs > 0).all(dim=2)
    #         is_standard_straight = (rank_diff == 4) & is_unique
    #
    #         # Проверка на колесо (A, 5, 4, 3, 2 -> 12, 3, 2, 1, 0)
    #         wheel_pattern = torch.tensor([12, 3, 2, 1, 0], device=device).view(1, 1, 5)
    #         is_wheel = (ranks_sorted == wheel_pattern).all(dim=2)
    #
    #         is_straight = is_standard_straight | is_wheel
    #
    #         # 3. Counts (Pair, Trips, Quads)
    #         ranks_one_hot = torch.nn.functional.one_hot(ranks, num_classes=13)
    #         counts_hist = ranks_one_hot.sum(dim=2)  # (Batch, 21, 13)
    #
    #         # Сортируем "веса": count * 100 + rank
    #         weights = counts_hist * 100 + torch.arange(13, device=device).unsqueeze(0).unsqueeze(0)
    #         sorted_weights, _ = torch.sort(weights, dim=2, descending=True)
    #
    #         c1 = sorted_weights[:, :, 0] // 100
    #         c2 = sorted_weights[:, :, 1] // 100
    #
    #         is_quads = (c1 == 4)
    #         is_fh = (c1 == 3) & (c2 == 2)
    #         is_trips = (c1 == 3) & (c2 == 1)
    #         is_2pair = (c1 == 2) & (c2 == 2)
    #         is_pair = (c1 == 2) & (c2 == 1)
    #
    #         is_str_flush = is_straight & is_flush
    #
    #         # --- Scoring ---
    #         # Восстанавливаем ранги кикеров
    #         r1 = sorted_weights[:, :, 0] % 100
    #         r2 = sorted_weights[:, :, 1] % 100
    #         r3 = sorted_weights[:, :, 2] % 100
    #         r4 = sorted_weights[:, :, 3] % 100
    #         r5 = sorted_weights[:, :, 4] % 100
    #
    #         # Если Wheel, то старшая карта для стрита это 5 (rank index 3), а не Туз
    #         r1_straight = torch.where(is_wheel, torch.tensor(3, device=device), r1)
    #
    #         kickers_score = (r1 * 16 ** 4) + (r2 * 16 ** 3) + (r3 * 16 ** 2) + (r4 * 16 ** 1) + r5
    #         kickers_score_str = r1_straight * 16 ** 4
    #
    #         final_score = torch.zeros_like(is_flush, dtype=torch.float)
    #
    #         # Приоритеты комбинаций
    #         final_score = torch.where(is_str_flush, 8e10 + kickers_score_str, final_score)
    #
    #         mask = (~is_str_flush) & is_quads
    #         final_score = torch.where(mask, 7e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_quads) & is_fh
    #         final_score = torch.where(mask, 6e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_fh) & is_flush
    #         final_score = torch.where(mask, 5e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_flush) & is_straight
    #         final_score = torch.where(mask, 4e10 + kickers_score_str, final_score)
    #
    #         mask = mask & (~is_straight) & is_trips
    #         final_score = torch.where(mask, 3e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_trips) & is_2pair
    #         final_score = torch.where(mask, 2e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_2pair) & is_pair
    #         final_score = torch.where(mask, 1e10 + kickers_score, final_score)
    #
    #         mask = mask & (~is_pair)
    #         final_score = torch.where(mask, kickers_score, final_score)
    #
    #         return final_score.max(dim=1)[0]  # Max over 21 combinations
    #
    #     hero_scores = get_batch_scores(hero_7)
    #     villain_scores = get_batch_scores(villain_7)
    #
    #     wins = (hero_scores > villain_scores).sum()
    #     ties = (hero_scores == villain_scores).sum()
    #
    #     return ((wins + ties / 2.0) / iters).item()
    #
    # # --- Пример запуска ---
    # if __name__ == "__main__":
    #     # Создаем карты через твой класс
    #     hero = [Card("Heart", "A"), Card("Diamond", "K")]
    #     board = [Card("Spade", "T"), Card("Spade", "J"), Card("Spade", "Q")]  # Royal Flush draw?
    #
    #     # Если CUDA починишь, код сам переключится. Пока сработает и на CPU, но медленнее.
    #     equity = evaluate_hand_strength_torch(hero, board, iters=5000)
    #     print(f"Equity: {equity * 100:.2f}%")