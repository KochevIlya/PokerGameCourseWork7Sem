import unittest
from itertools import combinations
from .Poker import *
class TestPokerLogic(unittest.TestCase):
    def test_agent_straight_bug(self):
        # Данные из твоего лога (Round 0)
        hole_cards = [Card("♦7"), Card("♥6")]
        board = [Card("♦T"), Card("♦9"), Card("♦5"), Card("♦8"), Card("♦2")]
        all_cards = hole_cards + board
        
        # ВАЖНО: Твой эвалюатор работает только с 5 картами.
        # В Холдеме нужно перебрать все комбинации 5 из 7.
        all_5_card_combinations = list(combinations(all_cards, 5))
        
        # Ищем лучшую категорию среди всех комбинаций
        found_categories = [categorize_hand(list(combo)) for combo in all_5_card_combinations]
        
        self.assertIn("Straight", found_categories, 
                      f"Агент должен был собрать Straight, но найдено: {set(found_categories)}")

    def test_is_straight_logic(self):
        # Проверка конкретно функции стрита на 5 картах
        hand = [Card("♦T"), Card("♦9"), Card("♦8"), Card("♦7"), Card("♦6")]
        match, res = is_straight(hand)
        self.assertTrue(match, "Стрит T-9-8-7-6 должен распознаваться")

if __name__ == '__main__':
    unittest.main()