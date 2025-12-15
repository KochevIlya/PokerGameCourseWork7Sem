# tests/conftest.py
import pytest
from mocks import *
from Poker.poker_rules import compare_hands

@pytest.fixture(autouse=True)
def setup_hand_categories():
    global hand_categories
    hand_categories = [
        "HighCard",
        "Pair",
        "TwoPair",
        "ThreeofaKind",
        "Straight",
        "Flush",
        "FullHouse",
        "FourofaKind",
        "StraightFlush",
        "RoyalFlush",
    ]
def test_royal_flush_tie():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Spade", "K"),
        MockCard("Spade", "Q"),
        MockCard("Spade", "J"),
        MockCard("Spade", "T"),
    ]
    hand2 = [
        MockCard("Heart", "K"),
        MockCard("Heart", "A"),

        MockCard("Heart", "Q"),
        MockCard("Heart", "J"),
        MockCard("Heart", "T"),
    ]

    assert compare_hands(hand1, hand2) == 0


# =========================
# Straight Flush
# =========================

def test_straight_flush_higher_wins():
    hand1 = [
        MockCard("Spade", "9"),
        MockCard("Spade", "8"),
        MockCard("Spade", "7"),
        MockCard("Spade", "6"),
        MockCard("Spade", "5"),
    ]
    hand2 = [
        MockCard("Heart", "8"),
        MockCard("Heart", "7"),
        MockCard("Heart", "6"),
        MockCard("Heart", "5"),
        MockCard("Heart", "4"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Four of a Kind
# =========================

def test_four_of_a_kind_quad_comparison():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "K"),
        MockCard("Club", "K"),
        MockCard("Diamond", "K"),
        MockCard("Spade", "K"),
    ]
    hand2 = [
        MockCard("Spade", "Q"),
        MockCard("Heart", "Q"),
        MockCard("Club", "Q"),
        MockCard("Diamond", "Q"),
        MockCard("Spade", "A"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Full House
# =========================

def test_full_house_triple_wins():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "A"),
        MockCard("Diamond", "K"),
        MockCard("Spade", "K"),
    ]
    hand2 = [
        MockCard("Spade", "Q"),
        MockCard("Heart", "Q"),
        MockCard("Club", "Q"),
        MockCard("Diamond", "A"),
        MockCard("Spade", "A"),
    ]

    assert compare_hands(hand1, hand2) == 1

def test_two_pairs_win():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "K"),
        MockCard("Diamond", "7"),
        MockCard("Spade", "7"),
    ]
    hand2 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "4"),
        MockCard("Diamond", "7"),
        MockCard("Spade", "7"),
    ]

    assert compare_hands(hand1, hand2) == 1

# =========================
# Flush
# =========================

def test_flush_high_card_comparison():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Spade", "J"),
        MockCard("Spade", "9"),
        MockCard("Spade", "6"),
        MockCard("Spade", "3"),
    ]
    hand2 = [
        MockCard("Heart", "K"),
        MockCard("Heart", "J"),
        MockCard("Heart", "9"),
        MockCard("Heart", "6"),
        MockCard("Heart", "3"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Straight
# =========================

def test_straight_comparison():
    hand1 = [
        MockCard("Spade", "9"),
        MockCard("Heart", "8"),
        MockCard("Club", "7"),
        MockCard("Diamond", "6"),
        MockCard("Spade", "5"),
    ]
    hand2 = [
        MockCard("Spade", "8"),
        MockCard("Heart", "7"),
        MockCard("Club", "6"),
        MockCard("Diamond", "5"),
        MockCard("Spade", "4"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Three of a Kind
# =========================

def test_three_of_a_kind():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "A"),
        MockCard("Diamond", "9"),
        MockCard("Spade", "2"),
    ]
    hand2 = [
        MockCard("Spade", "K"),
        MockCard("Heart", "K"),
        MockCard("Club", "K"),
        MockCard("Diamond", "Q"),
        MockCard("Spade", "2"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Two Pair
# =========================

def test_two_pair_comparison():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "K"),
        MockCard("Diamond", "K"),
        MockCard("Spade", "2"),
    ]
    hand2 = [
        MockCard("Spade", "Q"),
        MockCard("Heart", "Q"),
        MockCard("Club", "J"),
        MockCard("Diamond", "J"),
        MockCard("Spade", "A"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# Pair
# =========================

def test_pair_comparison():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "A"),
        MockCard("Club", "9"),
        MockCard("Diamond", "6"),
        MockCard("Spade", "3"),
    ]
    hand2 = [
        MockCard("Spade", "K"),
        MockCard("Heart", "K"),
        MockCard("Club", "Q"),
        MockCard("Diamond", "6"),
        MockCard("Spade", "3"),
    ]

    assert compare_hands(hand1, hand2) == 1


# =========================
# High Card
# =========================

def test_high_card():
    hand1 = [
        MockCard("Spade", "A"),
        MockCard("Heart", "J"),
        MockCard("Club", "9"),
        MockCard("Diamond", "6"),
        MockCard("Spade", "3"),
    ]
    hand2 = [
        MockCard("Spade", "K"),
        MockCard("Heart", "J"),
        MockCard("Club", "9"),
        MockCard("Diamond", "6"),
        MockCard("Spade", "3"),
    ]

    assert compare_hands(hand1, hand2) == 1



