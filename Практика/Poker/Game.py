import copy
class Game:
    """
    Game class — отвечает за состояние стола:
    - зарегистрированные игроки
    - активные игроки в текущей раздаче
    - минимальная ставка
    - текущие блайнды
    - управление циклическим порядком игроков
    """

    def __init__(self, min_bet=10, players=None):
        self.min_bet = min_bet
        self.registered_players = players[:] if players else []
        self.players = self.registered_players.copy()
        self.betting_players = self.players.copy()
        self.blind_index = -1
        self.manager = None
        self.loosers_list = list()

    def add_player(self, player):
        """Добавляет нового игрока в список зарегистрированных."""
        self.registered_players.append(player)
        self.players.append(player)
        self.betting_players.append(player)

    def get_loosers_list(self):
        return self.loosers_list
    def reset_betting_players(self):

        self.loosers_list.extend([p for p in self.players if p.get_stack() <= 0.1])
        self.players = [p for p in self.players if p.get_stack() > 0.1]
        self.betting_players = self.players.copy()

    def remove_player_betting_round(self, player):
        if player in self.betting_players:
            self.betting_players.remove(player)

    def remove_player(self, player):
        """Удаляет игрока из игры и текущей раздачи."""
        if player in self.players:
            self.players.remove(player)
        self.remove_player_betting_round(player)

    def active_players_count(self):
        """Количество игроков, которые ещё участвуют в раздаче."""
        return len(self.betting_players)

    def next_blinds(self):
        """Переходит к следующему кругу блайндов."""
        if not self.players:
            return
        self.blind_index = (self.blind_index + 1) % len(self.players)

    def get_small_blind_player(self):
        if not self.players:
            return None
        return self.players[self.blind_index]

    def get_big_blind_player(self):
        if not self.players:
            return None
        big_idx = (self.blind_index + 1) % len(self.players)
        return self.players[big_idx]

    def __str__(self):
        players = [str(p) for p in self.players]
        sb = self.get_small_blind_player()
        bb = self.get_big_blind_player()

        return (
            f"Poker Game State:\n"
            f"  Registered players: {len(self.registered_players)} → {players}\n"
            f"  Active in current round: {len(self.players)}\n"
            f"  Min bet: {self.min_bet}\n"
            f"  SB: {sb}\n"
            f"  BB: {bb}\n"
        )

    def __repr__(self):
        return str(self)
    
    
