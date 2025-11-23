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
        self.blind_index = 0
        self.manager = None


    def add_player(self, player):
        """Добавляет нового игрока в список зарегистрированных."""
        self.registered_players.append(player)
        self.players.append(player)

    def remove_player(self, player):
        """Удаляет игрока из игры и текущей раздачи."""
        if player in self.registered_players:
            self.registered_players.remove(player)
        if player in self.players:
            self.players.remove(player)

    def active_players_count(self):
        """Количество игроков, которые ещё участвуют в раздаче."""
        return len(self.players)

    def next_blinds(self):
        """Переходит к следующему кругу блайндов."""
        if not self.registered_players:
            return
        self.blind_index = (self.blind_index + 1) % len(self.registered_players)

    def get_small_blind_player(self):
        if not self.registered_players:
            return None
        return self.registered_players[self.blind_index]

    def get_big_blind_player(self):
        if not self.registered_players:
            return None
        big_idx = (self.blind_index + 1) % len(self.registered_players)
        return self.registered_players[big_idx]

    def __str__(self):
        players = [str(p) for p in self.registered_players]
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
    
    
