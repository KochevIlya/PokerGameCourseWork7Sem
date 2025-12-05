
from .Deck import Deck
from .PlayerManager import PlayerManager
from .Game import Game
from .Player import Player
from .BotManager import BotManager
from .bot import SimpleGeneticBot

class GameManager:
    """
    GameManager управляет:
    - началом новой раздачи
    - постановкой блайндов
    - очередностью ставок
    - стадиями (preflop / flop / turn / river)
    - сбором решений игроков через PlayerManager
    - определением победителя
    """

    def __init__(self, game: Game):
        """
        game                — объект Game
        """
        self.game = game
        self.current_bet = 0.0
        # Менеджеры на каждого игрока
        self.pm = dict()
        for player in self.game.players:
            if isinstance(player, SimpleGeneticBot):
                self.pm[player] = BotManager(player)
            else:
                self.pm[player] = PlayerManager(player)

        self.table = []
        self.pot = 0.0

    def start_game(self, num_rounds):
        for i in range(num_rounds):
            print(f"Round {i}\n")
            print(f"{str(self.game)}\n")
            self._prepare_round()


            if len(self.game.players) <= 1:
                print(f"Game is over, because of the players amount")
                self.game.registered_players.sort(key=lambda p: p.stack, reverse=True)
                return self.game.registered_players
            else:
                self.game.next_blinds()
                print(f"After preparing: {str(self.game)}\n")
                print(f'\033[32mВыигрывает(ют): {self.start_round()}\033[0m\n')
        print(f"Game is over, because of the rounds amount")
        self.game.registered_players.sort(key=lambda p: p.stack, reverse=True)
        return self.game.registered_players


    def start_round(self):
        """
        Основной игровой цикл одной раздачи.
        """

        self._post_blinds()
        
        self._betting_round(start_from="UTG")
        
        self._deal_flop()
        self.show_current_situation()
        self._betting_round(start_from="SB")

        self._deal_turn()
        self.show_current_situation()
        self._betting_round(start_from="SB")

        self._deal_river()
        self._betting_round(start_from="SB")


        winners = self._determine_winner()

        self.winners_distribution(winners)


        return winners

    def winners_distribution(self, winners: list[Player]):

        for winner in winners:
            winner.add_stack(self.pot / len(winners))




    def show_current_situation(self):
        print(f"\nКарты на столе: {self.table}")
        

    def _prepare_round(self):
        """Сбрасывает всё состояние перед новой раздачей."""

        self.table = []
        self.pot = 0.0
        for p in self.game.players:
            p.reset_for_new_hand()

        self.deck = Deck()
        self.deck.shuffle()

        self.game.reset_betting_players()

        # Раздача холд-карт
        for p in self.game.players:
            p.add_card(self.deck.dealcard())
            p.add_card(self.deck.dealcard())


    def _post_blinds(self):
        """Ставит малый и большой блайнд согласно Game.blind_index."""

        sb_player = self.game.get_small_blind_player()
        bb_player = self.game.get_big_blind_player()

        sb_pm = self.pm[sb_player]
        bb_pm = self.pm[bb_player]

        sb_amount = self.game.min_bet // 2
        bb_amount = self.game.min_bet

        sb_amount = sb_pm.apply_bet(sb_amount)
        bb_amount = bb_pm.apply_bet(bb_amount)
        self.pot += sb_amount + bb_amount

        sb_player.set_decision("sb")
        bb_player.set_decision("bb")


        self.current_bet = max(bb_amount, sb_amount)

        


    def _betting_round(self, start_from="SB"):
        """
        Один круг ставок.
        start_from:
            "UTG" — сразу после BB (префлоп)
            "SB" — с малого блайнда (флоп/терн/ривер)
        """

        active_players = self.game.betting_players
        if len(active_players) <= 1:
            return

        order = self._betting_order(start_from)
        players_to_act = set(order)

        while players_to_act:
            for player in order:
                if player not in players_to_act:
                    continue
                if self.game.active_players_count() == 1:
                    continue
                if not player.in_hand:
                    players_to_act.remove(player)
                    continue

                pm = self.pm[player]

                self.show_current_situation()
                if  isinstance(pm, BotManager):
                    pm.ask_decision(
                        current_bet=self.current_bet,
                        min_raise=self.game.min_bet,
                        community_cards=self.table.copy()
                    )

                elif isinstance(pm, PlayerManager):
                    pm.ask_decision(
                        current_bet=self.current_bet,
                        min_raise=self.game.min_bet
                    )



                if player.decision == "fold":
                    pm.fold()
                    players_to_act.remove(player)
                    self.game.remove_player_betting_round(player)
                    print(pm.player)
                    continue

                if player.decision == "raise":

                    needed = self.current_bet + self.game.min_bet

                    if not pm.can_apply(needed):
                        player.set_decision("call")

                    else:
                        to_raise = pm.raise_bet(needed)
                        self.current_bet += self.game.min_bet
                        self.pot += to_raise

                        players_to_act = set(order)
                        players_to_act.remove(player)
                        print(pm.player)
                        continue

                if player.decision == "call":

                    needed = self.current_bet
                    if pm.can_apply(needed):
                        to_call = pm.call(needed)
                        self.pot += to_call
                        players_to_act.remove(player)
                        print(pm.player)

                    else:
                        to_amount = pm.all_in()
                        self.pot += to_amount
                        players_to_act.remove(player)
                        print(pm.player)


            still_in = self.game.active_players_count()
            if still_in <= 1:
                break


    def _betting_order(self, start_from):
        """Создаёт порядок игроков для ставок."""

        players = self.game.betting_players
        sb_index = self.game.blind_index
        bb_index = (sb_index + 1) % len(players)

        if start_from == "UTG":
            start = (bb_index + 1) % len(players)
        else:
            start = sb_index

        order = players[start:] + players[:start]
        print(f"Порядок: {order}")
        return [p for p in order if p.in_hand]


    def _deal_flop(self):
        self.deck.burn()
        self.table = [self.deck.dealcard(), self.deck.dealcard(), self.deck.dealcard()]
        self._update_best_hands()

    def _deal_turn(self):
        self.deck.burn()
        self.table.append(self.deck.dealcard())
        self._update_best_hands()

    def _deal_river(self):
        self.deck.burn()
        self.table.append(self.deck.dealcard())
        self._update_best_hands()

    def _update_best_hands(self):
        for p in self.game.players:
            if p.in_hand:
                self.pm[p].update_best_hand(self.table)

    def _determine_winner(self):
        """Находит победителя одной раздачи."""

        active_players = self.game.betting_players

        if len(active_players) == 1:
            return active_players

        # сравнение best_hand — предполагаем, что best_hand = список значений
        active_players.sort(key=lambda p: p.best_hand, reverse=True)

        best = active_players[0].best_hand
        winners = [p for p in active_players if p.best_hand == best]



        return winners
