from . import compare_hands
from .NeuralAgent import *
from .Deck import Deck
from .PlayerManager import PlayerManager
from .Game import Game
from .Player import Player
from .BotManager import BotManager
from .bot import SimpleGeneticBot
from .NeuralAgentManager import *
from .NeuralAgentManager import *
from functools import cmp_to_key
from .Logger import *

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
            elif isinstance(player, NeuralAgent):
                self.pm[player] = NeuralAgentManager(player)
            else:
                self.pm[player] = PlayerManager(player)
        self.num_players = len(self.game.registered_players)
        self.max_chips = self.num_players * game.initial_stack
        self.table = []
        self.pot = 0.0
        self.current_decision_value = 0.0
        self.current_num_bets = 0
        self.games_count = 0
        self.games_not_trained = 30

    def start_game(self, num_rounds):
        for i in range(num_rounds):
            StaticLogger.print(f"Round {i}\n")
            StaticLogger.print(f"{str(self.game)}\n")
            self._prepare_round()


            if len(self.game.players) <= 1:
                # print(f"Game is over, because of the players amount")
                self.game.registered_players.sort(key=lambda p: p.stack, reverse=True)
                return self.game.registered_players
            else:
                self.game.next_blinds()
                StaticLogger.print(f"After preparing: {str(self.game)}\n")
                StaticLogger.print(f'\033[32mВыигрывает(ют): {self.start_round()}\033[0m\n')
        StaticLogger.print(f"Game is over, because of the rounds amount")
        self.game.registered_players.sort(key=lambda p: p.stack, reverse=True)
        return self.game.registered_players


    def start_round(self):
        """
        Основной игровой цикл одной раздачи.
        """
        self.games_count += 1
        StaticLogger.print(f"Gmae number: {self.games_count}\n")
        self.current_decision_value = 0.0
        self.current_num_bets = 0

        self._post_blinds()
        
        self._betting_round(stage="preflop")
        
        self._deal_flop()
        self.show_current_situation()


        self._betting_round(stage="flop")

        self._deal_turn()
        self.show_current_situation()
        self._betting_round(stage="turn")


        self._deal_river()
        self.show_current_situation()
        self._betting_round(stage="river")


        winners = self._determine_winner()

        self.winners_distribution(winners)


        return winners

    def winners_distribution(self, winners: list[Player]):
        """
        Распределяет выигрыш и раздает награды (Rewards) для обучения.
        Здесь мы наказываем за трусость и поощряем за умные фолды.
        """
        share = self.pot / len(winners)


        # 1. Раздаем фишки победителям
        for winner in winners:
            winner.add_stack(share)

        # Берем сильнейшую руку среди победителей для сравнения
        # (предполагаем, что winners уже отсортированы или у них равные руки)
        winner_hand_strength = winners[0].best_hand
        # Если best_hand это объект, нужно привести к числу.
        # Если у тебя best_hand это кортеж/список, логику сравнения нужно адаптировать.
        # Для простоты допустим, что мы можем сравнить силу рук заново.

        # 2. Проходим по ВСЕМ игрокам (включая тех, кто сфолдил)
        for player in self.game.players:
            pm = self.pm[player]

            # Нас интересуют только наши обучаемые агенты
            if isinstance(pm, NeuralAgentManager):



                folded_player = pm.player
                folder_strength = self.pm[folded_player].update_best_hand(self.table)

                winner_player = winners[0]
                winner_strength = winner_player.best_hand

                final_reward = -0.5 * folded_player.get_bet() / self.game.initial_stack / self.num_players



                # Сценарий А: Игрок дошел до конца (Active)
                if player.in_hand:
                    if player in winners:
                        # Победа: Большая награда
                        final_reward = 1 * self.pot / self.game.initial_stack / self.num_players

                        # Сценарий Б: Игрок сфолдил (Folded)

                else:
                    if  compare_hands(folder_strength, winner_strength) != 2:
                        # BAD FOLD: У нас карты были лучше, чем у того, кто забрал банк!
                        # Наказываем сильно.
                        StaticLogger.print(f"Bot {folded_player.name} FOLDED winning hand! Punishing.")
                        final_reward = -1 * folded_player.get_bet() / self.game.initial_stack / self.num_players



                s, a, _, s_next, _ = pm.episode_memory[-1]
                pm.episode_memory[-1] = (s, a, final_reward, s_next, True)
                self.pm[player].remember_episode(final_reward)
                # for transition in pm.episode_memory:
                #     pm.train_step(*transition)

            for player in self.game.players:
                if isinstance(self.pm[player], NeuralAgentManager):
                    self.pm[player].train_experience_replay()

            if self.games_count % self.games_not_trained == 0:
                for player in self.game.players:
                    if isinstance(self.pm[player], NeuralAgentManager):
                        self.pm[player].update_target_network()
                        StaticLogger.print("Target Network updated!")


    def _analyze_fold_decision(self, folded_player: Player, pm, winners):
        """
        Анализирует, правильным ли был фолд.
        Сравнивает карты сбросившего с картами победителя на текущем столе.
        """
        # Нам нужно оценить силу руки, которая была сброшена, учитывая ВЕСЬ стол (даже если фолд был на префлопе)
        # Вариант 1: Сравниваем "честно" (как если бы игрок дошел до конца с текущим столом)
        # HandCalculator должен уметь работать с текущим self.table (даже если там 0, 3 или 5 карт)

        # Считаем силу руки сбросившего
        folder_strength = self.pm[folded_player].update_best_hand(self.table)

        # Считаем силу руки победителя (берем первого попавшегося, так как они выиграли)
        winner_player = winners[0]
        winner_strength = winner_player.best_hand

        # Логика вознаграждения
        if compare_hands(folder_strength, winner_strength) !=2:
            # BAD FOLD: У нас карты были лучше, чем у того, кто забрал банк!
            # Наказываем сильно.
            StaticLogger.print(f"Bot {folded_player.name} FOLDED winning hand! Punishing.")
            pm.train_step(None, reward=-1.0)
        else:
            # GOOD FOLD: У победителя карты реально лучше.
            # Поощряем немного (за экономию стека).
            # Не делай награду слишком большой, иначе он будет только фолдить.
            # 0.2 - это "утешительный приз".
            StaticLogger.print(f"Bot {folded_player.name} made a GOOD FOLD.")
            pm.train_step(None, reward=0.2)

        StaticLogger.print(f"Folder_hand: {folder_strength}")
        StaticLogger.print(f"Winner_hand: {winner_strength}")




    def show_current_situation(self):
        StaticLogger.print(f"\nКарты на столе: {self.table}")
        

    def _prepare_round(self):
        """Сбрасывает всё состояние перед новой раздачей."""

        self.table = []
        self.pot = 0.0
        for p, pm in self.pm.items():
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

        


    def _betting_round(self, stage="preflop"):
        """
        Один круг ставок.
        start_from:
            "UTG" — сразу после BB (префлоп)
            "SB" — с малого блайнда (флоп/терн/ривер)
        """

        active_players = self.game.betting_players
        if len(active_players) <= 1:
            return

        order = self._betting_order(stage)
        players_to_act = set(order)

        for player in active_players:
            self.pm[player].num_bets = 0
            self.pm[player].decision_value = 0

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
                pm.update_best_hand(self.table)
                # self.show_current_situation()
                if  isinstance(pm, BotManager):
                    pm.ask_decision(
                        current_bet=self.current_bet,
                        min_raise=self.game.min_bet,
                        community_cards=self.table.copy()
                    )

                elif isinstance(pm, NeuralAgentManager):

                    next_state = pm.build_state_vector(
                        current_bet_normalized=self.current_bet / self.game.initial_stack / self.num_players,
                        current_stack_normalized=player.get_stack() / self.game.initial_stack / self.num_players,
                        pot_normalize=self.pot / self.game.initial_stack / self.num_players,
                        community_cards=self.table.copy(),
                        opponents_decision_value=(self.current_decision_value * self.current_num_bets - pm.decision_value * pm.num_bets) / (self.current_num_bets - pm.num_bets) if (self.current_num_bets - pm.num_bets) != 0 else 0 ,
                        stage=stage
                    )
                    pm.ask_decision(next_state)

                    pm.episode_memory.append(
                        (pm.last_state, pm.last_action, 0.0, next_state, False)
                    )

                elif isinstance(pm, PlayerManager):
                    pm.ask_decision(
                        current_bet=self.current_bet,
                        min_raise=self.game.min_bet
                    )




                if player.decision == "fold":
                    self.current_num_bets += 1
                    self.current_decision_value = self.current_decision_value / self.current_num_bets
                    pm.num_bets += 1
                    pm.decision_value = pm.decision_value / pm.num_bets

                    pm.fold()
                    players_to_act.remove(player)
                    self.game.remove_player_betting_round(player)
                    StaticLogger.print(pm.player)
                    continue

                if player.decision == "raise":
                    self.current_num_bets += 1
                    self.current_decision_value = self.current_decision_value / self.current_num_bets + 1 / self.current_num_bets
                    pm.num_bets += 1
                    pm.decision_value = pm.decision_value / pm.num_bets + 1 / pm.num_bets

                    needed = self.current_bet + self.game.min_bet

                    if not pm.can_apply(needed):
                        player.set_decision("call")

                    else:
                        to_raise = pm.raise_bet(needed)
                        self.current_bet += self.game.min_bet
                        self.pot += to_raise

                        players_to_act = set(order)
                        players_to_act.remove(player)
                        StaticLogger.print(pm.player)
                        continue

                if player.decision == "call":
                    self.current_num_bets += 1
                    self.current_decision_value += self.current_decision_value / self.current_num_bets + 0.5 / self.current_num_bets
                    pm.num_bets += 1
                    pm.decision_value = pm.decision_value / pm.num_bets + 0.5 / pm.num_bets

                    needed = self.current_bet
                    if pm.can_apply(needed):
                        to_call = pm.call(needed)
                        self.pot += to_call
                        players_to_act.remove(player)
                        StaticLogger.print(pm.player)

                    else:
                        to_amount = pm.all_in()
                        self.pot += to_amount
                        players_to_act.remove(player)
                        StaticLogger.print(pm.player)


            still_in = self.game.active_players_count()
            if still_in <= 1:
                break


    def _betting_order(self, stage):
        """Создаёт порядок игроков для ставок."""

        players = self.game.betting_players
        sb_index = self.game.blind_index
        bb_index = (sb_index + 1) % len(players)

        if stage == "preflop":
            start = (bb_index + 1) % len(players)
        else:
            start = sb_index

        order = players[start:] + players[:start]
        StaticLogger.print(f"Порядок: {order}")
        return [p for p in order if p.in_hand]

    def get_another_player(self, pm:PlayerManager):
        for key, value in self.pm.items():
            if pm == key:
                continue
            else:
                return value


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
        active_players.sort(key=cmp_to_key(compare_players_adapted), reverse=True)

        best = active_players[0].best_hand
        winners = [p for p in active_players if p.best_hand == best]



        return winners


def compare_players(player1, player2):
    return compare_hands(player1.best_hand, player2.best_hand)


def compare_players_adapted(player1, player2):
    """
    Адаптер для compare_hands, который преобразует 0,1,2 в -1,0,1
    """

    result = compare_hands(player1.best_hand, player2.best_hand)

    if result == 0:  # ничья
        return 0
    elif result == 1:  # player1.best_hand лучше
        return 1  # player1 должен быть первым (при reverse=True)
    else:  # result == 2, player2.best_hand лучше
        return -1  # player2 должен быть первым (при reverse=True)
