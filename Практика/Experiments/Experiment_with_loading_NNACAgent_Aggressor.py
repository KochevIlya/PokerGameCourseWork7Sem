from Практика.Poker import *


StaticLogger.configure("Experiment_with_smaller_batch_size_Aggressor.log", 1000)

learning_num_games = 50
learning_num_rounds = 50


num_rounds = 30
num_games = 30000

game_winners = []

players = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        NeuralACAgent()
    ]
pm = NeuralACAgentManager(players[1])
pm.load_ac_agent(filename="neural_ac_agent_small_batch.pth")

num_wins = { p:0 for p in players}

for i in range(num_games):

    game = Game()

    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    winners = gameManager.start_game(num_rounds, i)

    StaticLogger.print(f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')

    game_winners.append(winners)
    best_stack = winners[0].get_stack()
    for winner in winners:
        if (winner.stack == best_stack):
            num_wins[winner] += 1

StaticLogger.print(f'\033[32mМеста в порядке убывания: {game_winners}\033[0m\n')
StaticLogger.print(f'\033[32mКоличество выигрышей: {num_wins}\033[0m\n')
StaticLogger.flush()