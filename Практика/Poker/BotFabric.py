from .bot import SimpleGeneticBot
from .GameManager import GameManager
from .Game import Game

import random
import copy

ref_bots = [
        SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
        SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight"),
        SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff"),
        SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced"),
        SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac"),
    ]

class BotFabric:

    result_bot = SimpleGeneticBot([0, 0, 0], "result_bot")
    bot_list = list()
    mutation_rate = 0

    def __init__(self, mutation_rate=0.1, mutation_strength=0.1):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def get_result_bot(self):
        return self.result_bot

    def crossover(self, g1, g2):
        result_genome = [(a + b) / 2 for a, b in zip(g1, g2)]
        return self.normalize_genome(result_genome)

    def normalize_genome(self, genome : list[float]):
        total = sum(genome)
        return [x / total for x in genome]

    def mutate(self, genome, mutation_rate, mutation_strength):
    
        return [
            max(0, min(1, g + (random.uniform(-mutation_strength, mutation_strength) if random.random() < mutation_rate else 0)))
            for g in genome
        ]
    
    def set_bot_list(self, bots : list):
        self.bot_list = bots.copy()


    def get_new_generation(self, winners : list[SimpleGeneticBot], loosers : list[SimpleGeneticBot]) -> list:
        """
        Количество игроков - 10.
        Количество зарезервированных ботов - 2.
        Правило: Первые два - в следующее поколение, первый ~ (второй, третий, четвёртый), второй ~ (третий, четвёртый), третий ~ четвёртый,
        """
        new_generation = list()

        for player in winners:
            if player.get_stack() > 0:
                winners.remove(player)

        ranked_list = winners + list(reversed(loosers))
        new_generation.extend([
            SimpleGeneticBot(self.mutate(ranked_list[0].get_genome(), self.mutation_rate, self.mutation_strength), "First"),
            SimpleGeneticBot(self.mutate(ranked_list[1].get_genome(), self.mutation_rate, self.mutation_strength), "Second"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[0].get_genome(), ranked_list[1].get_genome()), self.mutation_rate, self.mutation_strength), "First~Second"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[0].get_genome(), ranked_list[2].get_genome()), self.mutation_rate, self.mutation_strength), "First~Third"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[0].get_genome(), ranked_list[3].get_genome()), self.mutation_rate, self.mutation_strength), "First~Fourth"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[1].get_genome(), ranked_list[2].get_genome()), self.mutation_rate, self.mutation_strength), "Second~Third"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[1].get_genome(), ranked_list[3].get_genome()), self.mutation_rate, self.mutation_strength), "Second~Fourth"),
            SimpleGeneticBot(self.mutate(self.crossover(ranked_list[2].get_genome(), ranked_list[3].get_genome()), self.mutation_rate, self.mutation_strength), "Third~Fourth"),
            copy.deepcopy(ref_bots[random.randint(0, len(ref_bots) - 1)]),
            copy.deepcopy(ref_bots[random.randint(0, len(ref_bots) - 1)])
        ])
        return new_generation

    def get_random_generation(self):
        return [copy.deepcopy(ref_bots[random.randint(0, len(ref_bots) - 1)]) for _ in range(10)]

    def fit(self, num_games : int, num_rounds : int):

        self.set_bot_list(self.get_random_generation())

        for i in range(num_games):

            game = Game()        
            game.players = self.bot_list

            game.set_registered_players(self.bot_list)
            
            game_manager = GameManager(game)
            winners  = game_manager.start_game(num_rounds)

            new_generation_bots = self.get_new_generation(winners, game.get_loosers_list())

            self.set_bot_list(new_generation_bots)

            self.result_bot = new_generation_bots[0]

            with open("result_bot_genetic.txt", "a" ) as f:
                f.write(f"Generation {i}:\n")
                f.write(str(self.result_bot) + "\n")




