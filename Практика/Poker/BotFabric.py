from .bot import SimpleGeneticBot
from .GameManager import GameManager
from .Game import Game

import random


ref_bots = [
    SimpleGeneticBot([0.8, 0.5, 0.2], "Ref_Bot_1"),
    SimpleGeneticBot([0.2, 0.2, 0.2], "Ref_Bot_2"),
    SimpleGeneticBot([0.7, 0.5, 0.2], "Ref_Bot_3"),
    SimpleGeneticBot([0.4, 0.3, 0.2], "Ref_Bot_4")
]

class BotFabric:

    result_bot = SimpleGeneticBot([0, 0, 0], "result_bot")
    bot_list = list()
    def crossover(self, g1, g2):
   
        return [(a + b) / 2 for a, b in zip(g1, g2)]

    def mutate(self, genome, mutation_rate, mutation_strength):
    
        return [
            max(0, min(1, g + (random.uniform(-mutation_strength, mutation_strength) if random.random() < mutation_rate else 0)))
            for g in genome
        ]
    
    def set_bot_list(self, bots : list):
        bot_list = bots.copy()

    def get_new_generation(self, winners : list) -> list:
        new_generation = list()

        return new_generation

    def fit(self, mutation_rate : float, num_games : int, num_rounds : int, num_ref_bots : int):
        
        for _ in range(num_games):
            game = Game()        
            game.players = self.bot_list

            for _ in range(num_ref_bots):
                game.add_player(ref_bots[random.randint(len(ref_bots))])
            
            game_manager = GameManager(game)
            winners  = game_manager.start_round()

            new_generation_bots = self.get_new_generation(winners)
            







