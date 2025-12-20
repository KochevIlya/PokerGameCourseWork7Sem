from Практика.Poker import Game
from matplotlib import pyplot as plt

class NNData:

    _instance = None
    episode_buffer = []
    BATCH_SIZE = 4096
    loss_critic_buffer = []
    loss_actor_buffer = []
    loss_buffer = []


    @staticmethod
    def add_buffer(args):
        NNData.episode_buffer.append(args)

    @staticmethod
    def is_full():
        return len(NNData.episode_buffer) >= NNData.BATCH_SIZE
    @staticmethod
    def get_buffer():
        return NNData.episode_buffer

    @staticmethod
    def clear():
        NNData.episode_buffer.clear()

    @staticmethod
    def add_loss_critic(args):
        NNData.loss_critic_buffer.append(args)
    @staticmethod
    def add_loss_actor(args):
        NNData.loss_actor_buffer.append(args)

    @staticmethod
    def add_loss_buffer(args):
        NNData.loss_buffer.append(args)

    @staticmethod
    def show_losses():
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_buffer) + 1), NNData.loss_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Total Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.2, 0.2)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_actor_buffer) + 1), NNData.loss_actor_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Actor Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.2, 0.2)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(NNData.loss_critic_buffer) + 1), NNData.loss_critic_buffer, 'b-', linewidth=2)
        plt.xlabel('Номер игры')
        plt.ylabel('Loss')
        plt.title(f'Critic Loss')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 0.6)
        plt.show()