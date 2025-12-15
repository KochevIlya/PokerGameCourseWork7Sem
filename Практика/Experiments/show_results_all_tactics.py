from Практика.Poker import *
from matplotlib import pyplot as plt

aggressor = SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor")
tight = SimpleGeneticBot([0.15, 0.05, 0.8], name="Tight")
bluff = SimpleGeneticBot([0.2, 0.6, 0.2], name="Bluff")
balanced = SimpleGeneticBot([0.33, 0.33, 0.33], name="Balanced")
maniac = SimpleGeneticBot([0.45, 0.45, 0.1], name="Maniac")

results = {
    aggressor : 398,
    tight : 8,
    bluff : 240,
    balanced : 148,
    maniac : 212,
}

bots = [bot.name for bot in results.keys()]
wins = list(results.values())

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(bots, wins, color=colors, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Типы ботов', fontsize=14, fontweight='bold')
ax.set_ylabel('Количество выигрышей', fontsize=14, fontweight='bold')
ax.set_title('Результаты выигрышей ботов', fontsize=16, fontweight='bold', pad=20)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.set_ylim(0, max(wins) * 1.15)

plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

plt.tight_layout()

plt.show()