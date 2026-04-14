from Практика.Poker import *
import numpy as np
import matplotlib.pyplot as plt

"""
ТЕСТ НА "ТРЕЗВОСТЬ" — проверка модели без энтропии и exploration

training_mode=False → argmax (строго выбирает лучшее действие)
Цель: если модель НАУЧИЛАСЬ фолдить, она начнёт это делать без шума энтропии.
"""

# Настраиваем логи
StaticLogger.configure_experiment_logs("Testing_without_entropy_and_training")

num_rounds = 30
num_games = 1000

players = [
    SimpleGeneticBot([0.8, 0.1, 0.1], name="Aggressor"),
    NeuralACAgent()
]

pm = NeuralACAgentManager(players[1])
pm.load_ac_agent(filename="Big_Experiment_better_NN.pth")

# Счётчик действий
action_counts = {0: 0, 1: 0, 2: 0}  # fold, raise, call
num_wins = {p: 0 for p in players}
win_rate_history = []
game_winners = []

print(f"\n{'='*60}")
print(f"🧪 ТЕСТ НА ТРЕЗВОСТЬ: {num_games} игр, training_mode=False")
print(f"{'='*60}\n")

# Начальная валидация рук
pm.validate_hand_values()

for i in range(num_games):
    game = Game()

    for player in players:
        game.add_player(player)

    gameManager = GameManager(game)

    # НАЙТИ и патчить NeuralACAgentManager ВНУТРИ GameManager
    # (gameManager создаёт свой собственный pm, не использует наш локальный)
    for p, mgr in gameManager.pm.items():
        if isinstance(mgr, NeuralACAgentManager):
            original_ask = mgr.ask_decision

            def ask_decision_no_training(s_actor, s_critic, can_check=False, can_raise=True):
                """Ask decision СТРОГО в режиме argmax (без exploration)"""
                action_idx = mgr.act(s_actor, s_critic, can_check, can_raise, training_mode=False)

                # Считаем действия сразу при выборе
                action_counts[action_idx] += 1

                action = ACTIONS[action_idx]
                mgr.player.set_decision(action)

                StaticLogger.print_to("game", f"\nИгрок {mgr.player.name}")
                StaticLogger.print_to("game", f"Ваши карты: {mgr.player.hole_cards}")
                StaticLogger.print_to("game", f"Текущая ставка: {mgr.player.bet}, стек: {mgr.player.stack}")
                StaticLogger.print_to("game", f"Ваша лучшая комбинация: {mgr.player.best_hand}")
                StaticLogger.print_to("game", f"Ваш выбор: {mgr.player.decision}")

                return mgr.player.decision

            mgr.ask_decision = ask_decision_no_training
            break

    winners = gameManager.start_game(num_rounds, i)

    StaticLogger.print_to("game", f'\033[32mМеста в порядке убывания: {winners}\033[0m\n')

    game_winners.append(winners)
    best_stack = winners[0].get_stack()
    for winner in winners:
        if winner.stack == best_stack:
            num_wins[winner] += 1

    win_rate = num_wins[players[1]] / (i + 1) * 100
    win_rate_history.append(win_rate)

    # Прогресс каждые 100 игр
    if (i + 1) % 100 == 0:
        total_actions = sum(action_counts.values())
        fold_pct = action_counts[0] / total_actions * 100
        call_pct = action_counts[2] / total_actions * 100
        raise_pct = action_counts[1] / total_actions * 100

        print(f"\n📊 Игра {i+1}/{num_games} | Win Rate: {win_rate:.1f}% | "
              f"Actions: Fold={fold_pct:.1f}%, Call={call_pct:.1f}%, Raise={raise_pct:.1f}%")

# ========== ФИНАЛЬНАЯ СТАТИСТИКА ==========
total_actions = sum(action_counts.values())
fold_pct = action_counts[0] / total_actions * 100
call_pct = action_counts[2] / total_actions * 100
raise_pct = action_counts[1] / total_actions * 100

print(f"\n{'='*60}")
print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА ДЕЙСТВИЙ (argmax mode):")
print(f"{'='*60}")
print(f"  Fold:  {action_counts[0]:6d} ({fold_pct:.1f}%)")
print(f"  Call:  {action_counts[2]:6d} ({call_pct:.1f}%)")
print(f"  Raise: {action_counts[1]:6d} ({raise_pct:.1f}%)")
print(f"  Total: {total_actions}")
print(f"{'='*60}")

print(f"\n🏆 Win Rate: {win_rate_history[-1]:.1f}%")
print(f"   Победы агента: {num_wins[players[1]]}")

StaticLogger.print_to("summary", f"\n{'='*60}")
StaticLogger.print_to("summary", f"📊 FINAL ACTION DISTRIBUTION (argmax mode):")
StaticLogger.print_to("summary", f"  Fold:  {action_counts[0]:6d} ({fold_pct:.1f}%)")
StaticLogger.print_to("summary", f"  Call:  {action_counts[2]:6d} ({call_pct:.1f}%)")
StaticLogger.print_to("summary", f"  Raise: {action_counts[1]:6d} ({raise_pct:.1f}%)")
StaticLogger.print_to("summary", f"  Total: {total_actions}")
StaticLogger.print_to("summary", f"  Win Rate: {win_rate_history[-1]:.1f}%")
StaticLogger.print_to("summary", f"{'='*60}")

StaticLogger.flush_all()

# ========== ГРАФИК ==========
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_games + 1), win_rate_history, 'b-', linewidth=2)
plt.axhline(y=win_rate_history[-1], color='r', linestyle='--', alpha=0.7)
plt.xlabel('Номер игры')
plt.ylabel('Win Rate, %')
plt.title(f'Win Rate при training_mode=False ({num_games} игр)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.show()

# ========== ИНТЕРПРЕТАЦИЯ ==========
print(f"\n🔍 ИНТЕРПРЕТАЦИЯ:")
if fold_pct > 20:
    print(f"  ✅ Модель ФОЛДИТ {fold_pct:.1f}% — она НАУЧИЛАСЬ фолдить!")
    print(f"     Проблема была в ЭНТРОПИИ — она мешала модели принимать решения.")
elif fold_pct > 10:
    print(f"  ⚡ Модель ФОЛДИТ {fold_pct:.1f}% — начинает понимать, когда сбрасывать.")
    print(f"     Энтропия частично мешала, но веса ещё не идеальны.")
else:
    print(f"  ❌ Модель ФОЛДИТ только {fold_pct:.1f}% — проблема в ВЕСАХ/НАГРАДАХ!")
    print(f"     Даже без энтропии модель не фолдит — значит Advantage не даёт сигнал.")
    print(f"     Нужно: проверить Advantage std, returns scale, требует более глубокой отладки.")
