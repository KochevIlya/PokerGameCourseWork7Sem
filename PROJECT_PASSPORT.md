# 📘 Паспорт проекта: Texas Hold'em Poker AI

> **Курсовая работа** | 8 семестр | 2026
> **Статус:** ✅ Активная разработка (100,000+ игр обучено)
> **Технология:** Actor-Critic с Dual-LSTM, Skip Connections, Policy Gradient
> **Фреймворк:** PyTorch + eval7 (Monte Carlo Hand Evaluator)

---

## 1. Общая информация

| Параметр | Значение |
|----------|----------|
| **Название** | Texas Hold'em Poker AI — нейросетевой агент |
| **Тип проекта** | Курсовая работа (обучение с подкреплением) |
| **Язык** | Python 3.10+ |
| **Фреймворк ML** | PyTorch (Actor-Critic, Policy Gradient) |
| **Цель** | Обучение нейросетевого агента игре в покер Texas Hold'em против ботов с разными стратегиями |
| **Метод обучения** | On-policy Policy Gradient (Actor-Critic) с Advantage и Entropy Decay |
| **Архитектура сети** | Dual-LSTM Actor-Critic с Skip Connections и LayerNorm |
| **Размер кодовой базы** | 68 Python-файлов, 47 моделей, 25+ экспериментов |
| **Статус обучения** | 100,000+ игр проведено, чекпоинты сохранены |

---

## 2. Архитектура системы

### 2.1. Основные компоненты

```
┌─────────────────────────────────────────────────────────┐
│                    GameManager                           │
│  (Управление игровой логикой, ставками, раундами)        │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Player 1    │  │  Player 2    │  │  Player N    │   │
│  │  + Manager   │  │  + Manager   │  │  + Manager   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Game                                  │
│  (Состояние стола, блайнды, игроки, колода)              │
└─────────────────────────────────────────────────────────┘
```

### 2.2. Классы и их роли

| Класс | Файл | Назначение |
|-------|------|------------|
| `Game` | `Game.py` | Состояние стола: игроки, блайнды, стек, ротация |
| `GameManager` | `GameManager.py` | Игровой цикл: preflop → flop → turn → river, ставки, определение победителя |
| `Player` | `Player.py` | Базовый класс игрока: карты, стек, решение |
| `PlayerManager` | `PlayerManager.py` | Базовый менеджер: ставка, call, fold, raise |
| `HandCalculator` | `HandCalculator.py` | Monte Carlo оценка силы руки (eval7, 200 итераций) |
| `StaticLogger` | `Logger.py` | Многокатегориальное логирование (game, training, loss, entropy, validation, summary) |
| `NNData` | `NNData.py` | Буфер эпизодов для батчевого обучения (BATCH_SIZE=4096 шагов) |

---

## 3. Игровой процесс

### 3.1. Правила

- **Формат:** Texas Hold'em No-Limit (упрощённый)
- **Количество игроков:** 2–8
- **Стартовый стек:** 100 фишек
- **Минимальная ставка (блайнд):** 10 фишек
- **Small Blind:** `min_bet // 2 = 5`
- **Big Blind:** `min_bet = 10`

### 3.2. Стадии раздачи

| Стадия | Карты | Описание |
|--------|-------|----------|
| **Preflop** | 2 hole cards | Первые ставки после раздачи |
| **Flop** | 3 community cards | Открываются 3 общие карты |
| **Turn** | +1 community card | 4-я общая карта |
| **River** | +1 community card | 5-я общая карта, финальные ставки |

### 3.3. Доступные действия

| Действие | Код | Описание |
|----------|-----|----------|
| **Fold** | 0 | Сбросить карты (выйти из раздачи) |
| **Call** | 2 | Уравнять текущую ставку |
| **Raise** | 1 | Повысить ставку на `min_bet` |

### 3.4. Порядок ставок

- **Preflop:** начинается с игрока после Big Blind (UTG)
- **Flop/Turn/River:** начинается с Small Blind
- Циклический порядок по часовой стрелке

---

## 4. Типы агентов

### 4.1. SimpleGeneticBot (боты-оппоненты)

| Тип | Genome `[aggression, bluff, patience]` | Стратегия |
|-----|----------------------------------------|-----------|
| **Aggressor** | `[0.8, 0.1, 0.1]` | Агрессивная: часто повышает |
| **Tight** | `[0.15, 0.05, 0.8]` | Осторожная: играет только сильные руки |
| **Bluff** | `[0.2, 0.6, 0.2]` | Блефующая: случайные повышения |
| **Balanced** | `[0.33, 0.33, 0.33]` | Сбалансированная |
| **Maniac** | `[0.45, 0.45, 0.1]` | Безумная: много блефа и агрессии |

**Формула решения:**
```python
score = genome[0] * hand_strength + genome[1] * bluff_rand + (1 - genome[2] * bet / (bet + stack))
if score > 0.6: raise
elif score > 0.3: call
else: fold
```

### 4.2. NeuralACAgent (нейросетевой агент)

- **Архитектура:** Actor-Critic с Dual-LSTM
- **Обучение:** On-policy Policy Gradient с Advantage
- **Режимы:** Training (сэмплирование) / Validation (argmax)
- **Файлы:** `ActorCritic.py`, `NeuralACAgentManager.py`

### 4.3. Другие агенты

| Агент | Файл | Описание |
|-------|------|----------|
| `RandomPlayer` | `RandomPlayer.py` | Случайные действия |
| `CallingPlayer` | `CallingPlayer.py` | Всегда делает call |

---

## 5. Архитектура нейросети

### 5.1. ActorCriticNet

```
┌──────────────────────────────────────────────────────┐
│                   INPUT                               │
│  s_actor: [10]  |  s_critic: [11]  |  history: [10×3]│
└────────┬─────────────────┬──────────────────┬─────────┘
         │                 │                  │
    ┌────▼────┐      ┌────▼────┐             │
    │  LSTM   │      │  LSTM   │             │
    │ (Actor) │      │(Critic) │             │
    └────┬────┘      └────┬────┘             │
         │ [16]           │ [16]             │
         └────┐      ┌────┘                  │
              ▼      ▼                       ▼
         ┌──────────────┐            ┌──────────────┐
         │  Cat + LSTM  │            │  Cat + LSTM  │
         │  context [16]│            │  context [16]│
         └──────┬───────┘            └──────┬───────┘
                │                           │
         ┌──────▼───────┐            ┌──────▼──────────┐
         │  s_actor[10] │            │  s_critic[11]   │
         │  + ctx[16]   │            │  + ctx[16]      │
         │  = [26]      │            │  = [27]         │
         └──────┬───────┘            └──────┬──────────┘
                │                           │
         ┌──────▼───────┐            ┌──────▼──────────┐
         │ ACTOR NET    │            │ CRITIC NET      │
         │ Linear(26)   │            │ Linear(27→128)  │
         │ → ReLU       │            │ → ReLU          │
         │ Linear(128→64│            │ → LayerNorm(128)│
         │ → ReLU       │            │                 │
         │ Linear(64→3) │            │  Skip Connect:  │
         │              │            │  key_feat[0,10] │
         │ Output:      │            │  Cat(128+2=130) │
         │ action_logits│            │ → ReLU          │
         │ [3]          │            │ Linear(130→64)  │
         │              │            │ → ReLU          │
         │              │            │ Linear(64→1)    │
         │              │            │ Output: value[1]│
         └──────────────┘            └─────────────────┘
```

### 5.2. Параметры слоёв

| Компонент | Слои | Размеры | Learning Rate |
|-----------|------|---------|---------------|
| **Actor LSTM** | `nn.LSTM` | input=3, hidden=16 | 2e-4 |
| **Critic LSTM** | `nn.LSTM` | input=3, hidden=16 | 1e-4 |
| **Actor Net** | Linear → ReLU → Linear → ReLU → Linear | 26→128→64→3 | 2e-4 |
| **Critic Net** | Linear → ReLU → LayerNorm → Cat(2) → Linear → ReLU → Linear | 27→128→130→64→1 | 1e-4 |

### 5.3. Skip Connections в Критике

Ключевые признаки (индексы в `s_critic`):
- `[0]` — `hand_strength` (сила руки героя, 0.0–1.0)
- `[10]` — `avg_opp_strength` (средняя сила рук оппонентов, 0.0–1.0)

Эти признаки передаются напрямую во второй слой Критика через конкатенацию, чтобы не «размываться» через ReLU.

**LayerNorm:** нормализует выход первого слоя (128 нейронов) перед конкатенацией, выравнивая масштаб с ключевыми признаками (0.0–1.0).

### 5.4. Вектор состояния (s_actor / s_critic)

| Индекс | Признак | Диапазон | Описание |
|--------|---------|----------|----------|
| 0 | `hand_strength` | 0.0–1.0 | Сила руки героя (Monte Carlo) |
| 1 | `current_bet_normalized` | 0.0–1.0 | Текущая ставка / max_chips |
| 2 | `current_stack_normalized` | 0.0–1.0 | Стек игрока / max_chips |
| 3 | `pot_normalized` | 0.0–1.0 | Банк / max_chips |
| 4 | `s_preflop` | 0/1 | Флаг стадии |
| 5 | `s_flop` | 0/1 | Флаг стадии |
| 6 | `s_turn` | 0/1 | Флаг стадии |
| 7 | `s_river` | 0/1 | Флаг стадии |
| 8 | `current_decision_value` | 0.0–1.0 | Средняя «агрессивность» решений |
| 9 | `avg_decision_value` | 0.0–1.0 | Личная агресс игрока |
| 10 | `avg_opp_strength` | 0.0–1.0 | Средняя сила рук всех оппонентов |

---

## 6. Обучение

### 6.1. Алгоритм

- **Метод:** On-policy Policy Gradient (Actor-Critic)
- **Advantage:** `A(s,a) = R - V(s)` (returns минус значение Критика)
- **Loss Актора:** `L_actor = -mean(log_prob(a) * advantage)`
- **Loss Критика:** `L_critic = HuberLoss(V(s), R, delta=1.0)`
- **Entropy Bonus:** `L_entropy = -entropy_coef * H(π)` (для exploration)
- **Gradient Clipping:** `max_norm=0.5`

### 6.2. Гиперпараметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `BATCH_SIZE` | 4096 шагов | Шагов в буфере перед обновлением |
| `gamma` | 0.99 | Дисконтирование наград |
| `actor_lr` | 2e-4 | Learning rate Актора (ускоренное обучение) |
| `critic_lr` | 1e-4 | Learning rate Критика (стабильная оценка) |
| `entropy_coef` | 0.0001 → 0.00001 | Коэффициент энтропии (decay 0.995) |
| `entropy_update_interval` | 200 игр | Частота уменьшения энтропии |
| `epoch_interval` | 200 игр | Частота вывода статистики |
| `history_len` | 10 | Длина истории действий оппонентов |
| `lstm_hidden` | 16 | Размер скрытого состояния LSTM |
| `hand_eval_iters` | 200 | Итераций Monte Carlo для оценки руки |
| `gradient_clip` | 0.5 | Максимальная норма градиента |
| `huber_delta` | 1.0 | Параметр Huber Loss для Критика |

### 6.3. Награды (Rewards)

```python
if player in winners:
    net_profit = pot - player.bet
else:
    net_profit = -player.bet
    # Reward Shaping: штраф за проигрыш на шоудауне
    if not player.is_folded:
        net_profit *= 1.5  # Showdown penalty

final_reward = net_profit / min_bet  # Нормализация через Big Blind
```

### 6.4. Action Masking

Перед сэмплированием/выбором действия применяется маска легальных действий:

```python
legal_mask = torch.tensor([True, can_raise, True], device=self.device)  # [fold, raise, call]
action_logits = action_logits.masked_fill(~legal_mask, -1e9)
```

Маска сохраняется в `episode_data` и применяется повторно при обучении в `_update_network()`.

### 6.5. Backpropagation Through Time (BPTT)

Внутри `_update_network()` раздача прогоняется **пошагово** в цикле:

```python
for i in range(len(s_actors_t)):
    logits, val, curr_actor_h, curr_critic_h = self.player.ac_net(
        s_actors_t[i].unsqueeze(0),
        s_critics_t[i].unsqueeze(0),
        histories_t[i].unsqueeze(0),
        actor_hidden=curr_actor_h,
        critic_hidden=curr_critic_h
    )
```

LSTM скрытое состояние передаётся от первого хода раздачи к последнему, обеспечивая корректный BPTT.

---

## 7. Система логирования

### 7.1. Категории логов

`StaticLogger.configure_experiment_logs(experiment_name)` создаёт:

```
logs/
└── {experiment_name}/
    ├── game.log          (buffer: 10000)  — игровые события
    ├── training.log      (buffer: 500)    — статистика обучения
    ├── loss.log          (buffer: 10)     — каждый update сети
    ├── entropy.log       (buffer: 1)      — мгновенно
    ├── validation.log    (buffer: 1)      — мгновенно
    └── summary.log       (buffer: 1)      — мгновенно
```

### 7.2. Формат записей

| Лог | Пример |
|-----|--------|
| **game** | `Игрок NeuralACAgent`, `Ваш выбор: raise`, `S_actor: [...]` |
| **training** | `[Epoch 200] Winrate(Last 500): 54% \| Loss(A/C): 0.021/0.14 \| AvgPot: 150BB` |
| **loss** | `Update (5 hands): Loss=0.0123, Actor=0.0087, Critic=0.0234` |
| **entropy** | `Entropy: 1.0986 \| Actor: 0.0087 \| Critic: 0.0234` |
| **validation** | `🧪 [Validation] Preflop strength check: 52.0% (52/100)` |
| **summary** | `[ACTION FREQ over 1000 games] Fold: 45.2%, Call: 38.1%, Raise: 16.7%` |

### 7.3. Дополнительная валидация

**Action Frequencies** (каждые 200 игр):
```
[ACTION FREQ over 200 games] Fold: 45.2%, Call: 38.1%, Raise: 16.7%
```

**Hand Value Validation** (при вызове `run_validation()`):
```
=== HAND VALUE VALIDATION ===
  AA (тузы)            → Value = +1.2345
  KK (короли)          → Value = +0.9876
  72o (мусор)          → Value = -0.8765
  random (средняя)     → Value = +0.1234
=== END HAND VALUE VALIDATION ===
```

---

## 8. Структура проекта

```
D:\Универ\8 сем\Курсовая\Практика\
├── Poker/
│   ├── __init__.py              # Экспорт всех модулей
│   ├── ActorCritic.py           # Модель ActorCriticNet + NeuralACAgent (Dual-LSTM)
│   ├── NeuralACAgentManager.py  # Менеджер обучения и действий (act, train, update)
│   ├── GameManager.py           # Игровой цикл, ставки, стадии, reward shaping
│   ├── Game.py                  # Состояние стола, блайнды, игроки
│   ├── HandCalculator.py        # Monte Carlo оценка руки (eval7)
│   ├── Logger.py                # StaticLogger (многокатегориальный)
│   ├── NNData.py                # Буфер эпизодов для обучения
│   ├── Player.py                # Базовый класс игрока
│   ├── PlayerManager.py         # Базовый менеджер действий
│   ├── bot.py                   # SimpleGeneticBot
│   ├── BotManager.py            # Менеджер бота
│   ├── NeuralAgent.py           # Legacy DQN-агент
│   ├── NeuralAgentManager.py    # Менеджер DQN-агента
│   ├── RandomPlayer.py          # Случайный игрок
│   ├── RandomPlayerManager.py   # Менеджер случайного игрока
│   ├── CallingPlayer.py         # Игрок, который всегда делает call
│   ├── CallingPlayerManager.py  # Менеджер calling-игрока
│   ├── Deck.py                  # Колода, раздача карт
│   ├── Card.py                  # Класс карты
│   ├── poker_rules.py           # Сравнение рук, правила покера
│   ├── poker_hands.py           # Комбинации (флеш, стрит, фулл-хаус...)
│   └── BotFabric.py             # Фабрика ботов
├── Experiments/
│   ├── Experiment_Aggressor_after_calling.py  # Основной эксперимент
│   ├── Big_Experiment_better_NN.py
│   ├── Experiment_with_AC_model.py
│   ├── testing_without_ent_and_training.py    # Тестирование без энтропии
│   └── ... (25+ экспериментов)
├── tests/
│   ├── conftest.py              # Конфигурация pytest
│   └── mocks.py                 # Моки для тестирования
├── main.py                    # Точка входа (базовый запуск)
├── check_model_keys.py        # Проверка совместимости чекпоинтов
├── models/                    # Сохранённые модели (.pth)
└── logs/                      # Логи экспериментов
```

---

## 9. Эксперименты

### 9.1. Основные эксперименты

| Эксперимент | Файл | Описание |
|-------------|------|----------|
| **Aggressor** | `Experiment_Aggressor_after_calling.py` | NeuralACAgent vs Aggressor бот |
| **Big Experiment** | `Big_Experiment_better_NN.py` | Масштабный эксперимент с улучшенной сетью |
| **All Tactics** | `All_tacktics_experiment.py` | Против всех тактик одновременно |
| **PPO Aggressor** | `Experiment_PPO_Aggressor.py` | Попытка PPO-подобного подхода |
| **Calling Player** | `Experiment_with_calling_player.py` | Against calling бота |
| **Random Player** | `Experiment_with_random_player.py` | Against случайного игрока |
| **Training Mode Off** | `Experiment_with_training_mode_off_*.py` | Inference без exploration |

### 9.2. Параметры типичного эксперимента

```python
num_games = 100_000          # Общее количество игр
num_rounds = 30              # Раундов в каждой игре
learning_num_games = 50      # Игр для обучения (legacy)
learning_num_rounds = 50     # Раундов для обучения (legacy)
```

### 9.3. Основной эксперимент (Experiment_Aggressor_after_calling.py)

- **Загрузка модели:** `Big_Experiment_better_NN.pth`
- **Начальная валидация:** `validate_hand_values()` сразу после загрузки
- **Чекпоинты:** каждые 5000 игр с валидацией и диагностикой
- **График:** matplotlib визуализация win rate over time
- **Финальная статистика:** win rate, количество побед, NNData.show_losses()

---

## 10. Недавние изменения (Апрель 2026)

### 10.1. Шаг 2: LayerNorm для выравнивания масштаба признаков

**Проблема:** Выход первого слоя Критика (5.0, 10.0, 20.0) «забивал» вероятности победы (0.0–1.0) при конкатенации.

**Решение:** Добавлен `nn.LayerNorm(128)` после ReLU и перед Skip Connection.

**Файл:** `ActorCritic.py`

### 10.2. Шаг 1: Логирование распределения действий

**Проблема:** Не было видно, какие действия выбирает модель (риск «схлопывания» в вечный Fold).

**Решение:** Счётчик действий `action_counter` с выводом каждые 1000 игр в `summary.log`.

**Файл:** `NeuralACAgentManager.py`

### 10.3. Шаг 3: Логирование Value по силе руки

**Проблема:** Неизвестно, понимает ли Критик разницу между сильными и слабыми руками.

**Решение:** Метод `validate_hand_values()` тестирует 4 руки (AA, KK, 72o, random) и выводит Value для каждой.

**Файлы:** `ActorCritic.py`, `NeuralACAgentManager.py`

### 10.4. Backpropagation Through Time (BPTT)

**Проблема:** LSTM «амнезия» — скрытое состояние не передавалось между шагами раздачи при обучении.

**Решение:** Замена batch forward на пошаговый цикл с передачей `curr_actor_h` / `curr_critic_h`.

**Файл:** `NeuralACAgentManager.py`

### 10.5. Удаление локальной нормализации Returns и Advantage

**Проблема:** Нормализация внутри траектории искажала исходные награды.

**Решение:** Убраны `(returns - mean) / std` и `(advantage - mean) / std`. Используется глобальная нормализация Advantage по всему батчу.

### 10.6. Экстремальное уменьшение коэффициента энтропии

**Проблема:** Высокая энтропия мешала эксплуатации обученной стратегии.

**Решение:**
- `entropy_coef`: 0.001 → 0.0001 (в 10 раз меньше)
- `entropy_coef_min`: 0.00001
- `entropy_decay`: 0.995
- Добавлены аварийные сбросы энтропии после 2000 и 5000 игр

**Файл:** `NeuralACAgentManager.py`

### 10.7. Action Masking для легальных действий

**Проблема:** Модель могла выбирать нелегальные действия (raise без фишек).

**Решение:** Маска легальных действий `legal_mask` применяется к логитам через `masked_fill(~legal_mask, -1e9)` как в `act()`, так и в `_update_network()`.

**Файл:** `NeuralACAgentManager.py`

### 10.8. Раздельные Learning Rates для Актора и Критика

**Проблема:** Одинаковые LR не позволяли Актору исследовать стратегии быстрее, чем Критик оценивает.

**Решение:**
- Actor LSTM + Net: `lr = 2e-4`
- Critic LSTM + Net: `lr = 1e-4`

**Файл:** `ActorCritic.py`

### 10.9. Reward Shaping: штраф за проигрыш на шоудауне

**Проблема:** Агент не различал «умный фолд» и «проигрыш на шоудауне».

**Решение:** Если игрок не фолднул и проиграл — штраф ×1.5 к net_profit.

**Файл:** `GameManager.py`

### 10.10. Диагностика агента (diagnose_agent)

**Решение:** Комплексная диагностика каждые 2000 игр:
1. `validate_hand_values()` — проверка Критика
2. `log_action_frequencies()` — проверка Актора
3. Вывод гиперпараметров
4. Интерпретация результатов

**Файл:** `NeuralACAgentManager.py`

### 10.11. Надёжная загрузка чекпоинтов с проверкой архитектуры

**Решение:** Метод `load_ac_agent()` теперь:
- Определяет архитектуру чекпоинта (Sequential vs Skip Connection)
- Фильтрует несовместимые веса Критика
- Проверяет совпадение ключей и размеров тензоров
- Выводит детальный отчёт о загрузке

**Файл:** `NeuralACAgentManager.py`

---

## 11. Зависимости

| Библиотека | Назначение |
|------------|------------|
| `torch` | Нейросети, оптимизация, градиенты |
| `torch.nn` | Слои: Linear, LSTM, LayerNorm |
| `torch.distributions` | Categorical для сэмплирования действий |
| `eval7` | Быстрая оценка рук (Cython, Monte Carlo) |
| `numpy` | Массивы, тензоры, буферы |
| `matplotlib` | Графики win rate, loss |
| `random`, `itertools` | Перемешивание, комбинации карт |
| `collections.deque` | Буферы с ограничением размера |
| `os` | Работа с файловой системой (логи, модели) |
| `pytest` | Тестирование (опционально) |

---

## 12. Как запустить

### 12.1. Базовый запуск

```bash
cd Практика
python main.py
```

Запускает 10 игр по 30 раундов с 5 ботами (Aggressor, Tight, Bluff, Balanced, Maniac).

### 12.2. Запуск эксперимента

```bash
cd Практика
python Experiments/Experiment_Aggressor_after_calling.py
```

Запускает 10 000 игр NeuralACAgent vs Aggressor.

### 12.3. Загрузка модели

```python
pm = NeuralACAgentManager(players[1])
pm.load_ac_agent(filename="neural_ac_agent_for_course_LSTM_after_calling.pth")
```

### 12.4. Сохранение модели

```python
pm.save_ac_agent("my_model.pth")
```

Чекпоинты автоматически сохраняются каждые 5000 игр.

---

## 13. Формат чекпоинта (.pth)

```python
checkpoint = {
    'ac_net_state_dict': ...,         # Веса модели
    'optimizer_state_dict': ...,      # Веса оптимизатора
    'gamma': 0.99,
    'actor_size': 10,
    'critic_size': 11,
    'action_size': 3,
    'lstm_hidden': 16,
    'history_len': 10,
    'action_vector_size': 3,
    'name': 'NeuralACAgent',
    'stack': 100,
    'model_type': 'ActorCritic_DualLSTM',
}
```

### 13.1. Доступные модели

Проект содержит 47 сохранённых чекпоинтов в директориях:
- `models/` — основные модели (checkpoint_game_5000.pth ... checkpoint_game_100000.pth)
- `Практика/models/` — локальные копии
- `Практика/Experiments/models/` — экспериментальные модели

**Ключевые модели:**
| Файл | Описание |
|------|----------|
| `Big_Experiment_better_NN.pth` | Основная предобученная модель |
| `neural_ac_agent_for_course_LSTM_after_calling.pth` | Финальная модель для курсовой |
| `checkpoint_game_*.pth` | Чекпоинты каждые 5000 игр (до 100,000) |

### 13.2. Совместимость чекпоинтов

Система загрузки поддерживает три версии архитектуры:
1. **Старая:** Sequential Critic (`critic_net`)
2. **Средняя:** Skip Connections без LayerNorm
3. **Новая:** Skip Connections + LayerNorm (`critic_layer_norm`)

При загрузке автоматически:
- Определяется версия архитектуры
- Фильтруются несовместимые веса
- Выводится детальный отчёт о совпадении ключей

---

## 15. Тестирование

### 15.1. Инфраструктура

Проект содержит базовую инфраструктуру для тестирования:
- `tests/conftest.py` — конфигурация pytest
- `tests/mocks.py` — моки для изоляции компонентов

### 15.2. Проверка совместимости моделей

Скрипт `check_model_keys.py` позволяет проверить совместимость чекпоинтов без загрузки весов.

---

## 16. Контактная информация

> **Автор:** [Студент 8 семестра]  
> **Учебное заведение:** [Университет]  
> **Год:** 2026
