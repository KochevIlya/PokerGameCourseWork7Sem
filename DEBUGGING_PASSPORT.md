# 🛠 Паспорт отладки проекта: Texas Hold'em Poker AI

> **Курсовая работа** | 8 семестр | 2026
> **Цель документа:** Пошаговое руководство по диагностике и отладке всех компонентов системы
> **Дата создания:** 14 апреля 2026

---

## 1. Быстрый старт: чек-лист при любой проблеме

```
[ ] 1. Воспроизвести проблему: запустить эксперимент с нуля
[ ] 2. Проверить логи: logs/{experiment_name}/ — 6 файлов
[ ] 3. Проверить консоль: есть ли ⚠️ ALERT сообщения?
[ ] 4. Проверить модель: загрузить чекпоинт через check_model_keys.py
[ ] 5. Запустить диагностику: diagnose_agent() — встроена в эксперимент
[ ] 6. Проверить градиенты: логи в entropy.log показывают Actor grad
[ ] 7. Проверить данные: NNData buffer заполняется? BATCH_SIZE=4096
```

---

## 2. Структура логов и что в них искать

### 2.1. Расположение

```
logs/
└── {experiment_name}/
    ├── game.log          ← Игровые события (буфер: 10000 строк)
    ├── training.log      ← Статистика обучения (буфер: 500)
    ├── loss.log          ← Каждый update сети (буфер: 10)
    ├── entropy.log       ← Мгновенно при каждом update (буфер: 1)
    ├── validation.log    ← Валидация рук (буфер: 1)
    └── summary.log       ← Финальная статистика (буфер: 1)
```

### 2.2. Ключевые индикаторы по логам

| Лог | Что искать | Норма | Проблема |
|-----|-----------|-------|----------|
| **entropy.log** | `Entropy` | 0.3–0.8 | >1.09 ≈ случайность, ≈0.0 — коллапс |
| **entropy.log** | `Adv(raw)` std | >0.1 | <0.01 — градиенты исчезают |
| **entropy.log** | `Adv(scaled)` std | 5–15 | <1 — слабый сигнал |
| **entropy.log** | `LogitsRange` Δ | >2.0 | <0.5 — логиты сплющены |
| **entropy.log** | `Actor grad` | >0.001 | <1e-6 — Актор не обучается |
| **loss.log** | `Actor` loss | 0.01–0.5 | >5.0 — взрыв градиентов |
| **loss.log** | `Critic` loss | 0.1–2.0 | >10.0 — Критик расходится |
| **training.log** | Winrate | >50% | <40% — агент проигрывает |
| **validation.log** | `Value gap` | >0.5 | <0.1 — Skip Connection не работает |
| **summary.log** | Action Freq | Fold<60% | Fold>80% — агент пассивен |

### 2.3. Формат записей entropy.log (расширенный)

```
Entropy: 0.6543 | Adv(raw): 0.1234±1.2345 | Adv(norm): ±1.0000 | Adv(scaled): ±10.0000 | LogitsRange: [-3.45, 4.12] (Δ=7.57) | Returns: 0.4567 | Values: 0.3456 | Actor grad: 0.012345 | logits_grad: True | Actor: 0.0234 | Critic: 0.1567 | EntCoef: 0.000000 | AdvScale: 10.0
```

**Расшифровка полей:**
- `Entropy` — средняя энтропия действий (ниже = увереннее выбор)
- `Adv(raw)` — mean±std Advantage ДО нормализации (сырой сигнал от Критика)
- `Adv(norm)` — std Advantage ПОСЛЕ нормализации, ДО масштабирования
- `Adv(scaled)` — std Advantage ПОСЛЕ умножения на `advantage_scale`
- `LogitsRange` — [min, max] логитов и их размах (Δ = max - min)
- `Returns` — средние дисконтированные награды в батче
- `Values` — средние предсказания Критика в батче
- `Actor grad` — норма градиента Актора (должна быть >0.001)
- `logits_grad` — требует ли тензор логитов градиентов (всегда True)
- `Actor/Critic` — значения лосс-компонентов
- `EntCoef` — текущий коэффициент энтропии (0.0 = отключена)
- `AdvScale` — множитель Advantage (по умолчанию 10.0)

---

## 3. Диагностика по симптомам

### 3.1. 🚨 «Агент всегда фолдит»

**Симптомы:**
- `summary.log`: `Fold: >80%, Call: <15%, Raise: <5%`
- `game.log`: повторяющиеся «Ваш выбор: fold»
- Winrate <30%

**Шаги диагностики:**

```python
# 1. Проверить распределение действий
pm = NeuralACAgentManager(player)
pm.log_action_frequencies()  # Вывод в summary.log

# 2. Проверить логиты
# В entropy.log: LogitsRange Δ < 0.5 → логиты сплющены
# Δ = max(logits) - min(logits) должно быть > 2.0

# 3. Проверить градиенты Актора
# В entropy.log: Actor grad < 1e-6 → градиенты не текут

# 4. Запустить ручную проверку
pm.validate_hand_values()  # Критик должен различать AA и 72o
```

**Возможные причины:**
| Причина | Как проверить | Решение |
|---------|--------------|---------|
| Энтропия слишком низкая | `EntCoef: 0.000000` | Установить `entropy_coef = 0.001` |
| Advantage std ≈ 0 | `Adv(raw) std < 0.01` | Проверить Критика (не обучается?) |
| Логиты сплющены | `LogitsRange Δ < 0.5` | Повысить `advantage_scale` до 20.0 |
| Actor LR слишком мал | В ActorCritic.py | Повысить `actor_lr` до 1e-3 |
| Action Masking блокирует всё | `game.log` → legal_mask | Проверить `can_raise` логику |

### 3.2. 🚨 «Агент делает случайные действия»

**Симптомы:**
- `entropy.log`: `Entropy > 1.09` ≈ ln(3) = полная случайность
- Action Freq: Fold/Call/Raise ≈ 33%/33%/33%
- Winrate ≈ 33% (случайный уровень)

**Шаги диагностики:**

```python
# 1. Проверить энтропию
# Entropy ≈ 1.0986 → равномерное распределение по 3 действиям

# 2. Проверить логиты
# LogitsRange Δ < 0.1 → все логиты почти равны

# 3. Проверить Advantage
# Adv(raw) std < 0.1 → Критик не даёт полезного сигнала
```

**Возможные причины:**
| Причина | Как проверить | Решение |
|---------|--------------|---------|
| Критик не обучается | `Critic loss` не уменьшается | Проверить critic_lr, Huber delta |
| Returns шум | `Returns` std > 5.0 | Reward shaping, нормализация |
| Модель не загрузилась | check_model_keys.py | Перезагрузить чекпоинт |
| Случайная инициализация | Первые 1000 игр | Это норма для начала обучения |

### 3.3. 🚨 «Loss взрывается»

**Симптомы:**
- `loss.log`: `Loss > 10.0`, `Actor > 5.0`
- Консоль: `NaN` или `inf` в выводах
- Модель перестаёт обучаться

**Шаги диагностики:**

```python
# 1. Проверить градиенты
# В entropy.log: Actor grad > 100 → взрыв градиентов

# 2. Проверить Advantage
# Adv(scaled) std > 100 → слишком большое масштабирование

# 3. Проверить логиты
# LogitsRange Δ > 50 → логиты уходят в бесконечность
```

**Возможные причины:**
| Причина | Как проверить | Решение |
|---------|--------------|---------|
| advantage_scale слишком большой | `Adv(scaled) std > 50` | Уменьшить до 5.0 |
| Gradient clipping отключён | Проверить код | `max_norm=0.5` должен быть |
| Actor LR слишком высокий | `actor_lr > 1e-3` | Снизить до 5e-4 |
| Returns не нормализованы | `Returns` mean > 10 | Нормализовать через BB |

### 3.4. 🚨 «Критик не различает руки»

**Симптомы:**
- `validation.log`: `Value gap < 0.1`
- AA и 72o имеют почти одинаковые Value
- Agent не знает, когда блефовать

**Шаги диагностики:**

```python
# 1. Запустить валидацию
pm.validate_hand_values()

# Ожидаемый вывод:
# AA (тузы)            → Value = +1.2345
# KK (короли)          → Value = +0.9876
# 72o (мусор)          → Value = -0.8765
# random (средняя)     → Value = +0.1234
# Value gap = 2.1110 — Критик хорошо различает руки!
```

**Возможные причины:**
| Причина | Как проверить | Решение |
|---------|--------------|---------|
| Критик не обучается | `Critic loss` ≈ const | Проверить critic_lr, returns |
| Skip Connection сломан | check_model_keys.py | Проверить critic_fc2 веса |
| LayerNorm сбивает | `validation.log` | Проверить critic_layer_norm grad |
| HandCalculator врёт | Сравнить с таблицами | Проверить eval7 итерации |

### 3.5. 🚨 «Модель не загружается»

**Симптомы:**
- `RuntimeError: Error(s) in loading state_dict`
- Ключи не совпадают
- Размеры тензоров разные

**Шаги диагностики:**

```bash
# Из корня проекта:
python check_model_keys.py models/ваш_чекпоинт.pth
```

**Скрипт проверит:**
- Все ключи в чекпоинте
- Все ключи в текущей модели
- Недостающие ключи (missing_in_model, missing_in_file)
- Несовпадение размеров тензоров
- LSTM hidden размеры
- Linear layer input/output размеры
- Попробует загрузить с `strict=False` и отчитывается

**Возможные причины:**
| Причина | Как проверить | Решение |
|---------|--------------|---------|
| Старая архитектура (Sequential) | `missing_in_model: critic_fc1` | Загрузить с strict=False |
| Новый LayerNorm отсутствует | `missing_in_model: critic_layer_norm` | Это ожидаемо для старых чекпоинтов |
| LSTM размер другой | `lstm_hidden mismatch` | Пересобрать модель с правильным размером |
| Файл битый | `torch.load` ошибка | Скачать/восстановить из бэкапа |

---

## 4. Инструменты диагностики

### 4.1. check_model_keys.py

**Расположение:** `D:\Универ\8 сем\Курсовая\check_model_keys.py`

**Использование:**
```bash
# Из корня проекта:
python check_model_keys.py
# Или конкретный файл:
python check_model_keys.py models/checkpoint_game_50000.pth
```

**Что делает:**
1. Загружает чекпоинт и печатает все ключи
2. Создаёт свежую модель с параметрами из чекпоинта
3. Сравнивает ключи чекпоинта и модели
4. Проверяет размеры тензоров для всех совпадающих ключей
5. Проверяет LSTM и LayerNorm размеры
6. Пытается загрузить с `strict=False` и отчитывается

**Пример вывода:**
```
======================================================================
🔍 СРАВНЕНИЕ КЛЮЧЕЙ ЧЕКПОИНТА И МОДЕЛИ
======================================================================
📁 Ключей в файле: 18
📁 Ключей в модели: 18
🔑 Совпадающих ключей: 18
❌ Недостающих в модели: 0
❌ Недостающих в файле: 0
✅ Все ключи совпали!
```

### 4.2. diagnose_agent()

**Расположение:** `NeuralACAgentManager.py`, метод `diagnose_agent()`

**Вызов:**
```python
pm = NeuralACAgentManager(player)
pm.load_ac_agent("models/my_model.pth")
pm.diagnose_agent()
```

**Что делает:**
1. `validate_hand_values()` — проверяет Критика на 4 руках (AA, KK, 72o, random)
2. `log_action_frequencies()` — проверяет распределение действий за последние 1000 игр
3. Выводит текущие гиперпараметры (entropy_coef, LR, game_counter)
4. Интерпретирует результаты (Критик знает vs Актор фолдит → проблема)

**Пример вывода:**
```
============================================================
🔍 AGENT DIAGNOSTICS
============================================================
=== HAND VALUE VALIDATION ===
  AA (тузы)            → Value = +1.2345
  KK (короли)          → Value = +0.9876
  72o (мусор)          → Value = -0.8765
  random (средняя)     → Value = +0.1234
✅ Value gap = 2.1110 — Критик хорошо различает руки!

[ACTION FREQ over 1000 games]
  Fold:  45.2%
  Call:  38.1%
  Raise: 16.7%

📊 HYPERPARAMS:
   entropy_coef = 0.000100
   Actor LR = 2e-4, Critic LR = 1e-4
   games_played = 50000
   entropy_update_interval = 200
```

### 4.3. validate_hand_values()

**Расположение:** `NeuralACAgentManager.py`, метод `validate_hand_values()`

**Вызов:**
```python
pm.validate_hand_values()
```

**Что делает:**
- Загружает 4 тестовые руки в Критик
- Собирает Value для каждой руки
- Считает Value gap = Value(AA) - Value(72o)
- Оценивает работу Skip Connection

**Пороги:**
| Value gap | Оценка |
|-----------|--------|
| >0.5 | ✅ Критик хорошо различает руки |
| 0.1–0.5 | ⚡ Критик сомневается, нужно дообучить |
| <0.1 | ⚠️ Skip Connection может не работать |

### 4.4. log_action_frequencies()

**Расположение:** `NeuralACAgentManager.py`, метод `log_action_frequencies()`

**Вызов:**
```python
pm.log_action_frequencies()  # Вызывается автоматически каждые 200 игр
```

**Что делает:**
- Выводит распределение Fold/Call/Raise за последние N игр
- Пишет в `summary.log` и в консоль

**Нормальные распределения:**
| Стратегия | Fold | Call | Raise |
|-----------|------|------|-------|
| Tight | 50–60% | 30–40% | 5–10% |
| Aggressive | 25–35% | 30–40% | 30–40% |
| Balanced | 35–45% | 35–45% | 15–25% |
| **Проблема** | >80% | <15% | <5% |

---

## 5. Консольные ALERT-сообщения

Код автоматически выводит предупреждения в консоль при аномалиях:

```python
⚠️ [ALERT] Advantage std=0.000012 — КРИТИЧЕСКИ МАЛЕНЬКИЙ! Градиенты могут исчезать.
⚠️ [ALERT] Entropy=1.0986 ≈ ln(3) — модель выдаёт случайные действия!
⚠️ [ALERT] Actor gradient norm=0.00000001 ≈ 0 — Актор НЕ ОБУЧАЕТСЯ!
```

**Реакция на каждый ALERT:**

| ALERT | Причина | Действие |
|-------|---------|----------|
| Advantage std маленький | Критик даёт одинаковые Value | Дообучить Критика, проверить returns |
| Entropy ≈ ln(3) | Все логиты равны | Проверить Advantage, Критик, LR |
| Actor gradient ≈ 0 | Градиенты не текут | Проверить Advantage std, advantage_scale, masking |

---

## 6. Отладка обучения: пошаговый чек-лист

### 6.1. Перед запуском

```
[ ] check_model_keys.py — чекпоинт совместим
[ ] validate_hand_values() — Value gap > 0.5
[ ] Гиперпараметры в норме (LR, entropy_coef, advantage_scale)
[ ] Логи настроены (configure_experiment_logs вызван)
[ ] BUFFER_SIZE в NNData = 4096
```

### 6.2. Первые 1000 игр

```
[ ] entropy.log появляется (буфер: 1 — мгновенно)
[ ] Entropy уменьшается от ~1.1 к ~0.7
[ ] Actor grad > 0.001
[ ] Adv(raw) std > 0.1
[ ] Консоль: нет ⚠️ ALERT
```

### 6.3. 1000–5000 игр

```
[ ] training.log: Winrate растёт
[ ] loss.log: Actor loss стабилизируется (0.01–0.5)
[ ] LogitsRange Δ растёт (от 0.5 к 3.0+)
[ ] EntCoef уменьшается (decay 0.995 каждые 200 игр)
[ ] summary.log: Action Freq — Fold уменьшается
```

### 6.4. 5000–50000 игр

```
[ ] training.log: Winrate > 50%
[ ] loss.log: Critic loss < 2.0
[ ] Adv(scaled) std ≈ 5–15
[ ] validation.log: Value gap > 0.5
[ ] Чекпоинты сохраняются каждые 5000 игр
```

### 6.5. После 50000 игр

```
[ ] training.log: Winrate стабилизировалась
[ ] Entropy ≈ 0.3–0.6 (уверенный выбор)
[ ] diagnose_agent() — все проверки пройдены
[ ] Чекпоинт загружается без ошибок
```

---

## 7. Гиперпараметры: значения и влияние

### 7.1. Текущие значения (после апрельских изменений)

| Параметр | Значение | Файл | Влияние |
|----------|----------|------|---------|
| `advantage_scale` | **10.0** | NeuralACAgentManager.py:45 | Усиление сигнала для Актора |
| `entropy_coef` | **0.0** | NeuralACAgentManager.py:36 | 0.0 = максимальная эксплуатация |
| `actor_lr` | **1e-3** | ActorCritic.py:108 | Скорость обучения Актора |
| `critic_lr` | **1e-4** | ActorCritic.py:111–115 | Скорость обучения Критика |
| `gamma` | 0.99 | ActorCritic.py:118 | Дисконтирование наград |
| `BATCH_SIZE` | 4096 | NNData.py:8 | Шагов перед обновлением |
| `gradient_clip` | 0.5 | NeuralACAgentManager.py:663 | Защита от взрыва градиентов |
| `huber_delta` | 1.0 | NeuralACAgentManager.py:642 | Параметр Huber Loss |
| `history_len` | 10 | NeuralACAgentManager.py:33 | Длина истории оппонентов |
| `lstm_hidden` | 16 | ActorCritic.py | Размер LSTM скрытого состояния |

### 7.2. Рекомендуемые диапазоны для экспериментов

| Параметр | Мин | Текущий | Макс | Что менять |
|----------|-----|---------|------|------------|
| `advantage_scale` | 1.0 | 10.0 | 50.0 | Если логиты сплющены — поднять |
| `entropy_coef` | 0.0 | 0.0 | 0.01 | Если агент фолдит — поднять до 0.001 |
| `actor_lr` | 1e-4 | 1e-3 | 5e-3 | Если обучение медленное — поднять |
| `critic_lr` | 1e-5 | 1e-4 | 5e-4 | Если Критик расходится — снизить |

---

## 8. Отладка конкретных компонентов

### 8.1. ActorCriticNet

**Файл:** `Практика/Poker/ActorCritic.py`

**Что логировать:**
```python
# Внутри forward() модели:
print(f"Actor logits: {logits}")
print(f"Critic value: {value}")
print(f"Actor LSTM hidden state shape: {actor_h.shape}")
print(f"Critic LSTM hidden state shape: {critic_h.shape}")
```

**Типичные проблемы:**
| Проблема | Симптом | Решение |
|----------|---------|---------|
| LSTM «забывает» | Градиенты не текут через шаги | Проверить BPTT цикл |
| Skip Connection не работает | Value gap < 0.1 | Проверить critic_fc2 входы |
| LayerNorm сбивает | Критик нестабилен | Проверить critic_layer_norm |

### 8.2. NeuralACAgentManager

**Файл:** `Практика/Poker/NeuralACAgentManager.py`

**Ключевые методы:**

| Метод | Назначение | Когда вызывать |
|-------|-----------|----------------|
| `act()` | Выбор действия | Каждый ход в игре |
| `train()` | Сбор траектории | Каждый ход в режиме обучения |
| `_update_network()` | Обновление весов | Когда NNData buffer заполнен (4096 шагов) |
| `validate_hand_values()` | Проверка Критика | При загрузке модели, каждые 5000 игр |
| `diagnose_agent()` | Комплексная проверка | Каждые 2000 игр, при проблемах |
| `load_ac_agent()` | Загрузка чекпоинта | Перед началом эксперимента |
| `save_ac_agent()` | Сохранение модели | Автоматически каждые 5000 игр |

### 8.3. GameManager

**Файл:** `Практика/Poker/GameManager.py`

**Что проверять:**
- Reward shaping: `net_profit *= 1.5` при проигрыше на шоудауне
- Нормализация: `final_reward = net_profit / min_bet` (в Big Blinds)
- Action masking: `legal_mask = [True, can_raise, True]`

**Типичные проблемы:**
| Проблема | Симптом | Решение |
|----------|---------|---------|
| Returns шум | Adv(raw) std > 10 | Проверить reward shaping |
| Маскирование блокирует raise | Agent никогда не raise | Проверить `can_raise` условие |
| Неправильные награды | Критик не обучается | Проверить формулу net_profit |

### 8.4. HandCalculator

**Файл:** `Практика/Poker/HandCalculator.py`

**Что проверять:**
- Monte Carlo итерации: 200 (по умолчанию)
- eval7 библиотека установлена
- Входные данные: hole cards + community cards

**Типичные проблемы:**
| Проблема | Симптом | Решение |
|----------|---------|---------|
| eval7 не установлен | ImportError | `pip install eval7` |
| Мало итераций | hand_strength шум | Увеличить до 500–1000 |
| Неправильные карты | Value gap < 0.1 | Проверить входные данные |

---

## 9. Отладочные скрипты

### 9.1. Быстрая проверка модели

```python
# Создать файл quick_check.py в корне проекта:
from Практика.Poker.ActorCritic import ActorCriticNet, NeuralACAgent
from Практика.Poker.NeuralACAgentManager import NeuralACAgentManager
from Практика.Poker.Player import Player
import torch

# 1. Создать модель
player = Player("Test", 100)
agent = NeuralACAgent(player)
pm = NeuralACAgentManager(player)

# 2. Проверить forward pass
s_actor = torch.randn(10)
s_critic = torch.randn(11)
history = torch.randn(10, 3)
logits, value, _, _ = player.ac_net(s_actor.unsqueeze(0), s_critic.unsqueeze(0), history.unsqueeze(0))
print(f"Logits: {logits}")
print(f"Value: {value}")
print(f"Logits range: {logits.max() - logits.min():.4f}")

# 3. Загрузить чекпоинт
pm.load_ac_agent("models/checkpoint_game_50000.pth")

# 4. Прогнать диагностику
pm.validate_hand_values()
pm.diagnose_agent()
```

### 9.2. Проверка обучения

```python
# Внутри _update_network() добавить временно:
print(f"\n=== UPDATE DEBUG ===")
print(f"Trajectories: {num_trajectories}")
print(f"Global log_probs shape: {global_log_probs.shape}")
print(f"Global values shape: {global_values.shape}")
print(f"Global returns shape: {global_returns.shape}")
print(f"Advantage mean/std: {global_advantage.mean():.4f}/{global_advantage.std():.4f}")
print(f"Actor loss: {actor_loss.item():.6f}")
print(f"Critic loss: {critic_loss.item():.6f}")
print(f"Entropy: {global_entropy.item():.6f}")
print(f"Total loss: {total_loss.item():.6f}")
```

### 9.3. Проверка градиентов

```python
# После backward(), перед optimizer.step():
for name, param in self.player.ac_net.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}, grad_norm={param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

---

## 10. Чек-поинты: управление и валидация

### 10.1. Доступные модели

| Файл | Описание | Примерный возраст |
|------|----------|-------------------|
| `Big_Experiment_better_NN.pth` | Основная предобученная модель | ~50000 игр |
| `neural_ac_agent_for_course_LSTM_after_calling.pth` | Финальная для курсовой | ~100000 игр |
| `checkpoint_game_5000.pth` … `checkpoint_game_100000.pth` | Чекпоинты каждые 5000 игр | 5000–100000 игр |

### 10.2. Валидация перед загрузкой

```bash
# Всегда запускать перед загрузкой в эксперимент:
python check_model_keys.py models/ваш_чекпоинт.pth
```

### 10.3. Формат чекпоинта

```python
checkpoint = {
    'ac_net_state_dict': ...,         # Веса модели (обязательно)
    'optimizer_state_dict': ...,      # Веса оптимизатора (опционально)
    'gamma': 0.99,                    # Дисконтирование
    'actor_size': 10,                 # Размер входа Актора
    'critic_size': 11,                # Размер входа Критика
    'action_size': 3,                 # Количество действий
    'lstm_hidden': 16,                # Размер LSTM
    'history_len': 10,                # Длина истории
    'action_vector_size': 3,          # Размер вектора действия
    'name': 'NeuralACAgent',          # Имя агента
    'stack': 100,                     # Стартовый стек
    'model_type': 'ActorCritic_DualLSTM',  # Тип модели
}
```

### 10.4. Совместимость архитектур

| Версия | Критик | LayerNorm | Skip Connect | Загрузка |
|--------|--------|-----------|--------------|----------|
| **Старая** | Sequential | ❌ | ❌ | strict=False, critic с нуля |
| **Средняя** | Skip Connect | ❌ | ✅ | strict=False, layer_norm с нуля |
| **Новая** | Skip Connect | ✅ | ✅ | strict=True, полная загрузка |

---

## 11. Часто задаваемые вопросы

### Q: Энтропия упала до 0.0 — это нормально?

**A:** Если `entropy_coef = 0.0` — да, энтропия будет падать, так как нет бонуса за exploration. Актор эксплуатирует текущие знания. Если агент при этом фолдит всегда — поднять `entropy_coef` до 0.001.

### Q: Advantage std = 0.0 после нормализации — что делать?

**A:** Это значит, что Критик даёт одинаковые Value для всех состояний в батче. Проверить:
1. Критик обучается? (Critic loss уменьшается)
2. Returns различаются? (returns std > 0.1)
3. HandCalculator работает? (hand_strength не константа)

### Q: Logits Range Δ не растёт — логиты сплющены

**A:** Это основная проблема, ради которой внесены апрельские изменения:
1. `advantage_scale = 10.0` должен раздвинуть логиты
2. `actor_lr = 1e-3` должен ускорить обучение
3. Если не помогло — поднять `advantage_scale` до 20–50

### Q: Как узнать, какая модель лучшая?

**A:** Сравнить по:
1. **Winrate** в `training.log` (чем выше, тем лучше)
2. **Value gap** в `validation.log` (чем выше, тем лучше Критик)
3. **Action Freq** в `summary.log` (Fold < 60%, Raise > 10%)
4. **Entropy** в `entropy.log` (0.3–0.8 — уверенный, но не случайный выбор)

### Q: Можно ли продолжить обучение с старого чекпоинта?

**A:** Да. Загрузить чекпоинт и продолжить эксперимент — оптимизатор перезапустится с новыми LR, но веса модели сохранятся. Это ожидаемое поведение.

### Q: Где смотреть полные логи?

**A:** `logs/{experiment_name}/` — 6 файлов. Открыть любым текстовым редактором. Большие файлы (game.log > 1GB) — использовать `tail` или открыть в Notepad++.

---

## 12. Контактная информация для экстренной помощи

Если ничего не помогает:

1. **check_model_keys.py** — проверить совместимость
2. **diagnose_agent()** — комплексная проверка
3. **Все 6 логов** — прочитать от начала до конца
4. **Консоль** — проверить ALERT сообщения
5. **Градиенты** — включить временную отладку (раздел 9.3)

---

## 13. История изменений паспорта

| Дата | Изменение | Автор |
|------|-----------|-------|
| 14.04.2026 | Создание паспорта отладки | Студент 8 семестра |
| 14.04.2026 | Добавлены: advantage_scale, расширенное логирование, новые пороги | Студент 8 семестра |
