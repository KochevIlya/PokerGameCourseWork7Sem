"""
Скрипт для диагностики несовпадения ключей state_dict модели Actor-Critic.
Запускать из корня проекта: python check_model_keys.py
"""
import torch
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.getcwd())

from Практика.Poker.ActorCritic import ActorCriticNet, NeuralACAgent

# Путь к файлу модели (можно изменить)
CHECKPOINT_PATH = "models/neural_ac_agent_for_course_LSTM_after_calling.pth"

def check_model(checkpoint_path):
    print(f"=== ЗАГРУЗКА CHECKPOINT: {checkpoint_path} ===\n")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Файл не найден: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("Ключи верхнего уровня в checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    print("\n=== КЛЮЧИ STATE_DICT В ФАЙЛЕ ===")
    file_state_dict = checkpoint['ac_net_state_dict']
    for key, value in sorted(file_state_dict.items()):
        print(f"  {key}: {list(value.shape)}")

    # Получаем параметры из checkpoint
    actor_size = checkpoint.get('actor_size', 'N/A')
    critic_size = checkpoint.get('critic_size', 'N/A')
    action_size = checkpoint.get('action_size', 'N/A')
    lstm_hidden = checkpoint.get('lstm_hidden', 'N/A')
    history_len = checkpoint.get('history_len', 'N/A')
    action_vector_size = checkpoint.get('action_vector_size', 'N/A')

    print(f"\nПараметры из checkpoint:")
    print(f"  actor_size: {actor_size}")
    print(f"  critic_size: {critic_size}")
    print(f"  action_size: {action_size}")
    print(f"  lstm_hidden: {lstm_hidden}")
    print(f"  history_len: {history_len}")
    print(f"  action_vector_size: {action_vector_size}")

    # Проверяем что все параметры int
    if any(v == 'N/A' for v in [actor_size, critic_size, action_size, lstm_hidden, history_len, action_vector_size]):
        print("\n⚠️ Некоторые параметры отсутствуют в checkpoint!")
        print("   Модель будет создана с параметрами по умолчанию.")
        actor_size = actor_size if actor_size != 'N/A' else 10
        critic_size = critic_size if critic_size != 'N/A' else 11
        action_size = action_size if action_size != 'N/A' else 3
        lstm_hidden = lstm_hidden if lstm_hidden != 'N/A' else 16
        history_len = history_len if history_len != 'N/A' else 10
        action_vector_size = action_vector_size if action_vector_size != 'N/A' else 3

    # Создаём модель с этими параметрами
    print("\n=== СОЗДАНИЕ ТЕКУЩЕЙ МОДЕЛИ ===")
    model = ActorCriticNet(
        actor_state_size=actor_size,
        critic_state_size=critic_size,
        action_size=action_size,
        history_len=history_len,
        action_input_dim=action_vector_size,
        lstm_hidden=lstm_hidden
    )

    print("\n=== КЛЮЧИ STATE_DICT В ТЕКУЩЕЙ МОДЕЛИ ===")
    current_state_dict = model.state_dict()
    for key, value in sorted(current_state_dict.items()):
        print(f"  {key}: {list(value.shape)}")

    # СРАВНЕНИЕ
    print("\n=== СРАВНЕНИЕ КЛЮЧЕЙ ===")
    file_keys = set(file_state_dict.keys())
    current_keys = set(current_state_dict.keys())

    print(f"Ключей в файле: {len(file_keys)}")
    print(f"Ключей в модели: {len(current_keys)}")

    missing_in_model = file_keys - current_keys
    missing_in_file = current_keys - file_keys
    common_keys = file_keys & current_keys

    if missing_in_model:
        print(f"\n❌ ЕСТЬ В ФАЙЛЕ, НО НЕТ В МОДЕЛИ ({len(missing_in_model)}):")
        for key in sorted(missing_in_model):
            print(f"  - {key}")

    if missing_in_file:
        print(f"\n❌ ЕСТЬ В МОДЕЛИ, НО НЕТ В ФАЙЛЕ ({len(missing_in_file)}):")
        for key in sorted(missing_in_file):
            print(f"  - {key}")

    if not missing_in_model and not missing_in_file:
        print("\n✅ Все ключи совпадают!")
    else:
        print(f"\n⚠️ НАЙДЕНЫ НЕСОВПАДЕНИЯ!")

    # Проверка размеров совпадающих ключей
    print("\n=== ПРОВЕРКА РАЗМЕРОВ СОВПАДАЮЩИХ КЛЮЧЕЙ ===")
    shape_mismatches = []
    for key in sorted(common_keys):
        file_shape = list(file_state_dict[key].shape)
        current_shape = list(current_state_dict[key].shape)
        if file_shape != current_shape:
            shape_mismatches.append(key)
            print(f"  ❌ {key}: файл={file_shape} != модель={current_shape}")

    if not shape_mismatches:
        print("  ✅ Все размеры совпадают!")
    else:
        print(f"\n⚠️ НАЙДЕНЫ НЕСОВПАДЕНИЯ РАЗМЕРОВ ({len(shape_mismatches)})!")

    # Проверка LSTM размеров
    print("\n=== ПРОВЕРКА LSTM РАЗМЕРОВ ===")
    print(f"Actor LSTM:  input_size={model.actor_lstm.input_size}, hidden_size={model.actor_lstm.hidden_size}")
    print(f"Critic LSTM: input_size={model.critic_lstm.input_size}, hidden_size={model.critic_lstm.hidden_size}")

    print(f"\nActor Net первый слой (Linear):")
    print(f"  in_features={model.actor_net[0].in_features}")
    print(f"  Ожидается: actor_state_size ({actor_size}) + lstm_hidden ({lstm_hidden}) = {actor_size + lstm_hidden}")
    match_actor = model.actor_net[0].in_features == (actor_size + lstm_hidden)
    print(f"  {'✅' if match_actor else '❌'} Совпадает: {match_actor}")

    print(f"\nCritic Net первый слой (Linear):")
    print(f"  in_features={model.critic_net[0].in_features}")
    print(f"  Ожидается: critic_state_size ({critic_size}) + lstm_hidden ({lstm_hidden}) = {critic_size + lstm_hidden}")
    match_critic = model.critic_net[0].in_features == (critic_size + lstm_hidden)
    print(f"  {'✅' if match_critic else '❌'} Совпадает: {match_critic}")

    # ПОПЫТКА ЗАГРУЗКИ
    print("\n=== ПОПЫТКА ЗАГРУЗКИ (strict=False) ===")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(file_state_dict, strict=False)

        if missing_keys:
            print(f"❌ Missing keys ({len(missing_keys)}):")
            for k in sorted(missing_keys):
                print(f"  - {k}")

        if unexpected_keys:
            print(f"❌ Unexpected keys ({len(unexpected_keys)}):")
            for k in sorted(unexpected_keys):
                print(f"  - {k}")

        if not missing_keys and not unexpected_keys:
            print("✅ Загрузка прошла успешно!")

    except Exception as e:
        print(f"❌ ОШИБКА ПРИ ЗАГРУЗКЕ: {e}")

    # ИТОГ
    print("\n" + "=" * 60)
    has_issues = missing_in_model or missing_in_file or shape_mismatches
    if has_issues:
        print("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ! Загрузка модели может завершиться с ошибкой.")
        print("   Нужно исправить сохранение/загрузку весов.")
    else:
        print("✅ Модель корректна, загрузка должна пройти успешно!")


if __name__ == "__main__":
    check_model(CHECKPOINT_PATH)
