import os
from collections import defaultdict


class StaticLogger:
    """Статический логгер (singleton) с поддержкой множественных лог-файлов"""

    _instance = None
    _filename = "app.log"
    _buffer_size = 1000
    _buffer = []
    _debug = False

    # === Поддержка множественных логгеров ===
    _experiment_mode = False
    _experiment_dir = ""
    _category_buffers = {}  # { category: [messages] }
    _category_buffers_size = {}  # { category: buffer_size }
    _default_category = "game"  # категория по умолчанию для StaticLogger.print()

    @staticmethod
    def print(*args):
        """Статический метод для логирования (пишет в категорию по умолчанию)"""
        category = StaticLogger._default_category if StaticLogger._experiment_mode else None
        StaticLogger.print_to(category, *args)

    @staticmethod
    def print_to(category: str, *args):
        """
        Запись сообщения в конкретную категорию лога.
        Если category=None, использует старый режим (один файл).
        """
        msg = ' '.join(str(x) for x in args)

        # Старый режим (один файл)
        if category is None:
            if StaticLogger._instance is None:
                StaticLogger._instance = StaticLogger()
            StaticLogger._buffer.append(msg)
            if len(StaticLogger._buffer) >= StaticLogger._buffer_size:
                StaticLogger._save()
            return

        # Новый режим (множественные логи)
        if category not in StaticLogger._category_buffers:
            StaticLogger._category_buffers[category] = []
            StaticLogger._category_buffers_size[category] = StaticLogger._buffer_size

        StaticLogger._category_buffers[category].append(msg)

        if len(StaticLogger._category_buffers[category]) >= StaticLogger._category_buffers_size[category]:
            StaticLogger._save_category(category)

    @staticmethod
    def _save_category(category: str):
        """Запись буфера конкретной категории в файл"""
        if category not in StaticLogger._category_buffers:
            return

        buffer = StaticLogger._category_buffers[category]
        if not buffer:
            return

        try:
            filename = f"{category}.log"
            filepath = os.path.join(StaticLogger._experiment_dir, filename)

            # Создаём директорию если нужно
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(filepath, 'a', encoding='utf-8') as f:
                content = '\n'.join(buffer) + '\n'
                f.write(content)

            buffer.clear()

        except Exception as e:
            if StaticLogger._debug:
                print(f"[LOGGER] ОШИБКА записи категории {category}: {e}")
            buffer.clear()

    @staticmethod
    def configure_experiment_logs(experiment_name: str, default_buffer: int = 500):
        """
        Создаёт структуру папок для эксперимента и настраивает категориальные логи.

        Структура:
        📁 logs/
          📁 {experiment_name}/
            ├── 📄 game.log          (buffer: 2000 — редко пишется)
            ├── 📄 training.log      (buffer: 50  — почти онлайн)
            ├── 📄 loss.log          (buffer: 10  — почти онлайн)
            ├── 📄 entropy.log       (buffer: 1   — мгновенно)
            ├── 📄 validation.log    (buffer: 1   — мгновенно)
            └── 📄 summary.log       (buffer: 1   — мгновенно)
        """
        # Определяем базовую директорию логов
        base_log_dir = os.path.join(os.getcwd(), "logs")
        StaticLogger._experiment_dir = os.path.join(base_log_dir, experiment_name)

        # Создаём директорию
        os.makedirs(StaticLogger._experiment_dir, exist_ok=True)

        # Разные размеры буфера для категорий
        category_configs = {
            "game": 10000,       # Большой буфер — много игровых событий
            "training": 500,     # Почти онлайн — статистика обучения
            "loss": 10,         # Почти онлайн — каждый update сети
            "entropy": 1,       # Мгновенно — энтропия при каждом update
            "validation": 1,    # Мгновенно — валидационные сообщения
            "summary": 1,       # Мгновенно — финальная статистика
        }

        # Очищаем старые логи если существуют
        for cat, buf_size in category_configs.items():
            filepath = os.path.join(StaticLogger._experiment_dir, f"{cat}.log")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('')
            StaticLogger._category_buffers[cat] = []
            StaticLogger._category_buffers_size[cat] = buf_size

        # Включаем режим эксперимента
        StaticLogger._experiment_mode = True
        StaticLogger._buffer_size = default_buffer

        if StaticLogger._debug:
            print(f"[LOGGER] Experiment logs configured: {StaticLogger._experiment_dir}")
            print(f"[LOGGER] Categories: {list(category_configs.keys())}")
            print(f"[LOGGER] Buffer sizes: {category_configs}")

    @staticmethod
    def flush_all():
        """Принудительная запись всех категорий"""
        if StaticLogger._experiment_mode:
            for category in list(StaticLogger._category_buffers.keys()):
                StaticLogger._save_category(category)
        else:
            StaticLogger.flush()

    @staticmethod
    def _save():
        if StaticLogger._debug:
            print(f"[LOGGER] _save() вызван")
            print(f"[LOGGER] Буфер для записи: {len(StaticLogger._buffer)} сообщений")
            print(f"[LOGGER] Файл: {StaticLogger._filename}")
            print(f"[LOGGER] Абсолютный путь: {os.path.abspath(StaticLogger._filename)}")
            print(
                f"[LOGGER] Директория существует: {os.path.exists(os.path.dirname(os.path.abspath(StaticLogger._filename)))}")

        if StaticLogger._buffer:
            try:
                if StaticLogger._debug:
                    print(f"[LOGGER] Пытаюсь записать в файл...")

                file_path = os.path.abspath(StaticLogger._filename)
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)


                with open(StaticLogger._filename, 'a', encoding='utf-8') as f:
                    content = '\n'.join(StaticLogger._buffer) + '\n'
                    f.write(content)

                    if StaticLogger._debug:
                        print(f"[LOGGER] Успешно записано {len(content)} байт")
                        print(f"[LOGGER] Первые 100 символов: {content[:100]}")

                StaticLogger._buffer.clear()

                if StaticLogger._debug:
                    print(f"[LOGGER] Буфер очищен")

            except Exception as e:
                if StaticLogger._debug:
                    print(f"[LOGGER] ОШИБКА записи: {e}")
                    import traceback
                    traceback.print_exc()
                StaticLogger._buffer.clear()
        else:
            if StaticLogger._debug:
                print(f"[LOGGER] Буфер пуст, нечего записывать")

    @staticmethod
    def configure(filename=None, buffer_size=None, debug=False):
        """Настройка логгера"""
        if StaticLogger._debug:
            print(f"[LOGGER] configure() вызван с filename={filename}, buffer_size={buffer_size}")

        if filename:
            StaticLogger._filename = filename
        if buffer_size:
            StaticLogger._buffer_size = buffer_size
        StaticLogger._debug = debug

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('')
        if StaticLogger._debug:
            print(f"[LOGGER] Новые настройки: файл={StaticLogger._filename}, буфер={StaticLogger._buffer_size}")

    @staticmethod
    def flush():
        """Принудительная запись"""
        if StaticLogger._debug:
            print(f"[LOGGER] flush() вызван")
        StaticLogger._save()