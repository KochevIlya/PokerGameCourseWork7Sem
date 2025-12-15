import os


class StaticLogger:
    """Статический логгер (singleton)"""

    _instance = None
    _filename = "app.log"
    _buffer_size = 1000
    _buffer = []
    _debug = False

    @staticmethod
    def print(*args):
        """Статический метод для логирования"""

        if StaticLogger._instance is None:
            StaticLogger._instance = StaticLogger()

        msg = ' '.join(str(x) for x in args)

        if StaticLogger._debug:
            print(f"[LOGGER] Добавляем в буфер: {msg[:50]}...")

        StaticLogger._buffer.append(msg)

        if StaticLogger._debug:
            print(f"[LOGGER] Размер буфера: {len(StaticLogger._buffer)}/{StaticLogger._buffer_size}")

        if len(StaticLogger._buffer) >= StaticLogger._buffer_size:
            if StaticLogger._debug:
                print(f"[LOGGER] Буфер полон, вызываем _save()")
            StaticLogger._save()

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