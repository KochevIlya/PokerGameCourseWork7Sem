class StaticLogger:
    """Статический логгер (singleton)"""

    _instance = None
    _filename = "app.log"
    _buffer_size = 1000
    _buffer = []

    @staticmethod
    def print(*args):
        """Статический метод для логирования"""
        # Создаём экземпляр если его нет
        if StaticLogger._instance is None:
            StaticLogger._instance = StaticLogger()

        msg = ' '.join(str(x) for x in args)
        StaticLogger._buffer.append(msg)

        if len(StaticLogger._buffer) >= StaticLogger._buffer_size:
            StaticLogger._save()

    @staticmethod
    def _save():
        if StaticLogger._buffer:
            try:
                with open(StaticLogger._filename, 'a') as f:
                    f.write('\n'.join(StaticLogger._buffer) + '\n')
                StaticLogger._buffer.clear()
            except:
                StaticLogger._buffer.clear()

    @staticmethod
    def configure(filename=None, buffer_size=None):
        """Настройка логгера"""
        if filename:
            StaticLogger._filename = filename
        if buffer_size:
            StaticLogger._buffer_size = buffer_size

    @staticmethod
    def flush():
        """Принудительная запись"""
        StaticLogger._save()
