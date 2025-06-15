import logging

class LoggerManager:
    _instances = {}

    @classmethod
    def get_logger(cls, name):
        if name not in cls._instances:
            logger = cls._build_logger(name)
            cls._instances[name] = logger
        return cls._instances[name]

    @staticmethod
    def _build_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.propagate = False

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger