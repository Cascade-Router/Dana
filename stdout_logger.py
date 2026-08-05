import logging

class StdoutLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self._logger = logging.getLogger(name)

    def log(self, level, msg, *args, **kwargs):
        print(f"{self.name}: {msg}", file=self._logger)

def configure_stdout_logger():
    logger = StdoutLogger("stdout")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
