import logging
from core_bus import CoreBus

class LoggerPlugin:
    def __init__(self, bus: CoreBus):
        self.bus = bus
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def stop(self):
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)


def register_plugin(bus: CoreBus):
    return LoggerPlugin(bus)


__all__ = ['register_plugin']
