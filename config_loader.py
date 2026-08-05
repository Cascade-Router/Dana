import json
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    def load_config(self):
        if not self.config_file_path.is_file():
            raise FileNotFoundError(f"Config file not found at path: {self.config_file_path}")

        with open(self.config_file_path, 'r') as config_file:
            return json.load(config_file)

class ConfigLoaderException(Exception):
    pass

def main():
    config_loader = ConfigLoader(Path('config.json'))
    try:
        config = config_loader.load_config()
        print(config)
    except ConfigLoaderException as e:
        print(e)
