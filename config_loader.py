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


class ProductLoader:
    def load_product(self, product_data):
        if not isinstance(product_data, dict):
            raise TypeError("Product data must be a dictionary")

        required_keys = ("price", "description")
        missing_keys = [key for key in required_keys if key not in product_data]
        if missing_keys:
            raise KeyError(f"Missing required key '{missing_keys[0]}'")

        return {
            "id": product_data.get("id"),
            "name": product_data.get("name"),
            "price": product_data.get("price"),
            "description": product_data.get("description"),
        }

    def load_products(self, products):
        if not isinstance(products, list):
            raise TypeError("Products must be a list")
        if not products:
            raise ValueError("Products list cannot be empty")
        return [self.load_product(product) for product in products]


def main():
    config_loader = ConfigLoader(Path('config.json'))
    try:
        config = config_loader.load_config()
        print(config)
    except ConfigLoaderException as e:
        print(e)
