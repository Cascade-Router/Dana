class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

class Config:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

config = Config()
