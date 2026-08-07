class Product:
    def __init__(self, name, price=None):
        if price is None or price < 0:
            raise ValueError(f"invalid price: {price!r}")
        self.name = name
        self.price = price

    def __str__(self):
        return f"{self.name} (${self.price})"

    def __eq__(self, other):
        if not isinstance(other, Product):
            return NotImplemented
        return self.name == other.name and self.price == other.price

class Config:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

config = Config()
