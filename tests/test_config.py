import pytest
from config import Product

def test_product_init():
    product = Product("Test Product", 19.99)
    assert product.name == "Test Product"
    assert product.price == 19.99

def test_product_init_invalid_price():
    with pytest.raises(ValueError):
        Product("Test Product", -1)

def test_product_init_missing_price():
    with pytest.raises(ValueError):
        Product("Test Product")

def test_product_str():
    product = Product("Test Product", 19.99)
    assert str(product) == "Test Product ($19.99)"

def test_product_eq():
    product1 = Product("Test Product", 19.99)
    product2 = Product("Test Product", 19.99)
    assert product1 == product2

def test_product_ne():
    product1 = Product("Test Product", 19.99)
    product2 = Product("Different Product", 19.99)
    assert product1 != product2
