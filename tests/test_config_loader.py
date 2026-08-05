import pytest
from config_loader import ProductLoader

def test_load_product_dict():
    loader = ProductLoader()
    expected_product = {
        "id": 1,
        "name": "Product A",
        "price": 10.99,
        "description": "This is product A"
    }
    actual_product = loader.load_product({"id": 1, "name": "Product A", "price": 10.99, "description": "This is product A"})
    assert expected_product == actual_product

def test_load_product_dict_with_missing_key():
    loader = ProductLoader()
    with pytest.raises(KeyError):
        loader.load_product({"id": 1, "name": "Product A"})

def test_load_product_dict_with_invalid_type():
    loader = ProductLoader()
    with pytest.raises(TypeError):
        loader.load_product("Invalid type")

def test_load_product_dict_with_missing_id():
    loader = ProductLoader()
    expected_product = {
        "id": None,
        "name": "Product B",
        "price": 9.99,
        "description": "This is product B"
    }
    actual_product = loader.load_product({"name": "Product B", "price": 9.99, "description": "This is product B"})
    assert expected_product == actual_product

def test_load_product_dict_with_multiple_products():
    loader = ProductLoader()
    products = [
        {"id": 1, "name": "Product A", "price": 10.99, "description": "This is product A"},
        {"id": 2, "name": "Product B", "price": 9.99, "description": "This is product B"}
    ]
    expected_products = [
        {"id": 1, "name": "Product A", "price": 10.99, "description": "This is product A"},
        {"id": 2, "name": "Product B", "price": 9.99, "description": "This is product B"}
    ]
    actual_products = loader.load_products(products)
    assert expected_products == actual_products

def test_load_product_dict_with_empty_list():
    loader = ProductLoader()
    with pytest.raises(ValueError):
        loader.load_products([])
