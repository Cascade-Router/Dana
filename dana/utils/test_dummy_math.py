import pytest
from dana.utils.dummy_math import divide_numbers

def test_divide_positive():
    assert divide_numbers(10, 2) == 5.0

def test_divide_by_zero():
    # Expecting custom handling or ValueError
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide_numbers(10, 0)