import pytest
from config_validator import ConfigValidator

def test_config_validator_fails():
    validator = ConfigValidator()
    assert not validator.is_valid({"key": "other"})

def test_config_validator_passes():
    validator = ConfigValidator()
    assert validator.is_valid({"key": "value"})  # should pass
