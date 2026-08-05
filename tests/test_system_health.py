import pytest

def test_system_health():
    with pytest.raises(SystemError):
        # Code that should raise SystemError
        raise SystemError("Simulated system error")
