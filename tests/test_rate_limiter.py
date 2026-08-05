import pytest
from rate_limiter import TokenBucket

def test_rate_limiter_initialization():
    bucket = TokenBucket(10)
    assert bucket.rate == 10
    assert bucket.burst == 1

def test_rate_limiter_fill():
    bucket = TokenBucket(10)
    bucket.fill()
    assert bucket.tokens > 0

def test_rate_limiter_get_available_tokens():
    bucket = TokenBucket(10, burst=5)
    bucket.fill()
    assert bucket.get_available_tokens() <= 5

def test_rate_limiter_get_token():
    bucket = TokenBucket(1)
    with pytest.raises(ValueError):
        bucket.get_token()

def test_rate_limiter_burst():
    bucket = TokenBucket(10)
    bucket.burst(2)
    assert bucket.tokens >= 12
