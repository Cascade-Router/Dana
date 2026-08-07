import time


class TokenBucket:
    def __init__(self, rate, burst=1):
        self.rate = rate
        self.burst = burst
        self.tokens = 0
        self.last_update = 0

    def fill(self):
        now = int(time.time())
        if now - self.last_update >= 1 / self.rate:
            self.tokens = min(self.rate, self.burst)
            self.last_update = now

    def get_available_tokens(self):
        return max(0, self.tokens)

    def get_token(self):
        available = self.get_available_tokens()
        if available <= 0:
            raise ValueError("no tokens available")
        self.tokens -= 1
        return True

    def add_burst(self, extra):
        self.tokens = self.rate + extra
