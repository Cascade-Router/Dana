class TokenBucket:
    def __init__(self, rate):
        self.rate = rate
        self.tokens = 0
        self.last_update = 0

    def fill(self):
        now = int(time.time())
        if now - self.last_update >= 1 / self.rate:
            self.tokens = self.rate
            self.last_update = now

    def get_available_tokens(self):
        return max(0, self.tokens)

    def get_token(self):
        available = self.get_available_tokens()
        if available > 0:
            self.tokens -= 1
            return True
        else:
            return False


import time
