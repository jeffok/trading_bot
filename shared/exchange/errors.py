
class ExchangeError(Exception):
    pass

class RateLimitError(ExchangeError):
    pass

class AuthError(ExchangeError):
    pass

class TemporaryError(ExchangeError):
    pass
