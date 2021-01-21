import warnings
from functools import wraps


def catch_warnings(warning):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=warning)
                result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator
