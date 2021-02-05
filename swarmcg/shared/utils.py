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


def parse_string_args(x):
    try:
        to_float = float(x)
        if int(to_float) - to_float != 0:
            return to_float
        else:
            return int(to_float)
    except ValueError as _:
        return x
