import warnings


def catch_warnings(warning):
    def decorator(function):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=warning)
                result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator
