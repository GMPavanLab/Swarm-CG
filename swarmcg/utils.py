import contextlib
import sys


def print_stdout_forced(*args, **kwargs):
    """Print forced stdout enabled"""
    with contextlib.redirect_stdout(sys.__stdout__):
        print(*args, **kwargs, flush=True)
