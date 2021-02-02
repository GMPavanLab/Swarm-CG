import contextlib
import random
import sys

import MDAnalysis as mda


def print_stdout_forced(*args, **kwargs):
    """Print forced stdout enabled"""
    with contextlib.redirect_stdout(sys.__stdout__):
        print(*args, **kwargs, flush=True)


def set_MDA_backend(ns):
    """Set MDAnalysis backend and number of threads

    ns creates:
        mda_backend
    """
    # NOTE: this is not used because MDA is not properly parallelized, in fact with OpenMP backend it's slower than in serial
    if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
        ns.mda_backend = 'OpenMP'
    else:
        ns.mda_backend = 'serial'


def draw_float(low, high, dg_rnd):
    """Draw random float between given range and apply rounding to given digit"""
    return round(random.uniform(low, high), dg_rnd)  # low and high included
