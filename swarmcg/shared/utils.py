import sys

import numpy as np

from . import exceptions


def forward_fill(arr, cond_value):
    """
    Foward fill a list of values with the last valid one.
    """
    valid_val = None
    for i in range(len(arr)):
        if arr[i] != cond_value:
            valid_val = arr[i]
        else:
            j = i
            while valid_val is None and j < len(arr):
                j += 1
                try:
                    if arr[j] != cond_value:
                        valid_val = arr[j]
                        break
                except IndexError as e:
                    msg = (
                        "Unexpected read of the optimization results, "
                        "please check that your simulations have not all been crashing"
                    )
                    raise exceptions.OptimisationResultsError(msg)

            if valid_val is not None:
                arr[i] = valid_val
            else:
                msg = (
                    "All simulations crashed, nothing to display. "
                    "Please check the setup and settings of your optimization run."
                )
                raise exceptions.OptimisationResultsError(msg)

    return arr


def sma(interval, window_size):
    """
    Implement simple moving average with convolution operator.
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def ewma(hist, alpha, windowSize):
    """
    Implement expontential moving average with convolution operator.
    """
    wghts = (1 - alpha) ** np.arange(windowSize)
    wghts = wghts / wghts.sum()
    out = np.full(len(hist), np.nan)
    out = np.convolve(hist, wghts, 'same')
    return out
