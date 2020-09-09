import sys

import numpy as np

from . import exceptions


def forward_fill(arr, cond_value):

    # out = np.empty(len(arr))
    valid_val = None
    for i in range(len(arr)):
        if arr[i] != cond_value:
            # out[i] = arr[i]
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
                    msg = """{} Unexpected read of the optimization results, 
                    please check that your simulations have not all been crashing
                    """.format(exceptions.header_error)
                    sys.exit(msg)

            if valid_val is not None:
                # out[i] = valid_val
                arr[i] = valid_val
            else:
                msg = """All simulations crashed, nothing to display.
                Please check the setup and settings of your optimization run
                """
                sys.exit(msg)


def sma(x, window_size):
    """
    Implement simple moving average with convolution operator.
    """
    return ewma(x, 1, window_size)


def ewma(x, alpha, windowSize):
    """
    Implement expontential moving average with convolution operator.
    """
    wghts = (1 - alpha) ** np.arange(windowSize)
    wghts = wghts / wghts.sum()
    return np.convolve(x, wghts, 'same')