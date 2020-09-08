import sys

import numpy as np

import exceptions


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

# simple moving average
def sma(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

# exponential moving average
def ewma(a, alpha, windowSize):
    wghts = (1 - alpha) ** np.arange(windowSize)
    wghts /= wghts.sum()
    out = np.full(len(a), np.nan)
    # out[windowSize-1:] = np.convolve(a, wghts, 'valid')
    out = np.convolve(a, wghts, 'same')
    return out