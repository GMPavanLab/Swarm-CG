import numpy as np
from numpy.testing import assert_almost_equal

from swarmcg.shared import utils


def test_sma():
    # given:
    x = np.arange(10)

    # when:
    window_size = 5
    result = utils.sma(x, window_size)

    # then:
    expected = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=float)
    assert_almost_equal(result, expected)

def test_ema():
    # given:
    x = np.arange(10)

    # when:
    window_size = 5
    result = utils.ewma(x, 1, window_size)

    # then:
    expected = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=float)
    assert_almost_equal(result, expected)