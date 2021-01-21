import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from swarmcg.shared import math_utils
from swarmcg.shared.exceptions import OptimisationResultsError


# TODO: add test on failure but needs dedicated exceptions rather and sys.exit
def test_forward_fill():
    # given:
    x = [1, 2, 10, 4, None, None, 10, 10]

    # when:
    cond_value = None
    result = math_utils.forward_fill(x, cond_value)

    # then:
    expected = [1, 2, 10, 4, 4, 4, 10, 10]
    assert result == expected


def test_forward_fill_fail():
    # given:
    x = [None, None]

    # when:
    cond_value = None
    with pytest.raises(OptimisationResultsError):
        _ = math_utils.forward_fill(x, cond_value)


def test_sma():
    # given:
    x = np.arange(10)

    # when:
    window_size = 5
    result = math_utils.sma(x, window_size)

    # then:
    expected = np.array([0.6, 1.2, 2., 3., 4., 5., 6., 7., 6., 4.8], dtype=float)
    assert_almost_equal(result, expected)


def test_ema():
    # given:
    x = np.arange(10)

    # when:
    window_size = 5
    result = math_utils.ewma(x, 1, window_size)

    # then:
    expected = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=float)
    assert_almost_equal(result, expected)

