import warnings

import pytest

from swarmcg.shared.utils import catch_warnings, parse_string_args


def test_cath_warnings():
    # given:
    def func():
        warnings.warn("This is a warning")
        return 1

    # when:
    with warnings.catch_warnings(record=True) as w:
        _ = func()

    # then:
    assert len(w) == 1

    # when:
    with warnings.catch_warnings(record=True) as w:
        # this is equivalent to add @decorator above function definition
        _ = catch_warnings(UserWarning)(func)()

    # then:
    assert len(w) == 0


@pytest.mark.parametrize("input, output", [("1.1", 1.1), ("1", 1), ("123niu", "123niu")])
def test_parse_string_args(input, output):
    assert output == parse_string_args(input)

