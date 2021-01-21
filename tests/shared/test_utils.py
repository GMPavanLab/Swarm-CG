import warnings

from swarmcg.shared import catch_warnings


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
