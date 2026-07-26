from datetime import datetime

from main import DailyCap


def test_not_exceeded_before_any_calls():
    cap = DailyCap(limit=2)
    assert cap.exceeded is False


def test_exceeded_once_limit_is_reached():
    cap = DailyCap(limit=2)
    cap.record()
    cap.record()
    assert cap.exceeded is True


def test_not_exceeded_one_below_limit():
    cap = DailyCap(limit=2)
    cap.record()
    assert cap.exceeded is False


def test_resets_when_the_day_rolls_over():
    day_one = [datetime(2026, 7, 22, 23, 59)]
    cap = DailyCap(limit=1, now_fn=lambda: day_one[0])
    cap.record()
    assert cap.exceeded is True

    day_one[0] = datetime(2026, 7, 23, 0, 1)  # next day
    assert cap.exceeded is False, "should reset once the calendar date changes"
