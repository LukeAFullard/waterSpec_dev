import numpy as np
import pytest

from waterSpec.changepoint_detector import get_changepoint_time


def test_get_changepoint_time_seconds():
    """Test getting changepoint time with input unit as seconds."""
    time = np.array([0, 86400 * 1, 86400 * 5, 86400 * 65, 86400 * 800])

    # 1 day -> hours (< 2 days)
    assert get_changepoint_time(1, time, "seconds") == "24.0 hours"

    # 5 days -> days (< 60 days)
    assert get_changepoint_time(2, time, "seconds") == "5.0 days"

    # 65 days -> months (< 730 days)
    assert get_changepoint_time(3, time, "seconds") == f"{65 / 30.44:.1f} months"

    # 800 days -> years (>= 730 days)
    assert get_changepoint_time(4, time, "seconds") == f"{800 / 365.25:.1f} years"


def test_get_changepoint_time_hours():
    """Test getting changepoint time with input unit as hours."""
    time = np.array([0, 24 * 1, 24 * 5, 24 * 65, 24 * 800])

    # 1 day -> hours (< 2 days)
    assert get_changepoint_time(1, time, "hours") == "24.0 hours"

    # 5 days -> days (< 60 days)
    assert get_changepoint_time(2, time, "hours") == "5.0 days"

    # 65 days -> months (< 730 days)
    assert get_changepoint_time(3, time, "hours") == f"{65 / 30.44:.1f} months"

    # 800 days -> years (>= 730 days)
    assert get_changepoint_time(4, time, "hours") == f"{800 / 365.25:.1f} years"


def test_get_changepoint_time_days():
    """Test getting changepoint time with input unit as days."""
    time = np.array([0, 1, 5, 65, 800])

    # 1 day -> hours (< 2 days)
    assert get_changepoint_time(1, time, "days") == "24.0 hours"

    # 5 days -> days (< 60 days)
    assert get_changepoint_time(2, time, "days") == "5.0 days"

    # 65 days -> months (< 730 days)
    assert get_changepoint_time(3, time, "days") == f"{65 / 30.44:.1f} months"

    # 800 days -> years (>= 730 days)
    assert get_changepoint_time(4, time, "days") == f"{800 / 365.25:.1f} years"


def test_get_changepoint_time_edge_cases():
    """Test edge case conditions."""
    time = np.array([0, 1.99, 59.9, 729.9])

    # Edge case: slightly less than 2 days
    assert get_changepoint_time(1, time, "days") == f"{1.99 * 24:.1f} hours"

    # Edge case: slightly less than 60 days
    assert get_changepoint_time(2, time, "days") == "59.9 days"

    # Edge case: slightly less than 730 days
    assert get_changepoint_time(3, time, "days") == f"{729.9 / 30.44:.1f} months"
