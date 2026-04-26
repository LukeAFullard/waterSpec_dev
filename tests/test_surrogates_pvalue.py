import numpy as np
import pytest
from waterSpec.surrogates import calculate_significance_p_value

def test_significance_p_value_empty():
    """Test that empty surrogate metrics return NaN."""
    p_val = calculate_significance_p_value(5.0, np.array([]))
    assert np.isnan(p_val)

def test_significance_p_value_two_sided():
    """Test two-sided p-value calculation (checks absolute magnitude)."""
    # Absolute values of surrogates: [1, 2, 3, 4]
    surrogates = np.array([-1, 2, -3, 4])

    # Obs = 5.0 (abs = 5.0). No surrogates >= 5.0. count = 0. p = 1 / 5 = 0.2
    assert calculate_significance_p_value(5.0, surrogates, two_sided=True) == 0.2

    # Obs = -4.0 (abs = 4.0). One surrogate (4) >= 4.0. count = 1. p = 2 / 5 = 0.4
    assert calculate_significance_p_value(-4.0, surrogates, two_sided=True) == 0.4

    # Obs = 0.0 (abs = 0.0). All four surrogates >= 0.0. count = 4. p = 5 / 5 = 1.0
    assert calculate_significance_p_value(0.0, surrogates, two_sided=True) == 1.0

def test_significance_p_value_one_sided():
    """Test one-sided p-value calculation (checks raw magnitude)."""
    surrogates = np.array([1, 2, 3, 4])

    # Obs = 5.0. No surrogates >= 5.0. count = 0. p = 1 / 5 = 0.2
    assert calculate_significance_p_value(5.0, surrogates, two_sided=False) == 0.2

    # Obs = 3.0. Surrogates >= 3.0 are [3, 4]. count = 2. p = 3 / 5 = 0.6
    assert calculate_significance_p_value(3.0, surrogates, two_sided=False) == 0.6

    # Obs = 0.0. All four surrogates >= 0.0. count = 4. p = 5 / 5 = 1.0
    assert calculate_significance_p_value(0.0, surrogates, two_sided=False) == 1.0

def test_significance_p_value_one_sided_negative():
    """Test one-sided p-value with negative numbers."""
    surrogates = np.array([-4, -3, -2, -1])

    # Obs = -2.0. Surrogates >= -2.0 are [-2, -1]. count = 2. p = 3 / 5 = 0.6
    assert calculate_significance_p_value(-2.0, surrogates, two_sided=False) == 0.6

    # Obs = 0.0. No surrogates >= 0.0. count = 0. p = 1 / 5 = 0.2
    assert calculate_significance_p_value(0.0, surrogates, two_sided=False) == 0.2

    # Obs = -5.0. All four surrogates >= -5.0. count = 4. p = 5 / 5 = 1.0
    assert calculate_significance_p_value(-5.0, surrogates, two_sided=False) == 1.0

def test_significance_p_value_exact_match():
    """Test exactly matching values."""
    surrogates = np.array([5.0, 5.0, 5.0])

    # Obs = 5.0. All three >= 5.0. count = 3. p = 4 / 4 = 1.0
    assert calculate_significance_p_value(5.0, surrogates, two_sided=True) == 1.0
    assert calculate_significance_p_value(5.0, surrogates, two_sided=False) == 1.0
