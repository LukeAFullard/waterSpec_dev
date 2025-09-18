import pytest
from waterSpec.interpreter import interpret_beta

def test_interpret_beta_strong_persistence():
    """Test interpretation for high beta values (strong persistence)."""
    interpretation = interpret_beta(1.7)
    assert "strong persistence" in interpretation.lower()
    assert "subsurface" in interpretation.lower()

def test_interpret_beta_weak_persistence():
    """Test interpretation for medium beta values (weak persistence)."""
    interpretation = interpret_beta(0.8)
    assert "weak persistence" in interpretation.lower()
    assert "mixed" in interpretation.lower()

def test_interpret_beta_event_driven():
    """Test interpretation for low beta values (event-driven)."""
    interpretation = interpret_beta(0.3)
    assert "event-driven" in interpretation.lower()

def test_interpret_beta_white_noise():
    """Test interpretation for beta values around 0 (white noise)."""
    interpretation = interpret_beta(0.0)
    assert "white noise" in interpretation.lower()

def test_interpret_beta_brownian_motion():
    """Test interpretation for beta values around 2 (Brownian motion)."""
    interpretation = interpret_beta(2.0)
    assert "brownian motion" in interpretation.lower()

def test_interpret_beta_invalid_input():
    """Test that the function handles invalid (negative) beta values."""
    with pytest.raises(ValueError, match="Beta value must be non-negative."):
        interpret_beta(-0.5)
