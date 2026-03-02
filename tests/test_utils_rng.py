import numpy as np
import pytest

from waterSpec.utils import make_rng

def test_make_rng_with_none():
    """Test that passing None creates a new default_rng."""
    rng = make_rng(None)
    assert isinstance(rng, np.random.Generator)

def test_make_rng_with_int():
    """Test that passing an int creates a new default_rng with that seed."""
    rng1 = make_rng(42)
    rng2 = make_rng(42)
    assert isinstance(rng1, np.random.Generator)
    assert isinstance(rng2, np.random.Generator)
    # The two RNGs should produce the same random sequence
    assert rng1.random() == rng2.random()

def test_make_rng_with_seedsequence():
    """Test that passing a SeedSequence creates a new default_rng with that seed."""
    seq1 = np.random.SeedSequence(123)
    seq2 = np.random.SeedSequence(123)
    rng1 = make_rng(seq1)
    rng2 = make_rng(seq2)
    assert isinstance(rng1, np.random.Generator)
    assert isinstance(rng2, np.random.Generator)
    # The two RNGs should produce the same random sequence
    assert rng1.random() == rng2.random()

def test_make_rng_with_generator():
    """Test that passing a Generator returns the Generator itself."""
    original_rng = np.random.default_rng(99)
    returned_rng = make_rng(original_rng)
    assert isinstance(returned_rng, np.random.Generator)
    assert original_rng is returned_rng

def test_make_rng_with_invalid_type():
    """Test that passing an invalid type raises a TypeError."""
    with pytest.raises(TypeError, match="Invalid seed type: <class 'str'>"):
        make_rng("invalid_seed")

    with pytest.raises(TypeError, match="Invalid seed type: <class 'float'>"):
        make_rng(3.14)

    with pytest.raises(TypeError, match="Invalid seed type: <class 'list'>"):
        make_rng([1, 2, 3])
