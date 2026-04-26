import pytest
from waterSpec.utils import sanitize_filename

def test_sanitize_filename_basic():
    """Test basic filename sanitization."""
    assert sanitize_filename("test file") == "test_file"
    assert sanitize_filename("test-file") == "test-file"
    assert sanitize_filename("test.file") == "test.file"

def test_sanitize_filename_special_chars():
    """Test sanitization of special characters."""
    assert sanitize_filename("test@file!") == "testfile"
    assert sanitize_filename("test/file") == "testfile"
    assert sanitize_filename("test\\file") == "testfile"

def test_sanitize_filename_unicode():
    """Test sanitization of unicode characters."""
    # Assuming the regex (?u)[^-\w.] allows unicode characters that are considered word characters
    assert sanitize_filename("tést_file") == "tést_file"
    assert sanitize_filename("测试_file") == "测试_file"

def test_sanitize_filename_spaces():
    """Test handling of multiple spaces and leading/trailing spaces."""
    assert sanitize_filename("  test  file  ") == "test__file"

def test_sanitize_filename_non_string():
    """Test handling of non-string inputs."""
    assert sanitize_filename(123) == "123"
    assert sanitize_filename(12.34) == "12.34"
