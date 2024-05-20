from code_to_optimize.standardize_name import standardize_name
import pytest


def test_exact_match():
    assert standardize_name("Brattle St") == "Brattle St"


def test_case_insensitivity():
    assert standardize_name("brattle st") == "Brattle St"
    assert standardize_name("MASSACHUSETTS AVE") == "Massachusetts Ave"


def test_handling_abbreviations():
    assert standardize_name("Beacon St.") == "Beacon St"
    assert standardize_name("Massachusetts Avenue") == "Massachusetts Ave"


def test_error_for_unknown_name():
    with pytest.raises(ValueError) as e:
        standardize_name("Infinite Loop")
    assert "Unknown street Infinite Loop" in str(e.value)
