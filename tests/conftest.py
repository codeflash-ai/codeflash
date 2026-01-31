"""Pytest configuration and fixtures for codeflash tests."""

import pytest

from codeflash.languages import reset_current_language


@pytest.fixture(autouse=True)
def set_python_language():
    """Ensure the current language is set to Python for all tests.

    This fixture runs automatically before each test to ensure a clean language state.
    """
    reset_current_language()
    yield
    # Reset again after test to clean up any changes
    reset_current_language()
