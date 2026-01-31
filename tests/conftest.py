"""Pytest configuration and fixtures for codeflash tests."""

import pytest

from codeflash.languages import reset_current_language

# Test modules that modify files in code_to_optimize/ must run in the same xdist worker
# to avoid race conditions when running tests in parallel
SHARED_FILE_TEST_MODULES = {
    "test_comparator",
    "test_instrumentation_run_results_aiservice",
    "test_instrument_all_and_run",
    "test_async_run_and_parse_tests",
    "test_instrument_tests",
}


def pytest_collection_modifyitems(items):
    """Mark tests that modify shared files to run in the same xdist worker."""
    for item in items:
        module_name = item.module.__name__.split(".")[-1]
        if module_name in SHARED_FILE_TEST_MODULES:
            item.add_marker(pytest.mark.xdist_group("code_to_optimize_files"))


@pytest.fixture(autouse=True)
def set_python_language():
    """Ensure the current language is set to Python for all tests.

    This fixture runs automatically before each test to ensure a clean language state.
    """
    reset_current_language()
    yield
    # Reset again after test to clean up any changes
    reset_current_language()
