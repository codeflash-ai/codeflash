import os
import tempfile
import pytest
from code_to_optimize.detect_test_framework import (
    detect_test_framework,
)  # Adjust import based on your module setup


def setup_pytest_config(directory, filename, contents):
    config_path = os.path.join(directory, filename)
    with open(config_path, "w") as f:
        f.write(contents)
    return config_path


def setup_python_test_file(directory, filename, contents):
    test_path = os.path.join(directory, filename)
    with open(test_path, "w") as f:
        f.write(contents)
    return test_path


def test_pytest_detected_from_config_files():
    with tempfile.TemporaryDirectory() as project_dir:
        setup_pytest_config(project_dir, "pytest.ini", "[pytest]")
        tests_dir = tempfile.mkdtemp(dir=project_dir)
        assert detect_test_framework(project_dir, tests_dir) == "pytest"


def test_mixed_signals_with_pytest_priority():
    with tempfile.TemporaryDirectory() as project_dir:
        setup_pytest_config(project_dir, "pytest.ini", "[pytest]")
        tests_dir = tempfile.mkdtemp(dir=project_dir)
        setup_python_test_file(
            tests_dir,
            "test_mytest.py",
            """
        import unittest

        class MyTest(unittest.TestCase):
            def test_something(self):
                pass
        """,
        )
        assert detect_test_framework(project_dir, tests_dir) == "pytest"
