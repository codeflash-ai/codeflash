from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class TestInfo:
    """Information about a test that exercises a function.

    Attributes:
        test_name: Name of the test function.
        test_file: Path to the test file.
        test_class: Name of the test class, if any.

    """

    test_name: str
    test_file: Path
    test_class: str | None = None

    @property
    def full_test_path(self) -> str:
        """Get full test path in pytest format (file::class::function)."""
        file_path = self.test_file.as_posix()
        if self.test_class:
            return f"{file_path}::{self.test_class}::{self.test_name}"
        return f"{file_path}::{self.test_name}"


@dataclass
class TestResult:
    """Language-agnostic test result.

    Captures the outcome of running a single test, including timing
    and behavioral data for equivalence checking.

    Attributes:
        test_name: Name of the test function.
        test_file: Path to the test file.
        passed: Whether the test passed.
        runtime_ns: Execution time in nanoseconds.
        return_value: The return value captured from the test.
        stdout: Standard output captured during test execution.
        stderr: Standard error captured during test execution.
        error_message: Error message if the test failed.

    """

    test_name: str
    test_file: Path
    passed: bool
    runtime_ns: int | None = None
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error_message: str | None = None
