"""Singleton for the current test framework being used in the codeflash session.

This module provides a centralized way to access and set the current test framework
throughout the codeflash codebase, similar to how the language singleton works.

For JavaScript/TypeScript projects, this determines whether to use Jest or Vitest
for test execution.

Usage:
    from codeflash.languages.test_framework import (
        current_test_framework,
        set_current_test_framework,
        is_jest,
        is_vitest,
    )

    # Set the test framework at the start of a session (auto-detected from package.json)
    set_current_test_framework("vitest")

    # Check the current test framework anywhere in the codebase
    if is_vitest():
        # Vitest-specific code
        ...

    # Get the current test framework
    framework = current_test_framework()
"""

from __future__ import annotations

from typing import Literal

TestFramework = Literal["jest", "vitest", "mocha", "pytest", "unittest"]

# Module-level singleton for the current test framework
_current_test_framework: TestFramework | None = None


def current_test_framework() -> TestFramework | None:
    """Get the current test framework being used in this codeflash session.

    Returns:
        The current test framework string, or None if not set.

    """
    return _current_test_framework


def set_current_test_framework(framework: TestFramework | str | None) -> None:
    """Set the current test framework for this codeflash session.

    This should be called once at the start of an optimization run,
    typically after reading the project configuration.

    Args:
        framework: Test framework name ("jest", "vitest", "mocha", "pytest", "unittest").

    """
    global _current_test_framework

    if _current_test_framework is not None:
        return

    if framework is not None:
        framework = framework.lower()
        if framework not in ("jest", "vitest", "mocha", "pytest", "unittest"):
            # Default to jest for unknown JS frameworks, pytest for unknown Python
            from codeflash.languages.current import is_javascript

            framework = "jest" if is_javascript() else "pytest"

    _current_test_framework = framework


def reset_test_framework() -> None:
    """Reset the current test framework to None.

    Useful for testing or when starting a new session.
    """
    global _current_test_framework
    _current_test_framework = None


def is_jest() -> bool:
    """Check if the current test framework is Jest.

    Returns:
        True if the current test framework is Jest.

    """
    return _current_test_framework == "jest"


def is_vitest() -> bool:
    """Check if the current test framework is Vitest.

    Returns:
        True if the current test framework is Vitest.

    """
    return _current_test_framework == "vitest"


def is_mocha() -> bool:
    """Check if the current test framework is Mocha.

    Returns:
        True if the current test framework is Mocha.

    """
    return _current_test_framework == "mocha"


def is_pytest() -> bool:
    """Check if the current test framework is pytest.

    Returns:
        True if the current test framework is pytest.

    """
    return _current_test_framework == "pytest"


def is_unittest() -> bool:
    """Check if the current test framework is unittest.

    Returns:
        True if the current test framework is unittest.

    """
    return _current_test_framework == "unittest"


def get_js_test_framework_or_default() -> TestFramework:
    """Get the current test framework for JS/TS, defaulting to 'jest' if not set.

    This is a convenience function for JS/TS code that needs a framework.

    Returns:
        The current test framework, or 'jest' as default.

    """
    if _current_test_framework in ("jest", "vitest", "mocha"):
        return _current_test_framework
    return "jest"
