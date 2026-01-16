"""JavaScript runtime files for codeflash test instrumentation.

This module provides paths to JavaScript files that are injected into
user projects during test instrumentation and execution.
"""

from pathlib import Path

# Directory containing the JavaScript runtime files
RUNTIME_DIR = Path(__file__).parent


def get_jest_helper_path() -> Path:
    """Get the path to the Jest helper file.

    This file provides capture/capturePerf/capturePerfLooped functions
    for instrumenting Jest tests to record function inputs, outputs, and timing.
    """
    return RUNTIME_DIR / "codeflash-jest-helper.js"


def get_comparator_path() -> Path:
    """Get the path to the comparator module.

    This file provides deep comparison logic for JavaScript values,
    handling special cases like NaN, Infinity, circular references, etc.
    """
    return RUNTIME_DIR / "codeflash-comparator.js"


def get_compare_results_path() -> Path:
    """Get the path to the compare-results script.

    This file provides the entry point for comparing test results
    between original and optimized code.
    """
    return RUNTIME_DIR / "codeflash-compare-results.js"


def get_serializer_path() -> Path:
    """Get the path to the serializer module.

    This file provides serialization utilities for JavaScript values,
    handling complex types that JSON.stringify cannot handle.
    """
    return RUNTIME_DIR / "codeflash-serializer.js"


def get_all_runtime_files() -> list[Path]:
    """Get paths to all JavaScript runtime files.

    Returns a list of all JS files that should be copied to the user's project.
    """
    return [
        get_jest_helper_path(),
        get_comparator_path(),
        get_compare_results_path(),
        get_serializer_path(),
    ]
