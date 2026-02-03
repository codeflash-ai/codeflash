"""Tests for Jest timing marker matching with function name fallback."""

from __future__ import annotations

import re


def test_marker_matching_with_function_name_fallback():
    """Test that markers can be matched by function name when test name is 'unknown'."""
    # Simulate the patterns from parse_test_output.py
    jest_start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")

    # Test case: marker has "unknown" as test name, but function name matches
    marker = "!$######unknown:unknown:testFunction:1:42_0######$!"
    match = jest_start_pattern.search(marker)

    assert match is not None, "Pattern should match marker"
    assert match.group(2) == "unknown", "Test name (group 2) should be 'unknown'"
    assert match.group(3) == "testFunction", "Function name (group 3) should be 'testFunction'"

    # Simulate the matching logic with function name fallback
    test_name = "testFunction"
    sanitized_test_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_name)
    test_func_name = test_name.split(" > ")[0] if " > " in test_name else test_name
    sanitized_func_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_func_name)

    # Original matching (would fail with "unknown" test name)
    matches_by_test_name = sanitized_test_name in match.group(2)
    assert not matches_by_test_name, "Should NOT match by test name (it's 'unknown')"

    # Fallback matching by function name (should succeed)
    matches_by_func_name = sanitized_func_name == match.group(3)
    assert matches_by_func_name, "Should match by function name"

    # Combined matching (what the fix implements)
    combined_match = sanitized_test_name in match.group(2) or sanitized_func_name == match.group(3)
    assert combined_match, "Combined matching should succeed via function name fallback"


def test_marker_matching_with_normal_test_name():
    """Test that markers still match when test name is available (not 'unknown')."""
    jest_start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")

    # Test case: marker has actual test name
    marker = "!$######myModule:test_fibonacci:fibonacci:1:42_0######$!"
    match = jest_start_pattern.search(marker)

    assert match is not None
    assert match.group(2) == "test_fibonacci"
    assert match.group(3) == "fibonacci"

    # Matching by test name should work
    test_name = "test_fibonacci"
    sanitized_test_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_name)
    matches_by_test_name = sanitized_test_name in match.group(2)
    assert matches_by_test_name, "Should match by test name"


def test_marker_matching_with_vitest_test_description():
    """Test matching with Vitest-style 'test > description' format."""
    jest_start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")

    # The marker will have sanitized test name (spaces and > replaced)
    marker = "!$######module:test___describe:myFunc:1:42_0######$!"
    match = jest_start_pattern.search(marker)

    assert match is not None

    # Original test name format: "test > describe"
    test_name = "test > describe"
    sanitized_test_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_name)

    # Extract function name for fallback
    test_func_name = test_name.split(" > ")[0] if " > " in test_name else test_name
    assert test_func_name == "test"

    sanitized_func_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_func_name)
    assert sanitized_func_name == "test"


def test_end_marker_matching_with_function_name_fallback():
    """Test that END markers can also be matched by function name."""
    jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")

    marker = "!######unknown:unknown:testFunction:1:42_0:12345######!"
    match = jest_end_pattern.search(marker)

    assert match is not None
    groups = match.groups()
    # groups: (module, testName, funcName, loopIndex, invocationId, durationNs)
    assert groups[1] == "unknown", "Test name should be 'unknown'"
    assert groups[2] == "testFunction", "Function name should match"
    assert groups[5] == "12345", "Duration should be captured"

    # Simulate end_key tuple used in parse_test_output.py
    end_key = groups[:5]

    # Test matching logic
    test_name = "testFunction"
    sanitized_test_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_name)
    test_func_name = test_name.split(" > ")[0] if " > " in test_name else test_name
    sanitized_func_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_func_name)

    # Combined matching (matches by function name since test name is 'unknown')
    matches = len(end_key) >= 3 and (
        sanitized_test_name in end_key[1] or sanitized_func_name == end_key[2]
    )
    assert matches, "Should match by function name fallback"


if __name__ == "__main__":
    test_marker_matching_with_function_name_fallback()
    test_marker_matching_with_normal_test_name()
    test_marker_matching_with_vitest_test_description()
    test_end_marker_matching_with_function_name_fallback()
    print("All tests passed!")
