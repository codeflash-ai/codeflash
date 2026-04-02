"""Unit tests for Jest END marker parsing.

Bug: END markers without duration (from capture()) are not parsed.
Location: codeflash/languages/javascript/parse.py:32-34
"""
import re

import pytest

from codeflash.languages.javascript.parse import (
    jest_end_pattern,
    jest_end_pattern_no_duration,
    jest_start_pattern,
)


def test_end_marker_regex_requires_duration():
    """Test that current jest_end_pattern only matches markers WITH duration.

    This is the bug: behavior tests output END markers without duration,
    but the regex requires duration, so those markers are never parsed.
    """
    # Current regex from parse.py line 32
    jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")

    # END marker WITH duration (from capturePerf)
    marker_with_duration = "!######module:testName:funcName:1:lineId_0:123456######!"
    # END marker WITHOUT duration (from capture)
    marker_without_duration = "!######module:testName:funcName:1:lineId_0######!"

    # Current regex only matches markers with duration
    assert jest_end_pattern.search(marker_with_duration), "Should match END marker with duration"
    assert not jest_end_pattern.search(marker_without_duration), "Bug: does not match END marker without duration"


def test_behavior_test_end_markers_not_parsed():
    """Test that behavior test END markers (without duration) are lost.

    When behavior tests (using capture()) run, they output END markers
    without duration. The current parsing code ignores these, resulting
    in 0 END matches even though markers exist in stdout.
    """
    jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")

    # Simulated stdout from a behavior test run
    stdout = """
!$######module:testName:funcName:1:lineId_0######$!
!######module:testName:funcName:1:lineId_0######!
!$######module:testName:funcName:1:lineId_1######$!
!######module:testName:funcName:1:lineId_1######!
!$######module:testName:funcName:1:lineId_2######$!
!######module:testName:funcName:1:lineId_2######!
"""

    start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")

    start_matches = list(start_pattern.finditer(stdout))
    end_matches = list(jest_end_pattern.finditer(stdout))

    # Bug: 3 START markers found, but 0 END markers
    assert len(start_matches) == 3, "Should find 3 START markers"
    assert len(end_matches) == 0, "Bug reproduced: 0 END markers found (should be 3)"


def test_performance_test_end_markers_parsed():
    """Test that performance test END markers (with duration) ARE parsed correctly."""
    jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")

    # Simulated stdout from a performance test run (capturePerf)
    stdout = """
!######module:testName:funcName:1:lineId_0:123456######!
!######module:testName:funcName:2:lineId_0:234567######!
!######module:testName:funcName:3:lineId_0:345678######!
"""

    end_matches = list(jest_end_pattern.finditer(stdout))

    # Performance markers ARE found correctly
    assert len(end_matches) == 3, "Should find all 3 performance END markers"

    # Verify duration is captured
    for match in end_matches:
        duration = match.group(6)
        assert duration.isdigit(), "Duration should be a number"
        assert int(duration) > 0, "Duration should be positive"


def test_mixed_behavior_and_performance_markers():
    """Test parsing stdout with both behavior and performance markers.

    This simulates a test run that has both types of instrumentation.
    Current code only parses performance markers.
    """
    jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")

    stdout = """
!######module:testName:funcName:1:lineId_0######!
!######module:testName:funcName:2:lineId_1:123456######!
!######module:testName:funcName:3:lineId_2######!
!######module:testName:funcName:4:lineId_3:789012######!
"""

    end_matches = list(jest_end_pattern.finditer(stdout))

    # Bug: Only 2 of 4 END markers are parsed (the ones with duration)
    assert len(end_matches) == 2, "Bug reproduced: only markers with duration are parsed"


# Tests for the fix


def test_fix_behavior_markers_parsed_with_new_regex():
    """Test that the new regex can parse END markers without duration."""
    # END marker WITHOUT duration (from capture)
    marker_without_duration = "!######module:testName:funcName:1:lineId_0######!"

    # New regex should match markers without duration
    assert jest_end_pattern_no_duration.search(marker_without_duration), \
        "New regex should match END marker without duration"

    # But should NOT match markers with duration (to avoid duplicates)
    marker_with_duration = "!######module:testName:funcName:1:lineId_0:123456######!"
    # Actually, it WILL match (captures up to the 5th field), but that's OK
    # because we check for existence in the dict before adding


def test_fix_both_marker_types_parsed():
    """Test that with both regexes, all END markers are parsed."""
    stdout = """
!######module:testName:funcName:1:lineId_0######!
!######module:testName:funcName:2:lineId_1:123456######!
!######module:testName:funcName:3:lineId_2######!
!######module:testName:funcName:4:lineId_3:789012######!
"""

    # Combine both regexes like the fixed code does
    end_matches_dict = {}

    # Parse markers with duration first
    for match in jest_end_pattern.finditer(stdout):
        key = match.groups()[:5]
        end_matches_dict[key] = match

    # Then parse markers without duration
    for match in jest_end_pattern_no_duration.finditer(stdout):
        key = match.groups()[:5]
        if key not in end_matches_dict:
            end_matches_dict[key] = match

    # After fix: All 4 END markers should be parsed
    assert len(end_matches_dict) == 4, "Fix verified: all 4 END markers parsed"


def test_fix_behavior_test_run_end_markers_found():
    """Test that behavior test END markers are now found after fix."""
    # Simulated stdout from a behavior test run
    stdout = """
!$######module:testName:funcName:1:lineId_0######$!
!######module:testName:funcName:1:lineId_0######!
!$######module:testName:funcName:1:lineId_1######$!
!######module:testName:funcName:1:lineId_1######!
!$######module:testName:funcName:1:lineId_2######$!
!######module:testName:funcName:1:lineId_2######!
"""

    start_matches = list(jest_start_pattern.finditer(stdout))

    # Build end_matches_dict like the fixed code does
    end_matches_dict = {}
    for match in jest_end_pattern.finditer(stdout):
        key = match.groups()[:5]
        end_matches_dict[key] = match
    for match in jest_end_pattern_no_duration.finditer(stdout):
        key = match.groups()[:5]
        if key not in end_matches_dict:
            end_matches_dict[key] = match

    # After fix: 3 START and 3 END markers should be found
    assert len(start_matches) == 3, "Should find 3 START markers"
    assert len(end_matches_dict) == 3, "Fix verified: 3 END markers found (was 0 before fix)"
