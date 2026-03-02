"""Tests for React Profiler instrumentation and marker parsing."""

from __future__ import annotations

import pytest

from codeflash.languages.javascript.frameworks.react.profiler import (
    MARKER_PREFIX,
    generate_render_counter_code,
)
from codeflash.languages.javascript.parse import (
    REACT_RENDER_MARKER_PATTERN,
    RenderProfile,
    parse_react_render_markers,
)


class TestGenerateRenderCounterCode:
    def test_generates_counter_variable(self):
        code = generate_render_counter_code("MyComponent")
        assert "_codeflash_render_count_MyComponent" in code
        assert "let " in code

    def test_generates_on_render_callback(self):
        code = generate_render_counter_code("MyComponent")
        assert "_codeflashOnRender_MyComponent" in code
        assert "function " in code

    def test_marker_format_in_output(self):
        code = generate_render_counter_code("Counter")
        assert MARKER_PREFIX in code
        assert "console.log" in code

    def test_special_characters_sanitized(self):
        code = generate_render_counter_code("My-Component.Inner")
        assert "_codeflash_render_count_My_Component_Inner" in code
        assert "_codeflashOnRender_My_Component_Inner" in code


class TestReactRenderMarkerPattern:
    def test_matches_valid_marker(self):
        marker = "!######REACT_RENDER:Counter:mount:12.5:15.3:1######!"
        match = REACT_RENDER_MARKER_PATTERN.search(marker)
        assert match is not None
        assert match.group(1) == "Counter"
        assert match.group(2) == "mount"
        assert match.group(3) == "12.5"
        assert match.group(4) == "15.3"
        assert match.group(5) == "1"

    def test_matches_update_phase(self):
        marker = "!######REACT_RENDER:TaskList:update:3.2:8.1:5######!"
        match = REACT_RENDER_MARKER_PATTERN.search(marker)
        assert match is not None
        assert match.group(2) == "update"

    def test_no_match_for_invalid_marker(self):
        assert REACT_RENDER_MARKER_PATTERN.search("not a marker") is None
        assert REACT_RENDER_MARKER_PATTERN.search("!######OTHER:foo:bar######!") is None


class TestParseReactRenderMarkers:
    def test_single_marker(self):
        stdout = "Some output\n!######REACT_RENDER:Counter:mount:12.5:15.3:1######!\nMore output"
        profiles = parse_react_render_markers(stdout)
        assert len(profiles) == 1
        assert profiles[0] == RenderProfile(
            component_name="Counter",
            phase="mount",
            actual_duration_ms=12.5,
            base_duration_ms=15.3,
            render_count=1,
        )

    def test_multiple_markers(self):
        stdout = (
            "!######REACT_RENDER:Counter:mount:12.5:15.3:1######!\n"
            "!######REACT_RENDER:Counter:update:3.2:15.3:2######!\n"
            "!######REACT_RENDER:Counter:update:2.8:15.3:3######!\n"
        )
        profiles = parse_react_render_markers(stdout)
        assert len(profiles) == 3
        assert profiles[0].phase == "mount"
        assert profiles[1].phase == "update"
        assert profiles[2].render_count == 3

    def test_no_markers(self):
        stdout = "Test passed!\nAll good."
        profiles = parse_react_render_markers(stdout)
        assert profiles == []

    def test_marker_with_zero_duration(self):
        stdout = "!######REACT_RENDER:Empty:mount:0:0:1######!"
        profiles = parse_react_render_markers(stdout)
        assert len(profiles) == 1
        assert profiles[0].actual_duration_ms == 0.0

    def test_mixed_output(self):
        stdout = (
            "PASS src/Counter.test.tsx\n"
            "  Counter\n"
            "    ✓ renders correctly (23ms)\n"
            "!######REACT_RENDER:Counter:mount:5.4:8.1:1######!\n"
            "    ✓ increments on click (15ms)\n"
            "!######REACT_RENDER:Counter:update:2.1:8.1:2######!\n"
            "Test Suites: 1 passed\n"
        )
        profiles = parse_react_render_markers(stdout)
        assert len(profiles) == 2
        assert profiles[0].component_name == "Counter"
        assert profiles[1].render_count == 2
