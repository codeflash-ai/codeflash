"""Tests for React test generation helpers."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.discovery import (
    ComponentType,
    ReactComponentInfo,
)
from codeflash.languages.javascript.frameworks.react.testgen import (
    generate_rerender_test_template,
    post_process_react_tests,
)


class TestGenerateRerenderTestTemplate:
    def test_basic_template(self):
        template = generate_rerender_test_template("Counter")
        assert "Counter" in template
        assert "render" in template
        assert "rerender" in template
        assert "REACT_RENDER" in template

    def test_includes_import(self):
        template = generate_rerender_test_template("MyComponent")
        assert "@testing-library/react" in template

    def test_includes_render_count_check(self):
        template = generate_rerender_test_template("MyComponent")
        assert "renderCount" in template
        assert "expect(renderCount).toBe(1)" in template

    def test_props_interface_hint(self):
        template = generate_rerender_test_template("MyComponent", "MyComponentProps")
        assert "fill in props matching interface" in template


class TestPostProcessReactTests:
    def test_adds_testing_library_import(self):
        source = "describe('MyComp', () => { render(<MyComp />); });"
        result = post_process_react_tests(source, _make_info())
        assert "@testing-library/react" in result

    def test_skips_if_already_imported(self):
        source = "import { render } from '@testing-library/react';\ndescribe('MyComp', () => {});"
        result = post_process_react_tests(source, _make_info())
        assert result.count("@testing-library/react") == 1

    def test_adds_user_event_for_click(self):
        source = "import { render } from '@testing-library/react';\ntest('clicks button', () => { click(button); });"
        result = post_process_react_tests(source, _make_info())
        assert "@testing-library/user-event" in result

    def test_no_user_event_if_no_interaction(self):
        source = "import { render } from '@testing-library/react';\ntest('renders', () => { render(<Foo />); });"
        result = post_process_react_tests(source, _make_info())
        assert "@testing-library/user-event" not in result


def _make_info() -> ReactComponentInfo:
    return ReactComponentInfo(
        function_name="MyComp",
        component_type=ComponentType.FUNCTION,
        returns_jsx=True,
    )
