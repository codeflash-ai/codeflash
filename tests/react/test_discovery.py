"""Tests for React component discovery via tree-sitter."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.javascript.frameworks.react.discovery import (
    ComponentType,
    ReactComponentInfo,
    _extract_hooks_used,
    _has_server_directive,
    find_react_components,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def analyzer():
    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

    return TreeSitterAnalyzer("tsx")


class TestHasServerDirective:
    def test_use_server_double_quotes(self):
        assert _has_server_directive('"use server";\n\nexport function foo() {}') is True

    def test_use_server_single_quotes(self):
        assert _has_server_directive("'use server';\n\nexport function foo() {}") is True

    def test_no_directive(self):
        assert _has_server_directive("import React from 'react';\n\nexport function Foo() {}") is False

    def test_server_directive_not_at_top(self):
        assert _has_server_directive('import React from "react";\n"use server";\n') is False

    def test_use_client_is_not_server(self):
        assert _has_server_directive('"use client";\n\nexport function Foo() {}') is False

    def test_comment_before_directive(self):
        assert _has_server_directive('// some comment\n"use server";\n') is True


class TestExtractHooksUsed:
    def test_builtin_hooks(self):
        source = """
        const [count, setCount] = useState(0);
        useEffect(() => {}, [count]);
        const ref = useRef(null);
        """
        hooks = _extract_hooks_used(source)
        assert hooks == ["useState", "useEffect", "useRef"]

    def test_custom_hooks(self):
        source = "const value = useDebounce(input, 300);"
        hooks = _extract_hooks_used(source)
        assert hooks == ["useDebounce"]

    def test_no_hooks(self):
        source = "const x = 1; const y = 2;"
        hooks = _extract_hooks_used(source)
        assert hooks == []

    def test_no_duplicates(self):
        source = """
        useState(0);
        useState(1);
        """
        hooks = _extract_hooks_used(source)
        assert hooks == ["useState"]


class TestFindReactComponents:
    def test_counter_component(self, analyzer):
        source = FIXTURES_DIR.joinpath("Counter.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "Counter.tsx", analyzer)
        assert len(components) == 1
        comp = components[0]
        assert comp.function_name == "Counter"
        assert comp.component_type == ComponentType.FUNCTION
        assert comp.returns_jsx is True
        assert "useState" in comp.uses_hooks

    def test_hook_detected_as_hook(self, analyzer):
        source = FIXTURES_DIR.joinpath("useDebounce.ts").read_text(encoding="utf-8")
        # Need to use .ts analyzer for .ts file
        ts_analyzer = type(analyzer)("typescript")
        components = find_react_components(source, FIXTURES_DIR / "useDebounce.ts", ts_analyzer)
        assert len(components) == 1
        comp = components[0]
        assert comp.function_name == "useDebounce"
        assert comp.component_type == ComponentType.HOOK
        assert "useState" in comp.uses_hooks
        assert "useEffect" in comp.uses_hooks

    def test_server_component_skipped(self, analyzer):
        source = FIXTURES_DIR.joinpath("ServerComponent.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "ServerComponent.tsx", analyzer)
        assert components == []

    def test_memoized_component_detected(self, analyzer):
        source = FIXTURES_DIR.joinpath("MemoizedList.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "MemoizedList.tsx", analyzer)
        memoized_names = [c.function_name for c in components if c.is_memoized]
        assert len(memoized_names) > 0

    def test_task_list_component(self, analyzer):
        source = FIXTURES_DIR.joinpath("TaskList.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "TaskList.tsx", analyzer)
        names = [c.function_name for c in components]
        assert "TaskList" in names

    def test_data_table_component(self, analyzer):
        source = FIXTURES_DIR.joinpath("DataTable.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "DataTable.tsx", analyzer)
        assert len(components) >= 1
        comp = [c for c in components if c.function_name == "DataTable"][0]
        assert comp.returns_jsx is True

    def test_non_component_functions_excluded(self, analyzer):
        source = """
        function helperFunction() { return 42; }
        function anotherHelper(x: number) { return x * 2; }
        export function MyComponent() { return <div>hello</div>; }
        """
        components = find_react_components(source, Path("test.tsx"), analyzer)
        names = [c.function_name for c in components]
        assert "helperFunction" not in names
        assert "anotherHelper" not in names
        assert "MyComponent" in names

    def test_pascal_case_non_jsx_excluded(self, analyzer):
        source = """
        function MyUtil() { return 42; }
        function MyComponent() { return <div />; }
        """
        components = find_react_components(source, Path("test.tsx"), analyzer)
        names = [c.function_name for c in components]
        assert "MyUtil" not in names
        assert "MyComponent" in names
