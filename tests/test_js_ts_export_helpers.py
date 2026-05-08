"""Tests for JS/TS export helper functions with pre-read source content.

Verifies that _is_js_ts_function_exported and _is_js_ts_function_exists_but_not_exported
work correctly both with and without pre-read source content passed in.
"""

from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import (
    _is_js_ts_function_exists_but_not_exported,
    _is_js_ts_function_exported,
)


class TestIsJsTsFunctionExported:
    def test_named_export_detected(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        js_file.write_text(
            "export function add(a, b) {\n    return a + b;\n}\n",
            encoding="utf-8",
        )
        is_exported, export_name = _is_js_ts_function_exported(js_file, "add")
        assert is_exported is True
        assert export_name == "add"

    def test_named_export_with_source_param(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "export function add(a, b) {\n    return a + b;\n}\n"
        js_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(js_file, "add", source=content)
        assert is_exported is True
        assert export_name == "add"

    def test_default_export_detected(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "function compute(x) {\n    return x * 2;\n}\nexport default compute;\n"
        js_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(js_file, "compute", source=content)
        assert is_exported is True
        assert export_name == "default"

    def test_non_exported_function(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "function helper(x) {\n    return x + 1;\n}\n\nexport function main() {\n    return helper(5);\n}\n"
        js_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(js_file, "helper", source=content)
        assert is_exported is False
        assert export_name is None

    def test_separate_export_clause(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "function process(data) {\n    return data;\n}\n\nexport { process };\n"
        js_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(js_file, "process", source=content)
        assert is_exported is True
        assert export_name == "process"

    def test_aliased_export(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "function internalName(x) {\n    return x;\n}\n\nexport { internalName as publicName };\n"
        js_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(js_file, "internalName", source=content)
        assert is_exported is True
        assert export_name == "publicName"

    def test_typescript_export(self, tmp_path: Path) -> None:
        ts_file = tmp_path / "module.ts"
        content = "export function greet(name: string): string {\n    return `Hello, ${name}`;\n}\n"
        ts_file.write_text(content, encoding="utf-8")
        is_exported, export_name = _is_js_ts_function_exported(ts_file, "greet", source=content)
        assert is_exported is True
        assert export_name == "greet"

    def test_fallback_reads_from_disk_when_source_is_none(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        js_file.write_text(
            "export function fromDisk(x) {\n    return x;\n}\n",
            encoding="utf-8",
        )
        is_exported, export_name = _is_js_ts_function_exported(js_file, "fromDisk", source=None)
        assert is_exported is True
        assert export_name == "fromDisk"


class TestIsJsTsFunctionExistsButNotExported:
    def test_unexported_function_detected(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "function secret(x) {\n    return x * 2;\n}\n\nexport function pub() {\n    return 1;\n}\n"
        js_file.write_text(content, encoding="utf-8")
        assert _is_js_ts_function_exists_but_not_exported(js_file, "secret", source=content) is True

    def test_exported_function_returns_false(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "export function pub(x) {\n    return x;\n}\n"
        js_file.write_text(content, encoding="utf-8")
        assert _is_js_ts_function_exists_but_not_exported(js_file, "pub", source=content) is False

    def test_nonexistent_function_returns_false(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        content = "export function exists() {\n    return 1;\n}\n"
        js_file.write_text(content, encoding="utf-8")
        assert _is_js_ts_function_exists_but_not_exported(js_file, "nonexistent", source=content) is False

    def test_fallback_reads_from_disk(self, tmp_path: Path) -> None:
        js_file = tmp_path / "module.js"
        js_file.write_text(
            "function localOnly() {\n    return 42;\n}\n",
            encoding="utf-8",
        )
        assert _is_js_ts_function_exists_but_not_exported(js_file, "localOnly", source=None) is True

    def test_typescript_unexported(self, tmp_path: Path) -> None:
        ts_file = tmp_path / "utils.ts"
        content = (
            "function internal(x: number): number {\n    return x;\n}\n\n"
            "export function external(y: number): number {\n    return y + 1;\n}\n"
        )
        ts_file.write_text(content, encoding="utf-8")
        assert _is_js_ts_function_exists_but_not_exported(ts_file, "internal", source=content) is True
        assert _is_js_ts_function_exists_but_not_exported(ts_file, "external", source=content) is False

    def test_arrow_function_unexported(self, tmp_path: Path) -> None:
        js_file = tmp_path / "arrows.js"
        content = "const helper = (x) => {\n    return x + 1;\n};\n\nexport const main = () => {\n    return 2;\n};\n"
        js_file.write_text(content, encoding="utf-8")
        assert _is_js_ts_function_exists_but_not_exported(js_file, "helper", source=content) is True
        assert _is_js_ts_function_exists_but_not_exported(js_file, "main", source=content) is False

    def test_source_param_matches_disk_read(self, tmp_path: Path) -> None:
        js_file = tmp_path / "consistent.js"
        content = "function local() {\n    return 'hi';\n}\n\nexport function exported() {\n    return 'bye';\n}\n"
        js_file.write_text(content, encoding="utf-8")
        # Results should be identical whether source is passed or read from disk
        assert _is_js_ts_function_exists_but_not_exported(js_file, "local", source=content) == (
            _is_js_ts_function_exists_but_not_exported(js_file, "local", source=None)
        )
        assert _is_js_ts_function_exists_but_not_exported(js_file, "exported", source=content) == (
            _is_js_ts_function_exists_but_not_exported(js_file, "exported", source=None)
        )
