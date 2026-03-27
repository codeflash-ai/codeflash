"""Tests for JavaScript/TypeScript module resolution validation utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from codeflash.code_utils.config_js_validation import infer_js_module_root, validate_js_module_resolution

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateJsModuleResolution:
    def test_valid_source_in_module_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")
        src = project_root / "src"
        src.mkdir()
        source_file = src / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")

        valid, error = validate_js_module_resolution(source_file, project_root, src)
        assert valid is True
        assert error == ""

    def test_source_does_not_exist(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")
        source_file = project_root / "src" / "missing.js"

        valid, error = validate_js_module_resolution(source_file, project_root, project_root)
        assert valid is False
        assert "does not exist" in error

    def test_source_outside_project_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")
        outside_file = tmp_path / "outside.js"
        outside_file.write_text("export function foo() {}", encoding="utf-8")

        valid, error = validate_js_module_resolution(outside_file, project_root, project_root)
        assert valid is False
        assert "not within project root" in error

    def test_no_package_json(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        source_file = project_root / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")

        valid, error = validate_js_module_resolution(source_file, project_root, project_root)
        assert valid is False
        assert "No package.json" in error

    def test_source_outside_module_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")
        src = project_root / "src"
        src.mkdir()
        other = project_root / "other"
        other.mkdir()
        source_file = other / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")

        valid, error = validate_js_module_resolution(source_file, project_root, src)
        assert valid is False
        assert "not within module root" in error
        assert "moduleRoot" in error

    def test_module_root_equals_project_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")
        source_file = project_root / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")

        valid, error = validate_js_module_resolution(source_file, project_root, project_root)
        assert valid is True
        assert error == ""


class TestInferJsModuleRoot:
    def test_infers_src_directory(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        src = project_root / "src"
        src.mkdir()
        source_file = src / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")
        (project_root / "package.json").write_text("{}", encoding="utf-8")

        result = infer_js_module_root(source_file, project_root)
        assert result == src.resolve()

    def test_infers_from_package_json_main_field(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        app = project_root / "app"
        app.mkdir()
        source_file = app / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")
        (project_root / "package.json").write_text(json.dumps({"main": "app/index.js"}), encoding="utf-8")

        result = infer_js_module_root(source_file, project_root)
        assert result == app.resolve()

    def test_falls_back_to_project_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        source_file = project_root / "index.js"
        source_file.write_text("export function foo() {}", encoding="utf-8")
        (project_root / "package.json").write_text("{}", encoding="utf-8")

        result = infer_js_module_root(source_file, project_root)
        assert result == project_root.resolve()

    def test_falls_back_to_parent_without_package_json(self, tmp_path: Path) -> None:
        source_file = tmp_path / "standalone" / "index.js"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("export function foo() {}", encoding="utf-8")

        result = infer_js_module_root(source_file, project_root=None)
        assert result == source_file.parent.resolve()
