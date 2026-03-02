"""Tests for framework detection from package.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeflash.languages.javascript.frameworks.detector import (
    FrameworkInfo,
    _parse_major_version,
    _parse_version_string,
    detect_framework,
)


@pytest.fixture(autouse=True)
def clear_cache():
    detect_framework.cache_clear()
    yield
    detect_framework.cache_clear()


def _write_package_json(tmp_path: Path, data: dict) -> Path:
    package_json = tmp_path / "package.json"
    package_json.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


class TestDetectFramework:
    def test_react_18_with_testing_library(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"},
            "devDependencies": {"@testing-library/react": "^14.0.0", "jest": "^29.0.0"},
        })
        info = detect_framework(project)
        assert info.name == "react"
        assert info.version == "18.2.0"
        assert info.react_version_major == 18
        assert info.has_testing_library is True
        assert info.has_react_compiler is False

    def test_react_19_detects_compiler(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^19.0.0"},
        })
        info = detect_framework(project)
        assert info.name == "react"
        assert info.react_version_major == 19
        assert info.has_react_compiler is True

    def test_react_with_babel_compiler_plugin(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^18.3.0"},
            "devDependencies": {"babel-plugin-react-compiler": "^0.1.0"},
        })
        info = detect_framework(project)
        assert info.has_react_compiler is True

    def test_no_react(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"vue": "^3.0.0"},
        })
        info = detect_framework(project)
        assert info.name == "none"
        assert info.react_version_major is None

    def test_no_package_json(self, tmp_path):
        info = detect_framework(tmp_path)
        assert info.name == "none"

    def test_invalid_json(self, tmp_path):
        (tmp_path / "package.json").write_text("not json", encoding="utf-8")
        info = detect_framework(tmp_path)
        assert info.name == "none"

    def test_react_in_dev_dependencies(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "devDependencies": {"react": "^17.0.2"},
        })
        info = detect_framework(project)
        assert info.name == "react"
        assert info.react_version_major == 17

    def test_without_testing_library(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^18.2.0"},
        })
        info = detect_framework(project)
        assert info.has_testing_library is False

    def test_tilde_version_range(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "~16.14.0"},
        })
        info = detect_framework(project)
        assert info.version == "16.14.0"
        assert info.react_version_major == 16

    def test_exact_version(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "18.2.0"},
        })
        info = detect_framework(project)
        assert info.version == "18.2.0"

    def test_caching(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^18.2.0"},
        })
        info1 = detect_framework(project)
        info2 = detect_framework(project)
        assert info1 is info2

    def test_dev_dependencies_tracked(self, tmp_path):
        project = _write_package_json(tmp_path, {
            "dependencies": {"react": "^18.2.0"},
            "devDependencies": {"jest": "^29.0.0", "typescript": "^5.0.0"},
        })
        info = detect_framework(project)
        assert "jest" in info.dev_dependencies
        assert "typescript" in info.dev_dependencies
        assert "react" in info.dev_dependencies


class TestParseVersionString:
    def test_caret_range(self):
        assert _parse_version_string("^18.2.0") == "18.2.0"

    def test_tilde_range(self):
        assert _parse_version_string("~17.0.2") == "17.0.2"

    def test_exact(self):
        assert _parse_version_string("18.2.0") == "18.2.0"

    def test_gte_range(self):
        assert _parse_version_string(">=16.8.0") == "16.8.0"

    def test_invalid(self):
        assert _parse_version_string("latest") is None

    def test_empty(self):
        assert _parse_version_string("") is None


class TestParseMajorVersion:
    def test_standard(self):
        assert _parse_major_version("18.2.0") == 18

    def test_single_digit(self):
        assert _parse_major_version("3") == 3

    def test_none(self):
        assert _parse_major_version(None) is None

    def test_non_numeric(self):
        assert _parse_major_version("abc") is None
