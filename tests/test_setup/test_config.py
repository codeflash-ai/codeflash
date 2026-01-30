"""Tests for config schema and config writer."""

import json
from pathlib import Path

import pytest
import tomlkit

from codeflash.setup.config_schema import CodeflashConfig
from codeflash.setup.config_writer import (
    _write_package_json,
    _write_pyproject_toml,
    create_pyproject_toml,
    remove_config,
    write_config,
)
from codeflash.setup.detector import detect_project


class TestCodeflashConfig:
    """Tests for CodeflashConfig Pydantic model."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = CodeflashConfig(language="python")

        assert config.language == "python"
        assert config.module_root == "."
        assert config.tests_root is None
        assert config.formatter_cmds == []
        assert config.git_remote == "origin"
        assert config.disable_telemetry is False

    def test_all_fields(self):
        """Should accept all fields."""
        config = CodeflashConfig(
            language="javascript",
            module_root="src",
            tests_root="tests",
            test_runner="jest",
            formatter_cmds=["npx prettier --write $file"],
            ignore_paths=["dist", "node_modules"],
            benchmarks_root="benchmarks",
            git_remote="upstream",
            disable_telemetry=True,
        )

        assert config.language == "javascript"
        assert config.module_root == "src"
        assert config.tests_root == "tests"
        assert config.test_runner == "jest"
        assert config.formatter_cmds == ["npx prettier --write $file"]
        assert config.ignore_paths == ["dist", "node_modules"]
        assert config.git_remote == "upstream"
        assert config.disable_telemetry is True

    def test_to_pyproject_dict(self):
        """Should convert to pyproject.toml format with kebab-case."""
        config = CodeflashConfig(
            language="python",
            module_root="codeflash",
            tests_root="tests",
            formatter_cmds=["ruff format $file"],
            ignore_paths=["dist"],
        )

        result = config.to_pyproject_dict()

        assert result["module-root"] == "codeflash"
        assert result["tests-root"] == "tests"
        assert result["formatter-cmds"] == ["ruff format $file"]
        assert result["ignore-paths"] == ["dist"]
        # Should not include default values
        assert "git-remote" not in result
        assert "disable-telemetry" not in result

    def test_to_pyproject_dict_minimal(self):
        """Should only include non-default values."""
        config = CodeflashConfig(
            language="python",
            module_root="src",
        )

        result = config.to_pyproject_dict()

        assert "module-root" in result
        # Empty formatter should result in "disabled"
        assert result.get("formatter-cmds") == ["disabled"]

    def test_to_package_json_dict(self):
        """Should convert to package.json format with camelCase."""
        config = CodeflashConfig(
            language="javascript",
            module_root="lib",  # Non-default
            formatter_cmds=["npx prettier --write $file"],
            ignore_paths=["dist"],
            disable_telemetry=True,
        )

        result = config.to_package_json_dict()

        assert result["moduleRoot"] == "lib"
        assert result["formatterCmds"] == ["npx prettier --write $file"]
        assert result["ignorePaths"] == ["dist"]
        assert result["disableTelemetry"] is True
        # Should not include default values
        assert "gitRemote" not in result

    def test_to_package_json_dict_minimal(self):
        """Should be empty when all values are defaults."""
        config = CodeflashConfig(
            language="javascript",
            module_root="src",  # Default for JS
        )

        result = config.to_package_json_dict()

        # src is a default, should not be included
        assert "moduleRoot" not in result

    def test_from_detected_project(self, tmp_path):
        """Should create config from DetectedProject."""
        # Create a simple Python project
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "test").mkdir()
        (tmp_path / "test" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()

        detected = detect_project(tmp_path)
        config = CodeflashConfig.from_detected_project(detected)

        assert config.language == detected.language
        assert config.test_runner == detected.test_runner

    def test_from_pyproject_dict(self):
        """Should create config from pyproject.toml dict."""
        data = {
            "module-root": "src",
            "tests-root": "tests",
            "formatter-cmds": ["black $file"],
            "disable-telemetry": True,
        }

        config = CodeflashConfig.from_pyproject_dict(data)

        assert config.module_root == "src"
        assert config.tests_root == "tests"
        assert config.formatter_cmds == ["black $file"]
        assert config.disable_telemetry is True
        assert config.language == "python"  # Default for pyproject

    def test_from_package_json_dict(self):
        """Should create config from package.json dict."""
        data = {
            "moduleRoot": "lib",
            "formatterCmds": ["npx prettier --write $file"],
            "disableTelemetry": True,
        }

        config = CodeflashConfig.from_package_json_dict(data)

        assert config.module_root == "lib"
        assert config.formatter_cmds == ["npx prettier --write $file"]
        assert config.disable_telemetry is True
        assert config.language == "javascript"  # Default for package.json


class TestWritePyprojectToml:
    """Tests for writing to pyproject.toml."""

    def test_creates_new_pyproject(self, tmp_path):
        """Should create pyproject.toml if it doesn't exist."""
        config = CodeflashConfig(
            language="python",
            module_root="src",
            tests_root="tests",
        )

        success, message = _write_pyproject_toml(tmp_path, config)

        assert success is True
        assert (tmp_path / "pyproject.toml").exists()

        # Verify content
        content = (tmp_path / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        assert "tool" in data
        assert "codeflash" in data["tool"]
        assert data["tool"]["codeflash"]["module-root"] == "src"

    def test_preserves_existing_content(self, tmp_path):
        """Should preserve existing pyproject.toml content."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "myapp"\nversion = "1.0.0"\n\n[tool.ruff]\nline-length = 120'
        )

        config = CodeflashConfig(
            language="python",
            module_root="src",
        )

        success, message = _write_pyproject_toml(tmp_path, config)

        assert success is True

        # Verify existing content preserved
        content = (tmp_path / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        assert data["project"]["name"] == "myapp"
        assert data["tool"]["ruff"]["line-length"] == 120
        assert data["tool"]["codeflash"]["module-root"] == "src"

    def test_updates_existing_codeflash_section(self, tmp_path):
        """Should update existing codeflash section."""
        (tmp_path / "pyproject.toml").write_text(
            '[tool.codeflash]\nmodule-root = "old"\ntests-root = "old_tests"'
        )

        config = CodeflashConfig(
            language="python",
            module_root="new",
            tests_root="new_tests",
        )

        success, message = _write_pyproject_toml(tmp_path, config)

        assert success is True

        content = (tmp_path / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        assert data["tool"]["codeflash"]["module-root"] == "new"
        assert data["tool"]["codeflash"]["tests-root"] == "new_tests"


class TestWritePackageJson:
    """Tests for writing to package.json."""

    def test_adds_codeflash_section(self, tmp_path):
        """Should add codeflash section to package.json."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "version": "1.0.0"
        }, indent=2))

        config = CodeflashConfig(
            language="javascript",
            module_root="lib",
            formatter_cmds=["npx prettier --write $file"],
        )

        success, message = _write_package_json(tmp_path, config)

        assert success is True

        # Verify content
        with (tmp_path / "package.json").open() as f:
            data = json.load(f)
        assert data["name"] == "myapp"  # Preserved
        assert "codeflash" in data
        assert data["codeflash"]["moduleRoot"] == "lib"

    def test_preserves_existing_content(self, tmp_path):
        """Should preserve existing package.json content."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "dependencies": {"lodash": "^4.17.0"},
            "devDependencies": {"jest": "^29.0.0"}
        }, indent=2))

        config = CodeflashConfig(
            language="javascript",
            module_root="lib",
        )

        success, message = _write_package_json(tmp_path, config)

        assert success is True

        with (tmp_path / "package.json").open() as f:
            data = json.load(f)
        assert data["dependencies"]["lodash"] == "^4.17.0"
        assert data["devDependencies"]["jest"] == "^29.0.0"

    def test_removes_empty_codeflash_section(self, tmp_path):
        """Should remove codeflash section if all defaults."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "codeflash": {"moduleRoot": "old"}
        }, indent=2))

        # Config with all defaults - should result in empty dict
        config = CodeflashConfig(
            language="javascript",
            module_root="src",  # Default
        )

        success, message = _write_package_json(tmp_path, config)

        assert success is True

        with (tmp_path / "package.json").open() as f:
            data = json.load(f)
        # Empty codeflash section should be removed
        assert "codeflash" not in data

    def test_fails_if_no_package_json(self, tmp_path):
        """Should fail if package.json doesn't exist."""
        config = CodeflashConfig(language="javascript")

        success, message = _write_package_json(tmp_path, config)

        assert success is False
        assert "No package.json" in message


class TestWriteConfig:
    """Tests for the unified write_config function."""

    def test_writes_to_pyproject_for_python(self, tmp_path):
        """Should write to pyproject.toml for Python projects."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")

        detected = detect_project(tmp_path)
        success, message = write_config(detected)

        assert success is True
        assert "pyproject.toml" in message

    def test_writes_to_package_json_for_js(self, tmp_path):
        """Should write to package.json for JavaScript projects."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "src").mkdir()

        detected = detect_project(tmp_path)
        success, message = write_config(detected)

        assert success is True


class TestRemoveConfig:
    """Tests for remove_config function."""

    def test_removes_from_pyproject(self, tmp_path):
        """Should remove codeflash section from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\n\n[tool.codeflash]\nmodule-root = "src"'
        )

        success, message = remove_config(tmp_path, "python")

        assert success is True

        content = (tmp_path / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        assert data["project"]["name"] == "test"  # Preserved
        assert "codeflash" not in data.get("tool", {})

    def test_removes_from_package_json(self, tmp_path):
        """Should remove codeflash section from package.json."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "codeflash": {"moduleRoot": "src"}
        }, indent=2))

        success, message = remove_config(tmp_path, "javascript")

        assert success is True

        with (tmp_path / "package.json").open() as f:
            data = json.load(f)
        assert data["name"] == "test"  # Preserved
        assert "codeflash" not in data


class TestCreatePyprojectToml:
    """Tests for create_pyproject_toml function."""

    def test_creates_minimal_pyproject(self, tmp_path):
        """Should create minimal pyproject.toml."""
        success, message = create_pyproject_toml(tmp_path)

        assert success is True
        assert (tmp_path / "pyproject.toml").exists()

        content = (tmp_path / "pyproject.toml").read_text()
        assert "codeflash" in content.lower()

    def test_fails_if_already_exists(self, tmp_path):
        """Should fail if pyproject.toml already exists."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        success, message = create_pyproject_toml(tmp_path)

        assert success is False
        assert "already exists" in message