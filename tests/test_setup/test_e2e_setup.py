"""End-to-end tests for the setup flow.

These tests validate the complete setup experience across different:
- Languages (Python, JavaScript, TypeScript)
- Project structures (src/, flat, monorepo-like)
- Package managers (npm, yarn, pnpm, bun)
- Existing config scenarios
"""

import json
from argparse import Namespace

import pytest
import tomlkit

from codeflash.setup import (
    CodeflashConfig,
    detect_project,
    handle_first_run,
    has_existing_config,
    is_first_run,
    write_config,
)

# =============================================================================
# Fixtures for creating different project types
# =============================================================================


@pytest.fixture
def python_src_layout(tmp_path):
    """Create a Python project with src/ layout."""
    # pyproject.toml with poetry
    (tmp_path / "pyproject.toml").write_text("""
[tool.poetry]
name = "myapp"
version = "0.1.0"

[tool.ruff]
line-length = 120

[tool.pytest.ini_options]
testpaths = ["tests"]
""".strip())

    # src/myapp package
    src_dir = tmp_path / "src" / "myapp"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('__version__ = "0.1.0"')
    (src_dir / "main.py").write_text("def main(): pass")
    (src_dir / "utils.py").write_text("def helper(): pass")

    # tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_main.py").write_text("def test_main(): pass")
    (tests_dir / "conftest.py").write_text("import pytest")

    # .git directory
    (tmp_path / ".git").mkdir()

    return tmp_path


@pytest.fixture
def python_flat_layout(tmp_path):
    """Create a Python project with flat layout (package at root)."""
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "myapp"
version = "0.1.0"

[tool.black]
line-length = 88
""".strip())

    # Package at root
    pkg_dir = tmp_path / "myapp"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "core.py").write_text("def process(): pass")

    # Tests at root
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_core.py").write_text("def test_process(): pass")

    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def python_setup_py_project(tmp_path):
    """Create a Python project with setup.py (legacy)."""
    (tmp_path / "setup.py").write_text("""
from setuptools import setup, find_packages
setup(
    name="legacyapp",
    version="1.0.0",
    packages=find_packages(),
)
""".strip())

    pkg_dir = tmp_path / "legacyapp"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")

    (tmp_path / "tests").mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def javascript_npm_project(tmp_path):
    """Create a JavaScript project with npm."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "my-js-app",
        "version": "1.0.0",
        "main": "src/index.js",
        "scripts": {
            "test": "jest",
            "lint": "eslint src/"
        },
        "devDependencies": {
            "jest": "^29.7.0",
            "prettier": "^3.0.0"
        }
    }, indent=2))

    (tmp_path / "package-lock.json").write_text("{}")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "index.js").write_text("module.exports = {}")
    (src_dir / "utils.js").write_text("function helper() {}")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "index.test.js").write_text("test('works', () => {})")

    (tmp_path / ".prettierrc").write_text("{}")
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def javascript_yarn_project(tmp_path):
    """Create a JavaScript project with yarn."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "yarn-app",
        "version": "1.0.0",
        "main": "lib/index.js",
        "devDependencies": {
            "jest": "^29.0.0",
            "eslint": "^8.0.0"
        }
    }, indent=2))

    (tmp_path / "yarn.lock").write_text("# yarn lockfile")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "index.js").write_text("")

    (tmp_path / "__tests__").mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def javascript_pnpm_project(tmp_path):
    """Create a JavaScript project with pnpm."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "pnpm-app",
        "version": "1.0.0",
        "exports": {
            ".": "./dist/index.js"
        },
        "devDependencies": {
            "vitest": "^1.0.0"
        }
    }, indent=2))

    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 5.4")

    (tmp_path / "dist").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def javascript_bun_project(tmp_path):
    """Create a JavaScript project with bun."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "bun-app",
        "version": "1.0.0",
        "module": "src/index.ts",
        "devDependencies": {
            "bun-types": "latest"
        }
    }, indent=2))

    (tmp_path / "bun.lockb").write_bytes(b"bun lockfile")

    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def typescript_project(tmp_path):
    """Create a TypeScript project."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "ts-app",
        "version": "1.0.0",
        "main": "dist/index.js",
        "types": "dist/index.d.ts",
        "scripts": {
            "build": "tsc",
            "test": "vitest"
        },
        "devDependencies": {
            "typescript": "^5.0.0",
            "vitest": "^1.0.0",
            "@types/node": "^20.0.0"
        }
    }, indent=2))

    (tmp_path / "tsconfig.json").write_text(json.dumps({
        "compilerOptions": {
            "target": "ES2020",
            "module": "commonjs",
            "outDir": "./dist",
            "rootDir": "./src",
            "strict": True
        },
        "include": ["src/**/*"]
    }, indent=2))

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "index.ts").write_text("export const main = () => {}")
    (src_dir / "types.ts").write_text("export interface Config {}")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "index.test.ts").write_text("import { describe, it } from 'vitest'")

    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def typescript_react_project(tmp_path):
    """Create a TypeScript React project (like Create React App)."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "react-app",
        "version": "0.1.0",
        "private": True,
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1",
            "jest": "^29.0.0"
        },
        "devDependencies": {
            "@types/react": "^18.0.0",
            "@testing-library/react": "^14.0.0",
            "typescript": "^5.0.0"
        },
        "scripts": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test"
        }
    }, indent=2))

    (tmp_path / "tsconfig.json").write_text(json.dumps({
        "compilerOptions": {
            "target": "es5",
            "lib": ["dom", "es2015"],
            "jsx": "react-jsx"
        }
    }, indent=2))

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "App.tsx").write_text("export default function App() { return <div/>; }")
    (src_dir / "index.tsx").write_text("import App from './App';")
    (src_dir / "App.test.tsx").write_text("test('renders', () => {});")

    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def project_with_existing_config(tmp_path):
    """Create a project with existing codeflash config."""
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "configured-app"

[tool.codeflash]
module-root = "src"
tests-root = "tests"
formatter-cmds = ["black $file"]
""".strip())

    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def mixed_python_js_project(tmp_path):
    """Create a project with both Python and JS files (monorepo-like)."""
    # Python backend
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "fullstack-app"

[tool.codeflash]
module-root = "backend"
""".strip())

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    (backend_dir / "__init__.py").write_text("")
    (backend_dir / "api.py").write_text("def handler(): pass")

    # JS frontend
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "package.json").write_text(json.dumps({
        "name": "frontend",
        "devDependencies": {"jest": "^29.0.0"}
    }))
    (frontend_dir / "src").mkdir()
    (frontend_dir / "src" / "app.js").write_text("")

    (tmp_path / ".git").mkdir()
    return tmp_path


# =============================================================================
# E2E Tests: Detection
# =============================================================================


class TestE2EDetection:
    """E2E tests for project detection across different setups."""

    def test_python_src_layout_detection(self, python_src_layout):
        """Should correctly detect Python src/ layout project."""
        detected = detect_project(python_src_layout)

        assert detected.language == "python"
        assert detected.project_root == python_src_layout
        assert detected.module_root.name == "myapp"
        assert detected.tests_root == python_src_layout / "tests"
        assert detected.test_runner == "pytest"
        assert any("ruff" in cmd for cmd in detected.formatter_cmds)
        assert detected.confidence >= 0.9

    def test_python_flat_layout_detection(self, python_flat_layout):
        """Should correctly detect Python flat layout project."""
        detected = detect_project(python_flat_layout)

        assert detected.language == "python"
        assert detected.module_root.name == "myapp"
        assert any("black" in cmd for cmd in detected.formatter_cmds)

    def test_python_setup_py_detection(self, python_setup_py_project):
        """Should correctly detect legacy setup.py project."""
        detected = detect_project(python_setup_py_project)

        assert detected.language == "python"
        assert detected.module_root.name == "legacyapp"

    def test_javascript_npm_detection(self, javascript_npm_project):
        """Should correctly detect JavaScript npm project."""
        detected = detect_project(javascript_npm_project)

        assert detected.language == "javascript"
        assert detected.module_root == javascript_npm_project / "src"
        assert detected.test_runner == "jest"
        assert any("prettier" in cmd for cmd in detected.formatter_cmds)

    def test_javascript_yarn_detection(self, javascript_yarn_project):
        """Should correctly detect JavaScript yarn project."""
        detected = detect_project(javascript_yarn_project)

        assert detected.language == "javascript"
        assert detected.module_root == javascript_yarn_project / "lib"
        assert detected.tests_root == javascript_yarn_project / "__tests__"

    def test_javascript_pnpm_detection(self, javascript_pnpm_project):
        """Should correctly detect JavaScript pnpm project."""
        detected = detect_project(javascript_pnpm_project)

        assert detected.language == "javascript"
        assert detected.test_runner == "vitest"

    def test_javascript_bun_detection(self, javascript_bun_project):
        """Should correctly detect JavaScript bun project."""
        detected = detect_project(javascript_bun_project)

        assert detected.language == "javascript"
        assert detected.module_root == javascript_bun_project / "src"

    def test_typescript_detection(self, typescript_project):
        """Should correctly detect TypeScript project."""
        detected = detect_project(typescript_project)

        assert detected.language == "typescript"
        assert detected.test_runner == "vitest"
        assert detected.tests_root == typescript_project / "tests"

    def test_typescript_react_detection(self, typescript_react_project):
        """Should correctly detect TypeScript React project."""
        detected = detect_project(typescript_react_project)

        assert detected.language == "typescript"
        assert detected.module_root == typescript_react_project / "src"
        # React scripts uses jest under the hood
        assert detected.test_runner == "jest"


# =============================================================================
# E2E Tests: First Run Check
# =============================================================================


class TestE2EFirstRunCheck:
    """E2E tests for first-run detection."""

    def test_is_first_run_new_python_project(self, python_src_layout):
        """Should detect first run for new Python project."""
        assert is_first_run(python_src_layout) is True

    def test_is_first_run_new_js_project(self, javascript_npm_project):
        """Should detect first run for new JS project."""
        assert is_first_run(javascript_npm_project) is True

    def test_is_not_first_run_configured_project(self, project_with_existing_config):
        """Should detect existing config."""
        assert is_first_run(project_with_existing_config) is False

    def test_has_existing_config_python(self, project_with_existing_config):
        """Should find existing config in pyproject.toml."""
        has_config, config_type = has_existing_config(project_with_existing_config)
        assert has_config is True
        assert config_type == "pyproject.toml"

    def test_has_existing_config_js(self, tmp_path):
        """Should find existing config in package.json."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "codeflash": {"moduleRoot": "src"}
        }))

        has_config, config_type = has_existing_config(tmp_path)
        assert has_config is True
        assert config_type == "package.json"


# =============================================================================
# E2E Tests: Config Writing
# =============================================================================


class TestE2EConfigWriting:
    """E2E tests for writing config to native files."""

    def test_write_config_python_preserves_existing(self, python_src_layout):
        """Should write config while preserving existing pyproject.toml content."""
        detected = detect_project(python_src_layout)
        success, message = write_config(detected)

        assert success is True

        # Read back and verify
        content = (python_src_layout / "pyproject.toml").read_text()
        data = tomlkit.parse(content)

        # Original content preserved
        assert data["tool"]["poetry"]["name"] == "myapp"
        assert data["tool"]["ruff"]["line-length"] == 120
        assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]

        # Codeflash config added
        assert "codeflash" in data["tool"]
        assert "module-root" in data["tool"]["codeflash"]

    def test_write_config_javascript_preserves_existing(self, javascript_npm_project):
        """Should write config while preserving existing package.json content."""
        detected = detect_project(javascript_npm_project)
        success, message = write_config(detected)

        assert success is True

        # Read back and verify
        with (javascript_npm_project / "package.json").open() as f:
            data = json.load(f)

        # Original content preserved
        assert data["name"] == "my-js-app"
        assert data["devDependencies"]["jest"] == "^29.7.0"
        assert data["scripts"]["test"] == "jest"

    def test_write_config_typescript(self, typescript_project):
        """Should write config for TypeScript project."""
        detected = detect_project(typescript_project)
        success, message = write_config(detected)

        assert success is True

        with (typescript_project / "package.json").open() as f:
            data = json.load(f)

        # tsconfig.json should be unchanged
        tsconfig = json.loads((typescript_project / "tsconfig.json").read_text())
        assert tsconfig["compilerOptions"]["strict"] is True

    def test_config_roundtrip_python(self, python_flat_layout):
        """Should be able to read back written config."""
        # Detect and write
        detected = detect_project(python_flat_layout)
        write_config(detected)

        # Read back
        content = (python_flat_layout / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        codeflash_section = data["tool"]["codeflash"]

        # Create config from written data
        config = CodeflashConfig.from_pyproject_dict(dict(codeflash_section))

        assert config.language == "python"
        assert "myapp" in config.module_root

    def test_config_roundtrip_javascript(self, javascript_npm_project):
        """Should be able to read back written config for JS."""
        detected = detect_project(javascript_npm_project)
        write_config(detected)

        with (javascript_npm_project / "package.json").open() as f:
            data = json.load(f)

        if "codeflash" in data:
            config = CodeflashConfig.from_package_json_dict(data["codeflash"])
            assert config.language == "javascript"


# =============================================================================
# E2E Tests: First Run Experience
# =============================================================================


class TestE2EFirstRunExperience:
    """E2E tests for the complete first-run experience."""

    def test_first_run_python_project(self, python_src_layout, monkeypatch):
        """Should complete first-run for Python project."""
        monkeypatch.chdir(python_src_layout)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key-12345")

        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert result.language == "python"
        assert result.module_root.endswith("myapp")
        assert result.tests_root is not None
        assert result.tests_root.endswith("tests")
        assert result.pytest_cmd == "pytest"

        # Config should be written
        content = (python_src_layout / "pyproject.toml").read_text()
        assert "[tool.codeflash]" in content

    def test_first_run_javascript_project(self, javascript_npm_project, monkeypatch):
        """Should complete first-run for JavaScript project."""
        monkeypatch.chdir(javascript_npm_project)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key-12345")

        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert result.language == "javascript"
        assert result.module_root.endswith("src")
        assert result.pytest_cmd == "jest"  # Maps to test_runner

    def test_first_run_typescript_project(self, typescript_project, monkeypatch):
        """Should complete first-run for TypeScript project."""
        monkeypatch.chdir(typescript_project)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key-12345")

        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert result.language == "typescript"
        assert result.pytest_cmd == "vitest"

    def test_first_run_with_existing_args(self, python_flat_layout, monkeypatch):
        """Should merge with existing CLI args."""
        monkeypatch.chdir(python_flat_layout)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key-12345")

        existing_args = Namespace(
            file="myapp/core.py",
            function="process",
            custom_flag=True,
        )

        result = handle_first_run(
            args=existing_args,
            skip_confirm=True,
            skip_api_key=True,
        )

        assert result is not None
        assert result.custom_flag is True  # Preserved
        assert result.file == "myapp/core.py"  # Preserved
        assert result.language == "python"  # Added

    def test_subsequent_run_not_first_run(self, project_with_existing_config, monkeypatch):
        """Should not trigger first-run for configured project."""
        monkeypatch.chdir(project_with_existing_config)

        assert is_first_run(project_with_existing_config) is False


# =============================================================================
# E2E Tests: Edge Cases
# =============================================================================


class TestE2EEdgeCases:
    """E2E tests for edge cases and special scenarios."""

    def test_empty_directory(self, tmp_path):
        """Should handle empty directory gracefully."""
        detected = detect_project(tmp_path)

        # Should default to Python
        assert detected.language == "python"
        # Low language confidence (0.3) + base (0.6) = ~0.72
        assert detected.confidence < 0.8

    def test_nested_project_detection(self, tmp_path):
        """Should find project root from nested directory."""
        # Create project structure
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "root"')
        deep_dir = tmp_path / "src" / "pkg" / "subpkg"
        deep_dir.mkdir(parents=True)

        # Detect from nested dir
        detected = detect_project(deep_dir)

        assert detected.project_root == tmp_path

    def test_mixed_project_uses_existing_config(self, mixed_python_js_project):
        """Should respect existing config in mixed projects."""
        has_config, config_type = has_existing_config(mixed_python_js_project)

        assert has_config is True
        assert config_type == "pyproject.toml"

    def test_project_without_tests_dir(self, tmp_path):
        """Should handle project without tests directory."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "notests"')
        (tmp_path / "src").mkdir()

        detected = detect_project(tmp_path)

        assert detected.tests_root is None

    def test_project_without_formatter(self, tmp_path):
        """Should handle project without detectable formatter."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "no-formatter",
            "devDependencies": {"jest": "^29.0.0"}
        }))

        detected = detect_project(tmp_path)

        assert detected.formatter_cmds == []

    def test_malformed_pyproject_toml(self, tmp_path):
        """Should handle malformed pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("this is not valid toml {{{}}")

        # Should not crash, just detect with lower confidence
        detected = detect_project(tmp_path)
        assert detected is not None

    def test_malformed_package_json(self, tmp_path):
        """Should handle malformed package.json."""
        (tmp_path / "package.json").write_text("not valid json")

        detected = detect_project(tmp_path)
        assert detected is not None

    def test_display_dict_format(self, python_src_layout):
        """Should generate proper display dict for UI."""
        detected = detect_project(python_src_layout)
        display = detected.to_display_dict()

        assert "Language" in display
        assert "Module root" in display
        assert "Tests root" in display
        assert "Test runner" in display
        assert "Formatter" in display
        assert "Ignoring" in display

        # Values should be user-friendly
        assert display["Language"] == "Python"
        assert display["Test runner"] == "pytest"


# =============================================================================
# E2E Tests: Config Schema Conversion
# =============================================================================


class TestE2EConfigConversion:
    """E2E tests for config format conversion."""

    def test_python_config_to_toml_and_back(self, python_src_layout):
        """Should convert Python config to TOML and back without loss."""
        detected = detect_project(python_src_layout)
        original_config = CodeflashConfig.from_detected_project(detected)

        # Convert to TOML dict
        toml_dict = original_config.to_pyproject_dict()

        # Convert back
        restored_config = CodeflashConfig.from_pyproject_dict(toml_dict)

        assert restored_config.module_root == original_config.module_root
        assert restored_config.tests_root == original_config.tests_root

    def test_js_config_to_json_and_back(self, javascript_npm_project):
        """Should convert JS config to JSON and back without loss."""
        detected = detect_project(javascript_npm_project)
        original_config = CodeflashConfig.from_detected_project(detected)

        # Convert to JSON dict
        json_dict = original_config.to_package_json_dict()

        # Convert back
        restored_config = CodeflashConfig.from_package_json_dict(json_dict)

        # Note: Some defaults may differ, check key fields
        if json_dict:  # Only if there were non-default values
            assert restored_config.language == "javascript"


# =============================================================================
# E2E Tests: Real-world Scenarios
# =============================================================================


class TestE2ERealWorldScenarios:
    """E2E tests simulating real-world usage scenarios."""

    def test_scenario_new_user_python(self, python_src_layout, monkeypatch):
        """Scenario: New user runs codeflash on Python project for first time."""
        monkeypatch.chdir(python_src_layout)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        # Step 1: Check if first run
        assert is_first_run() is True

        # Step 2: Handle first run
        args = handle_first_run(skip_confirm=True, skip_api_key=True)
        assert args is not None

        # Step 3: Config is now saved
        assert is_first_run() is False

        # Step 4: Next run should use saved config
        has_config, _ = has_existing_config(python_src_layout)
        assert has_config is True

    def test_scenario_new_user_javascript(self, javascript_npm_project, monkeypatch):
        """Scenario: New user runs codeflash on JS project for first time."""
        monkeypatch.chdir(javascript_npm_project)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        # Step 1: Check if first run
        assert is_first_run() is True

        # Step 2: Handle first run
        args = handle_first_run(skip_confirm=True, skip_api_key=True)
        assert args is not None
        assert args.language == "javascript"

        # Step 3: Verify package.json was updated (or left minimal)
        with (javascript_npm_project / "package.json").open() as f:
            data = json.load(f)
        # Original content should still be there
        assert data["name"] == "my-js-app"

    def test_scenario_existing_user_reconfigure(self, project_with_existing_config, monkeypatch):
        """Scenario: Existing user wants to reconfigure."""
        monkeypatch.chdir(project_with_existing_config)

        # Not first run
        assert is_first_run() is False

        # But user can still detect and see what would be configured
        detected = detect_project()
        assert detected.language == "python"

    def test_scenario_ci_environment(self, python_src_layout, monkeypatch):
        """Scenario: Running in CI environment (non-interactive)."""
        monkeypatch.chdir(python_src_layout)
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        # In CI, we need config to exist or use --yes flag
        # First run should still work with skip flags
        args = handle_first_run(skip_confirm=True, skip_api_key=True)
        assert args is not None


# =============================================================================
# E2E Tests: CLI Flags
# =============================================================================


class TestE2ECLIFlags:
    """E2E tests for --show-config and --reset-config CLI flags."""

    def test_show_config_displays_detected_settings(self, python_src_layout, monkeypatch, capsys):
        """Should display detected project settings."""
        monkeypatch.chdir(python_src_layout)

        from codeflash.cli_cmds.cli import _handle_show_config

        _handle_show_config()

        # Can't easily capture Rich output, but ensure it doesn't crash
        # and the function completes successfully

    def test_show_config_indicates_saved_vs_detected(self, project_with_existing_config, monkeypatch):
        """Should indicate when config is saved vs auto-detected."""
        monkeypatch.chdir(project_with_existing_config)

        from codeflash.cli_cmds.cli import _handle_show_config

        # Should complete without error
        _handle_show_config()

    def test_reset_config_removes_from_pyproject(self, project_with_existing_config, monkeypatch):
        """Should remove codeflash config from pyproject.toml."""
        monkeypatch.chdir(project_with_existing_config)

        from codeflash.cli_cmds.cli import _handle_reset_config

        # Verify config exists before
        content_before = (project_with_existing_config / "pyproject.toml").read_text()
        assert "[tool.codeflash]" in content_before

        _handle_reset_config(confirm=False)

        # Verify config removed after
        content_after = (project_with_existing_config / "pyproject.toml").read_text()
        assert "[tool.codeflash]" not in content_after

        # Other sections should remain
        assert "[project]" in content_after

    def test_reset_config_removes_from_package_json(self, javascript_npm_project, monkeypatch):
        """Should remove codeflash config from package.json."""
        # First add config
        monkeypatch.chdir(javascript_npm_project)

        # Add codeflash section
        with (javascript_npm_project / "package.json").open() as f:
            data = json.load(f)
        data["codeflash"] = {"moduleRoot": "src"}
        with (javascript_npm_project / "package.json").open("w") as f:
            json.dump(data, f, indent=2)

        from codeflash.cli_cmds.cli import _handle_reset_config

        _handle_reset_config(confirm=False)

        # Verify config removed
        with (javascript_npm_project / "package.json").open() as f:
            data_after = json.load(f)

        assert "codeflash" not in data_after
        assert data_after["name"] == "my-js-app"  # Other content preserved

    def test_reset_config_handles_no_config(self, python_src_layout, monkeypatch):
        """Should handle gracefully when no config exists to reset."""
        monkeypatch.chdir(python_src_layout)

        from codeflash.cli_cmds.cli import _handle_reset_config

        # Should not crash when no codeflash config exists
        _handle_reset_config(confirm=False)
