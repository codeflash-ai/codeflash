"""Tests for JavaScript/TypeScript configuration detection and parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from codeflash.code_utils.config_js import (
    PACKAGE_JSON_CACHE,
    PACKAGE_JSON_DATA_CACHE,
    clear_cache,
    detect_formatter,
    detect_language,
    detect_module_root,
    detect_test_runner,
    find_package_json,
    get_package_json_data,
    parse_package_json_config,
)


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear all caches before each test."""
    clear_cache()


class TestGetPackageJsonData:
    """Tests for get_package_json_data function."""

    def test_loads_valid_package_json(self, tmp_path: Path) -> None:
        """Should load and return valid package.json data."""
        package_json = tmp_path / "package.json"
        data = {"name": "test-project", "version": "1.0.0"}
        package_json.write_text(json.dumps(data))

        result = get_package_json_data(package_json)

        assert result == data

    def test_caches_loaded_data(self, tmp_path: Path) -> None:
        """Should cache package.json data after first load."""
        package_json = tmp_path / "package.json"
        data = {"name": "test-project"}
        package_json.write_text(json.dumps(data))

        # First call
        result1 = get_package_json_data(package_json)
        # Modify file
        package_json.write_text(json.dumps({"name": "modified"}))
        # Second call should return cached data
        result2 = get_package_json_data(package_json)

        assert result1 == result2 == data

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Should return None for invalid JSON."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{ invalid json }")

        result = get_package_json_data(package_json)

        assert result is None

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Should return None for non-existent file."""
        package_json = tmp_path / "package.json"

        result = get_package_json_data(package_json)

        assert result is None

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't restrict read access on Windows")
    def test_returns_none_for_unreadable_file(self, tmp_path: Path) -> None:
        """Should return None if file cannot be read."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{}")
        package_json.chmod(0o000)

        try:
            result = get_package_json_data(package_json)
            assert result is None
        finally:
            package_json.chmod(0o644)


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detects_typescript_with_tsconfig(self, tmp_path: Path) -> None:
        """Should detect TypeScript when tsconfig.json exists."""
        (tmp_path / "tsconfig.json").write_text("{}")

        result = detect_language(tmp_path)

        assert result == "typescript"

    def test_detects_javascript_without_tsconfig(self, tmp_path: Path) -> None:
        """Should detect JavaScript when no tsconfig.json exists."""
        result = detect_language(tmp_path)

        assert result == "javascript"

    def test_detects_typescript_with_complex_tsconfig(self, tmp_path: Path) -> None:
        """Should detect TypeScript even with complex tsconfig."""
        tsconfig = {"compilerOptions": {"target": "ES2020", "module": "commonjs"}, "include": ["src/**/*"]}
        (tmp_path / "tsconfig.json").write_text(json.dumps(tsconfig))

        result = detect_language(tmp_path)

        assert result == "typescript"


class TestDetectModuleRoot:
    """Tests for detect_module_root function."""

    def test_detects_from_exports_string(self, tmp_path: Path) -> None:
        """Should detect module root from exports string field."""
        (tmp_path / "lib").mkdir()
        package_data = {"exports": "./lib/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "lib"

    def test_detects_from_exports_object_dot(self, tmp_path: Path) -> None:
        """Should detect module root from exports object with '.' key."""
        (tmp_path / "dist").mkdir()
        package_data = {"exports": {".": "./dist/index.js"}}

        result = detect_module_root(tmp_path, package_data)

        assert result == "dist"

    def test_detects_from_exports_object_nested(self, tmp_path: Path) -> None:
        """Should detect module root from nested exports object."""
        (tmp_path / "src").mkdir()
        package_data = {"exports": {".": {"import": "./src/index.mjs", "require": "./src/index.cjs"}}}

        result = detect_module_root(tmp_path, package_data)

        assert result == "src"

    def test_detects_from_exports_import_key(self, tmp_path: Path) -> None:
        """Should detect from exports with direct import key."""
        (tmp_path / "esm").mkdir()
        package_data = {"exports": {"import": "./esm/index.js"}}

        result = detect_module_root(tmp_path, package_data)

        assert result == "esm"

    def test_detects_from_module_field(self, tmp_path: Path) -> None:
        """Should detect module root from module field (ESM entry)."""
        (tmp_path / "es").mkdir()
        package_data = {"module": "./es/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "es"

    def test_detects_from_main_field(self, tmp_path: Path) -> None:
        """Should detect module root from main field (CJS entry)."""
        (tmp_path / "lib").mkdir()
        package_data = {"main": "./lib/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "lib"

    def test_prefers_exports_over_module(self, tmp_path: Path) -> None:
        """Should prefer exports field over module field."""
        (tmp_path / "exports-dir").mkdir()
        (tmp_path / "module-dir").mkdir()
        package_data = {"exports": "./exports-dir/index.js", "module": "./module-dir/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "exports-dir"

    def test_prefers_module_over_main(self, tmp_path: Path) -> None:
        """Should prefer module field over main field."""
        (tmp_path / "esm").mkdir()
        (tmp_path / "cjs").mkdir()
        package_data = {"module": "./esm/index.js", "main": "./cjs/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "esm"

    def test_detects_src_directory_convention(self, tmp_path: Path) -> None:
        """Should detect src/ directory when no package.json fields point elsewhere."""
        (tmp_path / "src").mkdir()
        package_data = {}

        result = detect_module_root(tmp_path, package_data)

        assert result == "src"

    def test_falls_back_to_current_directory(self, tmp_path: Path) -> None:
        """Should fall back to '.' when nothing else matches."""
        package_data = {}

        result = detect_module_root(tmp_path, package_data)

        assert result == "."

    def test_ignores_nonexistent_directory_from_exports(self, tmp_path: Path) -> None:
        """Should ignore exports pointing to non-existent directory."""
        (tmp_path / "src").mkdir()
        package_data = {"exports": "./nonexistent/index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "src"

    def test_ignores_root_level_main(self, tmp_path: Path) -> None:
        """Should ignore main that points to root level file."""
        (tmp_path / "src").mkdir()
        package_data = {"main": "./index.js"}

        result = detect_module_root(tmp_path, package_data)

        assert result == "src"

    def test_handles_deeply_nested_exports(self, tmp_path: Path) -> None:
        """Should handle deeply nested export paths."""
        (tmp_path / "packages" / "core" / "dist").mkdir(parents=True)
        package_data = {"exports": {".": {"import": "./packages/core/dist/index.mjs"}}}

        result = detect_module_root(tmp_path, package_data)

        assert result == "packages/core/dist"

    def test_handles_empty_exports(self, tmp_path: Path) -> None:
        """Should handle empty exports gracefully."""
        (tmp_path / "src").mkdir()
        package_data = {"exports": {}}

        result = detect_module_root(tmp_path, package_data)

        assert result == "src"

    def test_handles_null_exports(self, tmp_path: Path) -> None:
        """Should handle null/None exports gracefully."""
        package_data = {"exports": None}

        result = detect_module_root(tmp_path, package_data)

        assert result == "."


class TestDetectTestRunner:
    """Tests for detect_test_runner function."""

    def test_detects_vitest_from_dev_dependencies(self, tmp_path: Path) -> None:
        """Should detect vitest from devDependencies."""
        package_data = {"devDependencies": {"vitest": "^1.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "vitest"

    def test_detects_jest_from_dev_dependencies(self, tmp_path: Path) -> None:
        """Should detect jest from devDependencies."""
        package_data = {"devDependencies": {"jest": "^29.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_detects_mocha_from_dev_dependencies(self, tmp_path: Path) -> None:
        """Should detect mocha from devDependencies."""
        package_data = {"devDependencies": {"mocha": "^10.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "mocha"

    def test_detects_from_dependencies(self, tmp_path: Path) -> None:
        """Should also check dependencies (not just devDependencies)."""
        package_data = {"dependencies": {"jest": "^29.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_prefers_vitest_over_jest(self, tmp_path: Path) -> None:
        """Should prefer vitest when both are present."""
        package_data = {"devDependencies": {"vitest": "^1.0.0", "jest": "^29.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "vitest"

    def test_prefers_jest_over_mocha(self, tmp_path: Path) -> None:
        """Should prefer jest over mocha."""
        package_data = {"devDependencies": {"jest": "^29.0.0", "mocha": "^10.0.0"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_detects_vitest_from_test_script(self, tmp_path: Path) -> None:
        """Should detect vitest from scripts.test."""
        package_data = {"scripts": {"test": "vitest run"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "vitest"

    def test_detects_jest_from_test_script(self, tmp_path: Path) -> None:
        """Should detect jest from scripts.test."""
        package_data = {"scripts": {"test": "jest --coverage"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_detects_mocha_from_test_script(self, tmp_path: Path) -> None:
        """Should detect mocha from scripts.test."""
        package_data = {"scripts": {"test": "mocha tests/**/*.js"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "mocha"

    def test_detects_from_npx_command(self, tmp_path: Path) -> None:
        """Should detect runner from npx command in test script."""
        package_data = {"scripts": {"test": "npx jest"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_detects_case_insensitive(self, tmp_path: Path) -> None:
        """Should detect runner case-insensitively from scripts."""
        package_data = {"scripts": {"test": "JEST --ci"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_prefers_deps_over_scripts(self, tmp_path: Path) -> None:
        """Should prefer devDependencies detection over scripts."""
        package_data = {"devDependencies": {"vitest": "^1.0.0"}, "scripts": {"test": "jest"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "vitest"

    def test_defaults_to_jest(self, tmp_path: Path) -> None:
        """Should default to jest when nothing is detected."""
        package_data = {}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_handles_complex_test_script(self, tmp_path: Path) -> None:
        """Should detect from complex test scripts."""
        package_data = {"scripts": {"test": "NODE_OPTIONS='--experimental-vm-modules' jest --coverage"}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_handles_missing_scripts(self, tmp_path: Path) -> None:
        """Should handle missing scripts gracefully."""
        package_data = {"name": "test"}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"

    def test_handles_non_string_test_script(self, tmp_path: Path) -> None:
        """Should handle non-string test script gracefully."""
        package_data = {"scripts": {"test": 123}}

        result = detect_test_runner(tmp_path, package_data)

        assert result == "jest"


class TestDetectFormatter:
    """Tests for detect_formatter function."""

    def test_detects_prettier_from_dev_dependencies(self, tmp_path: Path) -> None:
        """Should detect prettier from devDependencies."""
        package_data = {"devDependencies": {"prettier": "^3.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result == ["npx prettier --write $file"]

    def test_detects_eslint_from_dev_dependencies(self, tmp_path: Path) -> None:
        """Should detect eslint from devDependencies."""
        package_data = {"devDependencies": {"eslint": "^8.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result == ["npx eslint --fix $file"]

    def test_detects_from_dependencies(self, tmp_path: Path) -> None:
        """Should also check dependencies."""
        package_data = {"dependencies": {"prettier": "^3.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result == ["npx prettier --write $file"]

    def test_prefers_prettier_over_eslint(self, tmp_path: Path) -> None:
        """Should prefer prettier when both are present."""
        package_data = {"devDependencies": {"prettier": "^3.0.0", "eslint": "^8.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result == ["npx prettier --write $file"]

    def test_returns_none_when_no_formatter(self, tmp_path: Path) -> None:
        """Should return None when no formatter is detected."""
        package_data = {"devDependencies": {"typescript": "^5.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result is None

    def test_returns_none_for_empty_deps(self, tmp_path: Path) -> None:
        """Should return None for empty dependencies."""
        package_data = {}

        result = detect_formatter(tmp_path, package_data)

        assert result is None

    def test_detects_eslint_related_packages(self, tmp_path: Path) -> None:
        """Should detect eslint even with scoped packages."""
        package_data = {"devDependencies": {"eslint": "^8.0.0", "@eslint/js": "^8.0.0"}}

        result = detect_formatter(tmp_path, package_data)

        assert result == ["npx eslint --fix $file"]


class TestFindPackageJson:
    """Tests for find_package_json function."""

    def test_finds_explicit_package_json(self, tmp_path: Path) -> None:
        """Should find explicitly provided package.json path."""
        package_json = tmp_path / "package.json"
        package_json.write_text("{}")

        result = find_package_json(package_json)

        assert result == package_json

    def test_returns_none_for_wrong_filename(self, tmp_path: Path) -> None:
        """Should return None if explicit path is not package.json."""
        other_file = tmp_path / "other.json"
        other_file.write_text("{}")

        result = find_package_json(other_file)

        assert result is None

    def test_returns_none_for_nonexistent_explicit(self, tmp_path: Path) -> None:
        """Should return None if explicit package.json doesn't exist."""
        package_json = tmp_path / "package.json"

        result = find_package_json(package_json)

        assert result is None


class TestParsePackageJsonConfig:
    """Tests for parse_package_json_config function."""

    def test_parses_minimal_package_json(self, tmp_path: Path) -> None:
        """Should parse package.json without codeflash section."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "devDependencies": {"jest": "^29.0.0"}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, path = result
        assert config["language"] == "javascript"
        assert config["test_framework"] == "jest"
        assert config["pytest_cmd"] == "jest"
        assert path == package_json

    def test_parses_typescript_project(self, tmp_path: Path) -> None:
        """Should detect TypeScript project."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test"}))
        (tmp_path / "tsconfig.json").write_text("{}")

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["language"] == "typescript"

    def test_auto_detects_module_root(self, tmp_path: Path) -> None:
        """Should auto-detect module root from package.json."""
        (tmp_path / "src").mkdir()
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "main": "./src/index.js"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "src").resolve())

    def test_respects_module_root_override(self, tmp_path: Path) -> None:
        """Should respect moduleRoot override in codeflash config."""
        (tmp_path / "lib").mkdir()
        (tmp_path / "src").mkdir()
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps({"name": "test", "main": "./src/index.js", "codeflash": {"moduleRoot": "lib"}})
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "lib").resolve())

    def test_auto_detects_formatter(self, tmp_path: Path) -> None:
        """Should auto-detect formatter from devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "devDependencies": {"prettier": "^3.0.0"}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["formatter_cmds"] == ["npx prettier --write $file"]

    def test_respects_formatter_override(self, tmp_path: Path) -> None:
        """Should respect formatterCmds override."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test",
                    "devDependencies": {"prettier": "^3.0.0"},
                    "codeflash": {"formatterCmds": ["custom-formatter $file"]},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["formatter_cmds"] == ["custom-formatter $file"]

    def test_parses_ignore_paths(self, tmp_path: Path) -> None:
        """Should parse ignorePaths from codeflash config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": {"ignorePaths": ["dist", "node_modules"]}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert str((tmp_path / "dist").resolve()) in config["ignore_paths"]
        assert str((tmp_path / "node_modules").resolve()) in config["ignore_paths"]

    def test_parses_benchmarks_root(self, tmp_path: Path) -> None:
        """Should parse benchmarksRoot from codeflash config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": {"benchmarksRoot": "__benchmarks__"}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["benchmarks_root"] == str((tmp_path / "__benchmarks__").resolve())

    def test_parses_disable_telemetry(self, tmp_path: Path) -> None:
        """Should parse disableTelemetry from codeflash config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": {"disableTelemetry": True}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["disable_telemetry"] is True

    def test_defaults_disable_telemetry_to_false(self, tmp_path: Path) -> None:
        """Should default disableTelemetry to False."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["disable_telemetry"] is False

    def test_sets_backwards_compat_defaults(self, tmp_path: Path) -> None:
        """Should set backwards compatibility defaults."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["git_remote"] == "origin"
        assert config["disable_imports_sorting"] is False
        assert config["override_fixtures"] is False

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Should return None for invalid JSON."""
        package_json = tmp_path / "package.json"
        package_json.write_text("invalid json")

        result = parse_package_json_config(package_json)

        assert result is None

    def test_handles_non_dict_codeflash_config(self, tmp_path: Path) -> None:
        """Should handle non-dict codeflash section."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": "invalid"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        # Should use auto-detected/default values
        assert "language" in config

    def test_empty_formatter_when_none_detected(self, tmp_path: Path) -> None:
        """Should have empty formatter_cmds when no formatter detected."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "devDependencies": {"typescript": "^5.0.0"}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["formatter_cmds"] == []

    def test_parses_git_remote_from_config(self, tmp_path: Path) -> None:
        """Should parse gitRemote from codeflash config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": {"gitRemote": "upstream"}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["git_remote"] == "upstream"

    def test_defaults_git_remote_to_origin(self, tmp_path: Path) -> None:
        """Should default gitRemote to 'origin' when not specified."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["git_remote"] == "origin"

    def test_handles_empty_git_remote(self, tmp_path: Path) -> None:
        """Should handle empty gitRemote in config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "codeflash": {"gitRemote": ""}}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        # Empty string should be treated as the value (not defaulted to origin)
        assert config["git_remote"] == ""


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clears_both_caches(self, tmp_path: Path) -> None:
        """Should clear both path and data caches."""
        # Populate caches
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test"}))
        get_package_json_data(package_json)

        assert len(PACKAGE_JSON_DATA_CACHE) > 0

        clear_cache()

        assert len(PACKAGE_JSON_CACHE) == 0
        assert len(PACKAGE_JSON_DATA_CACHE) == 0


class TestRealWorldPackageJsonExamples:
    """Tests with real-world-like package.json configurations."""

    def test_nextjs_project(self, tmp_path: Path) -> None:
        """Should handle Next.js project configuration."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tsconfig.json").write_text("{}")
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "my-nextjs-app",
                    "scripts": {"test": "jest"},
                    "devDependencies": {"jest": "^29.0.0", "prettier": "^3.0.0", "typescript": "^5.0.0"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["language"] == "typescript"
        assert config["module_root"] == str((tmp_path / "src").resolve())
        assert config["test_framework"] == "jest"
        assert config["formatter_cmds"] == ["npx prettier --write $file"]

    def test_vite_react_project(self, tmp_path: Path) -> None:
        """Should handle Vite + React project configuration."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tsconfig.json").write_text("{}")
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "vite-react-app",
                    "type": "module",
                    "scripts": {"test": "vitest"},
                    "devDependencies": {"vitest": "^1.0.0", "eslint": "^8.0.0"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["language"] == "typescript"
        assert config["test_framework"] == "vitest"
        assert config["formatter_cmds"] == ["npx eslint --fix $file"]

    def test_library_with_exports(self, tmp_path: Path) -> None:
        """Should handle library with modern exports field."""
        (tmp_path / "dist").mkdir()
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "my-library",
                    "exports": {".": {"import": "./dist/index.mjs", "require": "./dist/index.cjs"}},
                    "devDependencies": {"vitest": "^1.0.0", "prettier": "^3.0.0"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "dist").resolve())

    def test_monorepo_package(self, tmp_path: Path) -> None:
        """Should handle monorepo package configuration."""
        (tmp_path / "packages" / "core" / "src").mkdir(parents=True)
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {"name": "@myorg/core", "main": "./packages/core/src/index.js", "devDependencies": {"jest": "^29.0.0"}}
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "packages/core/src").resolve())

    def test_node_cli_project(self, tmp_path: Path) -> None:
        """Should handle Node.js CLI project."""
        (tmp_path / "bin").mkdir()
        (tmp_path / "lib").mkdir()
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "my-cli",
                    "bin": {"my-cli": "./bin/cli.js"},
                    "main": "./lib/index.js",
                    "devDependencies": {"mocha": "^10.0.0"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "lib").resolve())
        assert config["test_framework"] == "mocha"

    def test_minimal_project(self, tmp_path: Path) -> None:
        """Should handle minimal package.json."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "minimal"}))

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["language"] == "javascript"
        assert config["module_root"] == str(tmp_path.resolve())
        assert config["test_framework"] == "jest"
        assert config["formatter_cmds"] == []

    def test_existing_codeflash_config_with_overrides(self, tmp_path: Path) -> None:
        """Should handle existing codeflash config with custom overrides."""
        (tmp_path / "custom-src").mkdir()
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "configured-project",
                    "devDependencies": {"jest": "^29.0.0", "prettier": "^3.0.0"},
                    "codeflash": {
                        "moduleRoot": "custom-src",
                        "formatterCmds": ["npx prettier --write --single-quote $file"],
                        "ignorePaths": ["dist", "coverage"],
                        "disableTelemetry": True,
                    },
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["module_root"] == str((tmp_path / "custom-src").resolve())
        assert config["formatter_cmds"] == ["npx prettier --write --single-quote $file"]
        assert len(config["ignore_paths"]) == 2
        assert config["disable_telemetry"] is True


class TestTestFrameworkConfigOverride:
    """Tests for explicit test-framework config override (matches Python's pyproject.toml)."""

    def test_test_framework_overrides_auto_detection(self, tmp_path: Path) -> None:
        """Should use test-framework from codeflash config instead of auto-detecting from devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^1.0.0"},
                    "codeflash": {"test-framework": "jest"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "jest"
        assert config["pytest_cmd"] == "jest"

    def test_explicit_vitest_config_with_jest_in_deps(self, tmp_path: Path) -> None:
        """Should use explicit vitest config even when jest is in devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"jest": "^29.0.0", "vitest": "^1.0.0"},
                    "codeflash": {"test-framework": "vitest"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "vitest"

    def test_explicit_mocha_overrides_vitest_and_jest(self, tmp_path: Path) -> None:
        """Should use explicit mocha config even when vitest and jest are in devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^1.0.0", "jest": "^29.0.0"},
                    "codeflash": {"test-framework": "mocha"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "mocha"

    def test_auto_detection_when_no_explicit_config(self, tmp_path: Path) -> None:
        """Should auto-detect test framework when no explicit config is provided."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^1.0.0"},
                    "codeflash": {"moduleRoot": "src"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "vitest"

    def test_empty_test_framework_falls_back_to_auto_detection(self, tmp_path: Path) -> None:
        """Should auto-detect when test-framework is empty string."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"jest": "^29.0.0"},
                    "codeflash": {"test-framework": ""},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "jest"

    def test_custom_test_framework_value(self, tmp_path: Path) -> None:
        """Should accept custom test framework values not in the standard list."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^1.0.0"},
                    "codeflash": {"test-framework": "ava"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "ava"

    def test_pytest_cmd_matches_test_framework_with_override(self, tmp_path: Path) -> None:
        """Should set pytest_cmd to match test_framework when using explicit config."""
        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^1.0.0"},
                    "codeflash": {"test-framework": "jest"},
                }
            )
        )

        result = parse_package_json_config(package_json)

        assert result is not None
        config, _ = result
        assert config["test_framework"] == "jest"
        assert config["pytest_cmd"] == "jest"
        assert config["test_framework"] == config["pytest_cmd"]
