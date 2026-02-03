"""Tests for the universal project detector."""

import json

from codeflash.setup.detector import (
    _detect_js_formatter,
    _detect_js_module_root,
    _detect_js_test_runner,
    _detect_language,
    _detect_python_formatter,
    _detect_python_module_root,
    _detect_python_test_runner,
    _detect_tests_root,
    _find_project_root,
    detect_project,
    has_existing_config,
    is_build_output_dir,
)


class TestFindProjectRoot:
    """Tests for _find_project_root function."""

    def test_finds_git_directory(self, tmp_path):
        """Should find project root by .git directory."""
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "deep"
        subdir.mkdir(parents=True)

        result = _find_project_root(subdir)
        assert result == tmp_path

    def test_finds_pyproject_toml(self, tmp_path):
        """Should find project root by pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        subdir = tmp_path / "src"
        subdir.mkdir()

        result = _find_project_root(subdir)
        assert result == tmp_path

    def test_finds_package_json(self, tmp_path):
        """Should find project root by package.json."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        subdir = tmp_path / "lib"
        subdir.mkdir()

        result = _find_project_root(subdir)
        assert result == tmp_path

    def test_returns_none_when_no_markers(self, tmp_path):
        """Should return None when no project markers found."""
        subdir = tmp_path / "orphan"
        subdir.mkdir()

        result = _find_project_root(subdir)
        # Will walk up to filesystem root and not find anything
        assert result is None or result == tmp_path


class TestDetectLanguage:
    """Tests for _detect_language function."""

    def test_detects_typescript_from_tsconfig(self, tmp_path):
        """Should detect TypeScript when tsconfig.json exists."""
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "package.json").write_text('{"name": "test"}')

        lang, confidence, detail = _detect_language(tmp_path)
        assert lang == "typescript"
        assert confidence == 1.0
        assert "tsconfig.json" in detail

    def test_detects_python_from_pyproject(self, tmp_path):
        """Should detect Python when pyproject.toml exists."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        lang, confidence, detail = _detect_language(tmp_path)
        assert lang == "python"
        assert confidence == 1.0
        assert "pyproject.toml" in detail

    def test_detects_python_from_setup_py(self, tmp_path):
        """Should detect Python when setup.py exists."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        lang, confidence, detail = _detect_language(tmp_path)
        assert lang == "python"
        assert confidence == 1.0
        assert "setup.py" in detail

    def test_detects_javascript_from_package_json(self, tmp_path):
        """Should detect JavaScript when only package.json exists."""
        (tmp_path / "package.json").write_text('{"name": "test"}')

        lang, confidence, detail = _detect_language(tmp_path)
        assert lang == "javascript"
        assert confidence == 0.9
        assert "package.json" in detail

    def test_defaults_to_python(self, tmp_path):
        """Should default to Python when no markers found."""
        lang, confidence, detail = _detect_language(tmp_path)
        assert lang == "python"
        assert confidence < 0.5  # Low confidence


class TestDetectModuleRoot:
    """Tests for module root detection."""

    def test_python_detects_src_layout(self, tmp_path):
        """Should detect src/ layout for Python."""
        src_dir = tmp_path / "src" / "mypackage"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("")

        module_root, detail = _detect_python_module_root(tmp_path)
        assert module_root == src_dir
        assert module_root.name == "mypackage"
        assert module_root.parent.name == "src"

    def test_python_detects_package_at_root(self, tmp_path):
        """Should detect package at project root."""
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        module_root, detail = _detect_python_module_root(tmp_path)
        assert module_root == pkg_dir

    def test_python_uses_pyproject_name(self, tmp_path):
        """Should use project name from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "myapp"')
        pkg_dir = tmp_path / "myapp"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        module_root, detail = _detect_python_module_root(tmp_path)
        assert module_root == pkg_dir
        assert "pyproject.toml" in detail

    def test_js_detects_from_exports(self, tmp_path):
        """Should detect module root from package.json exports when no common src dir exists."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "exports": {".": "./packages/core/index.js"}
        }))
        (tmp_path / "packages" / "core").mkdir(parents=True)

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "packages" / "core"
        assert "exports" in detail

    def test_js_detects_src_convention(self, tmp_path):
        """Should detect src/ directory for JS."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "src").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "src"

    def test_js_prefers_src_over_build_src(self, tmp_path):
        """Should prefer src/ over build/src/ even when package.json points to build/."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "build/src/index.js",
            "module": "build/src/index.js"
        }))
        (tmp_path / "src").mkdir()
        (tmp_path / "build" / "src").mkdir(parents=True)

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "src"
        assert "src/ directory" in detail

    def test_js_skips_build_dir_from_main(self, tmp_path):
        """Should skip build output directories from package.json main field."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "build/index.js"
        }))
        (tmp_path / "build").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path
        assert "project root" in detail

    def test_js_skips_dist_dir_from_exports(self, tmp_path):
        """Should skip dist output directories from package.json exports field."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "exports": {".": "./dist/index.js"}
        }))
        (tmp_path / "dist").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path
        assert "project root" in detail

    def test_js_skips_out_dir_from_module(self, tmp_path):
        """Should skip out output directories from package.json module field."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "module": "out/esm/index.js"
        }))
        (tmp_path / "out" / "esm").mkdir(parents=True)

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path
        assert "project root" in detail

    def test_js_prefers_lib_over_build_dir(self, tmp_path):
        """Should prefer lib/ over build output directories."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "dist/index.js"
        }))
        (tmp_path / "lib").mkdir()
        (tmp_path / "dist").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "lib"
        assert "lib/ directory" in detail

    def test_js_prefers_source_over_build_dir(self, tmp_path):
        """Should prefer source/ over build output directories."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "build/index.js"
        }))
        (tmp_path / "source").mkdir()
        (tmp_path / "build").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "source"
        assert "source/ directory" in detail

    def test_js_falls_back_to_valid_exports_path(self, tmp_path):
        """Should use exports path when no common source dirs exist and path is not build output."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "exports": {".": "./packages/core/index.js"}
        }))
        (tmp_path / "packages" / "core").mkdir(parents=True)

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "packages" / "core"
        assert "exports" in detail

    def test_js_falls_back_to_valid_main_path(self, tmp_path):
        """Should use main path when no common source dirs exist and path is not build output."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "packages/main/index.js"
        }))
        (tmp_path / "packages" / "main").mkdir(parents=True)

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "packages" / "main"
        assert "main" in detail

    def test_js_falls_back_to_valid_module_path(self, tmp_path):
        """Should use module path when no common source dirs exist and path is not build output."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "module": "esm/index.js"
        }))
        (tmp_path / "esm").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path / "esm"
        assert "module" in detail

    def test_js_returns_project_root_when_all_paths_are_build_output(self, tmp_path):
        """Should return project root when all package.json paths point to build outputs."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "main": "dist/cjs/index.js",
            "module": "dist/esm/index.js",
            "exports": {".": "./build/index.js"}
        }))
        (tmp_path / "dist" / "cjs").mkdir(parents=True)
        (tmp_path / "dist" / "esm").mkdir(parents=True)
        (tmp_path / "build").mkdir()

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path
        assert "project root" in detail

    def test_js_handles_malformed_package_json(self, tmp_path):
        """Should handle malformed package.json gracefully."""
        (tmp_path / "package.json").write_text("{ invalid json }")

        module_root, detail = _detect_js_module_root(tmp_path)
        assert module_root == tmp_path
        assert "project root" in detail


class TestIsBuildOutputDir:
    """Tests for is_build_output_dir function."""

    def test_detects_build_dir(self):
        """Should detect build/ as build output."""
        from pathlib import Path
        assert is_build_output_dir(Path("build"))
        assert is_build_output_dir(Path("build/src"))
        assert is_build_output_dir(Path("build/src/index.js"))

    def test_detects_dist_dir(self):
        """Should detect dist/ as build output."""
        from pathlib import Path
        assert is_build_output_dir(Path("dist"))
        assert is_build_output_dir(Path("dist/esm"))
        assert is_build_output_dir(Path("dist/cjs/index.js"))

    def test_detects_out_dir(self):
        """Should detect out/ as build output."""
        from pathlib import Path
        assert is_build_output_dir(Path("out"))
        assert is_build_output_dir(Path("out/src"))

    def test_detects_next_dir(self):
        """Should detect .next/ as build output."""
        from pathlib import Path
        assert is_build_output_dir(Path(".next"))
        assert is_build_output_dir(Path(".next/static"))

    def test_detects_nuxt_dir(self):
        """Should detect .nuxt/ as build output."""
        from pathlib import Path
        assert is_build_output_dir(Path(".nuxt"))
        assert is_build_output_dir(Path(".nuxt/dist"))

    def test_detects_nested_build_dir(self):
        """Should detect build dir nested in path."""
        from pathlib import Path
        assert is_build_output_dir(Path("packages/build/index.js"))
        assert is_build_output_dir(Path("foo/dist/bar"))

    def test_does_not_detect_src(self):
        """Should not detect src/ as build output."""
        from pathlib import Path
        assert not is_build_output_dir(Path("src"))
        assert not is_build_output_dir(Path("src/index.js"))

    def test_does_not_detect_lib(self):
        """Should not detect lib/ as build output."""
        from pathlib import Path
        assert not is_build_output_dir(Path("lib"))
        assert not is_build_output_dir(Path("lib/utils"))

    def test_does_not_detect_source(self):
        """Should not detect source/ as build output."""
        from pathlib import Path
        assert not is_build_output_dir(Path("source"))

    def test_does_not_detect_packages(self):
        """Should not detect packages/ as build output."""
        from pathlib import Path
        assert not is_build_output_dir(Path("packages"))
        assert not is_build_output_dir(Path("packages/core"))

    def test_does_not_detect_similar_names(self):
        """Should not detect directories with similar but different names."""
        from pathlib import Path
        assert not is_build_output_dir(Path("builder"))
        assert not is_build_output_dir(Path("distribution"))
        assert not is_build_output_dir(Path("output"))


class TestDetectTestsRoot:
    """Tests for tests root detection."""

    def test_detects_tests_directory(self, tmp_path):
        """Should detect tests/ directory."""
        (tmp_path / "tests").mkdir()

        tests_root, detail = _detect_tests_root(tmp_path, "python")
        assert tests_root == tmp_path / "tests"

    def test_detects_test_directory(self, tmp_path):
        """Should detect test/ directory."""
        (tmp_path / "test").mkdir()

        tests_root, detail = _detect_tests_root(tmp_path, "python")
        assert tests_root == tmp_path / "test"

    def test_detects_dunder_tests(self, tmp_path):
        """Should detect __tests__/ directory (JS convention)."""
        (tmp_path / "__tests__").mkdir()

        tests_root, detail = _detect_tests_root(tmp_path, "javascript")
        assert tests_root == tmp_path / "__tests__"

    def test_returns_none_when_not_found(self, tmp_path):
        """Should return None when no tests directory found."""
        tests_root, detail = _detect_tests_root(tmp_path, "python")
        assert tests_root is None


class TestDetectTestRunner:
    """Tests for test runner detection."""

    def test_python_detects_pytest_from_ini(self, tmp_path):
        """Should detect pytest from pytest.ini."""
        (tmp_path / "pytest.ini").write_text("[pytest]")

        runner, detail = _detect_python_test_runner(tmp_path)
        assert runner == "pytest"

    def test_python_detects_pytest_from_conftest(self, tmp_path):
        """Should detect pytest from conftest.py."""
        (tmp_path / "conftest.py").write_text("import pytest")

        runner, detail = _detect_python_test_runner(tmp_path)
        assert runner == "pytest"

    def test_js_detects_jest_from_deps(self, tmp_path):
        """Should detect jest from devDependencies."""
        (tmp_path / "package.json").write_text(json.dumps({
            "devDependencies": {"jest": "^29.0.0"}
        }))

        runner, detail = _detect_js_test_runner(tmp_path)
        assert runner == "jest"

    def test_js_detects_vitest_from_deps(self, tmp_path):
        """Should detect vitest from devDependencies (preferred over jest)."""
        (tmp_path / "package.json").write_text(json.dumps({
            "devDependencies": {"vitest": "^1.0.0", "jest": "^29.0.0"}
        }))

        runner, detail = _detect_js_test_runner(tmp_path)
        assert runner == "vitest"

    def test_js_detects_from_config_file(self, tmp_path):
        """Should detect test runner from config file."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "vitest.config.js").write_text("export default {}")

        runner, detail = _detect_js_test_runner(tmp_path)
        assert runner == "vitest"


class TestDetectFormatter:
    """Tests for formatter detection."""

    def test_python_detects_ruff(self, tmp_path):
        """Should detect ruff from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120")

        formatter, detail = _detect_python_formatter(tmp_path)
        assert any("ruff" in cmd for cmd in formatter)

    def test_python_detects_black(self, tmp_path):
        """Should detect black from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[tool.black]\nline-length = 88")

        formatter, detail = _detect_python_formatter(tmp_path)
        assert any("black" in cmd for cmd in formatter)

    def test_js_detects_prettier(self, tmp_path):
        """Should detect prettier from config file."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / ".prettierrc").write_text("{}")

        formatter, detail = _detect_js_formatter(tmp_path)
        assert any("prettier" in cmd for cmd in formatter)

    def test_js_detects_prettier_from_deps(self, tmp_path):
        """Should detect prettier from devDependencies."""
        (tmp_path / "package.json").write_text(json.dumps({
            "devDependencies": {"prettier": "^3.0.0"}
        }))

        formatter, detail = _detect_js_formatter(tmp_path)
        assert any("prettier" in cmd for cmd in formatter)


class TestDetectProject:
    """Integration tests for detect_project function."""

    def test_detects_python_project(self, tmp_path):
        """Should correctly detect a Python project."""
        # Create Python project structure
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "myapp"\n\n[tool.ruff]\nline-length = 120'
        )
        (tmp_path / "myapp").mkdir()
        (tmp_path / "myapp" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / ".git").mkdir()

        detected = detect_project(tmp_path)

        assert detected.language == "python"
        assert detected.project_root == tmp_path
        assert detected.module_root == tmp_path / "myapp"
        assert detected.tests_root == tmp_path / "tests"
        assert detected.test_runner == "pytest"
        assert any("ruff" in cmd for cmd in detected.formatter_cmds)

    def test_detects_javascript_project(self, tmp_path):
        """Should correctly detect a JavaScript project."""
        # Create JS project structure
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "devDependencies": {"jest": "^29.0.0", "prettier": "^3.0.0"}
        }))
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / ".git").mkdir()

        detected = detect_project(tmp_path)

        assert detected.language == "javascript"
        assert detected.project_root == tmp_path
        assert detected.module_root == tmp_path / "src"
        assert detected.tests_root == tmp_path / "tests"
        assert detected.test_runner == "jest"
        assert any("prettier" in cmd for cmd in detected.formatter_cmds)

    def test_detects_typescript_project(self, tmp_path):
        """Should correctly detect a TypeScript project."""
        # Create TS project structure
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "devDependencies": {"vitest": "^1.0.0", "typescript": "^5.0.0"}
        }))
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / ".git").mkdir()

        detected = detect_project(tmp_path)

        assert detected.language == "typescript"
        assert detected.test_runner == "vitest"

    def test_to_display_dict(self, tmp_path):
        """Should generate display dictionary correctly."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "test").mkdir()
        (tmp_path / "test" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()

        detected = detect_project(tmp_path)
        display = detected.to_display_dict()

        assert "Language" in display
        assert "Module root" in display
        assert "Test runner" in display


class TestHasExistingConfig:
    """Tests for has_existing_config function."""

    def test_detects_pyproject_config(self, tmp_path):
        """Should detect config in pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text(
            '[tool.codeflash]\nmodule-root = "src"'
        )

        has_config, config_type = has_existing_config(tmp_path)
        assert has_config is True
        assert config_type == "pyproject.toml"

    def test_detects_package_json_config(self, tmp_path):
        """Should detect config in package.json."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "codeflash": {"moduleRoot": "src"}
        }))

        has_config, config_type = has_existing_config(tmp_path)
        assert has_config is True
        assert config_type == "package.json"

    def test_returns_false_when_no_config(self, tmp_path):
        """Should return False when no codeflash config exists."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        has_config, config_type = has_existing_config(tmp_path)
        assert has_config is False
        assert config_type is None

    def test_returns_false_for_empty_directory(self, tmp_path):
        """Should return False for empty directory."""
        has_config, config_type = has_existing_config(tmp_path)
        assert has_config is False
        assert config_type is None
