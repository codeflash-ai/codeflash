from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from codeflash.either import is_successful
from codeflash.languages.java.function_optimizer import JavaFunctionOptimizer


def _make_optimizer(tmp_path: Path, source_file: Path, module_root: Path | None = None) -> JavaFunctionOptimizer:
    """Create a JavaFunctionOptimizer with minimal mocked dependencies."""
    optimizer = object.__new__(JavaFunctionOptimizer)
    optimizer.project_root = tmp_path.resolve()

    fto = MagicMock()
    fto.file_path = source_file.resolve()
    fto.language = "java"
    fto.function_name = "doSomething"
    optimizer.function_to_optimize = fto

    args = Namespace(module_root=module_root or tmp_path, project_root=tmp_path, no_gen_tests=False)
    optimizer.args = args
    return optimizer


class TestTryCorrectModuleRoot:
    def test_noop_when_config_is_correct(self, tmp_path: Path) -> None:
        project_root = tmp_path
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        source_file = pkg_dir / "Foo.java"
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        optimizer = _make_optimizer(project_root, source_file, module_root=src_root)

        result = optimizer.try_correct_module_root()
        assert result is True
        # Module root should remain unchanged
        assert Path(optimizer.args.module_root).resolve() == src_root.resolve()

    def test_corrects_module_root_and_updates_config(self, tmp_path: Path) -> None:
        project_root = tmp_path
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        source_file = pkg_dir / "Foo.java"
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        # Start with wrong module root (project root instead of src/main/java)
        optimizer = _make_optimizer(project_root, source_file, module_root=project_root)

        # Mock the config file update since we don't want to actually write config
        with patch.object(optimizer, "_update_config_module_root"):
            result = optimizer.try_correct_module_root()

        assert result is True
        assert Path(optimizer.args.module_root).resolve() == src_root.resolve()

    def test_returns_false_when_inferred_root_doesnt_contain_source(self, tmp_path: Path) -> None:
        project_root = tmp_path
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        # Source file is outside any reasonable source root
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        source_file = outside_dir / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        # Module root is wrong and doesn't contain the source file
        wrong_root = project_root / "src" / "main" / "java"
        wrong_root.mkdir(parents=True)
        optimizer = _make_optimizer(project_root, source_file, module_root=wrong_root)
        # Also set the fto.file_path to a file outside the project root
        # This ensures the validation fails and infer can't fix it
        optimizer.function_to_optimize.file_path = source_file.resolve()

        result = optimizer.try_correct_module_root()
        assert result is False


class TestCanBeOptimized:
    def test_returns_failure_when_source_outside_module_root(self, tmp_path: Path) -> None:
        project_root = tmp_path
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        module_root = project_root / "src" / "main" / "java"
        module_root.mkdir(parents=True)

        # Source file is outside the module root
        source_file = project_root / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        optimizer = _make_optimizer(project_root, source_file, module_root=module_root)

        result = optimizer.can_be_optimized()
        assert not is_successful(result)
        assert "Java module validation failed" in result.failure()

    def test_delegates_to_super_when_valid(self, tmp_path: Path) -> None:
        project_root = tmp_path
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        source_file = pkg_dir / "Foo.java"
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        optimizer = _make_optimizer(project_root, source_file, module_root=src_root)

        # Mock super().can_be_optimized() since it has many dependencies
        with patch.object(
            JavaFunctionOptimizer.__bases__[0], "can_be_optimized", return_value=MagicMock()
        ) as mock_super:
            result = optimizer.can_be_optimized()
            mock_super.assert_called_once()
