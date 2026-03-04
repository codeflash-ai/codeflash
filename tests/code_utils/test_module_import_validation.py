from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomlkit

from codeflash.code_utils.code_utils import infer_module_root_from_file, validate_module_import


class TestValidateModuleImport:
    def test_known_stdlib_module(self, tmp_path: Path) -> None:
        ok, err = validate_module_import("json", tmp_path)
        assert ok is True
        assert err == ""

    def test_nonexistent_module(self, tmp_path: Path) -> None:
        ok, err = validate_module_import("totally_nonexistent_module_xyz_123", tmp_path)
        assert ok is False
        assert err != ""

    def test_finds_module_in_project_root(self, tmp_path: Path) -> None:
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "utils.py").write_text("x = 1\n", encoding="utf-8")

        ok, err = validate_module_import("mypkg.utils", tmp_path)
        assert ok is True
        assert err == ""

    def test_project_root_not_left_in_sys_path(self, tmp_path: Path) -> None:
        root_str = str(tmp_path)
        assert root_str not in sys.path
        validate_module_import("nonexistent_mod", tmp_path)
        assert root_str not in sys.path

    def test_project_root_preserved_if_already_in_sys_path(self, tmp_path: Path) -> None:
        root_str = str(tmp_path)
        sys.path.insert(0, root_str)
        try:
            validate_module_import("json", tmp_path)
            assert root_str in sys.path
        finally:
            sys.path.remove(root_str)

    def test_find_spec_returns_none(self, tmp_path: Path) -> None:
        with patch("codeflash.code_utils.code_utils.find_spec", return_value=None) as mock_fs:
            ok, err = validate_module_import("some.mod", tmp_path)
            mock_fs.assert_called_once_with("some.mod")
        assert ok is False
        assert "not found" in err

    def test_find_spec_raises_module_not_found(self, tmp_path: Path) -> None:
        with patch(
            "codeflash.code_utils.code_utils.find_spec",
            side_effect=ModuleNotFoundError("No module named 'boom'"),
        ):
            ok, err = validate_module_import("boom", tmp_path)
        assert ok is False
        assert "boom" in err

    def test_find_spec_raises_generic_exception(self, tmp_path: Path) -> None:
        with patch(
            "codeflash.code_utils.code_utils.find_spec",
            side_effect=RuntimeError("something broke"),
        ):
            ok, err = validate_module_import("broken.mod", tmp_path)
        assert ok is False
        assert "something broke" in err

    def test_sys_path_cleaned_on_exception(self, tmp_path: Path) -> None:
        root_str = str(tmp_path)
        assert root_str not in sys.path
        with patch("codeflash.code_utils.code_utils.find_spec", side_effect=RuntimeError("boom")):
            validate_module_import("mod", tmp_path)
        assert root_str not in sys.path


class TestInferModuleRootFromFile:
    def test_single_package(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        mod = pkg / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == pkg.resolve()

    def test_nested_package_returns_top_level(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (sub / "__init__.py").write_text("", encoding="utf-8")
        mod = sub / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == pkg.resolve()

    def test_deeply_nested_package(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = a / "b"
        c = b / "c"
        c.mkdir(parents=True)
        for d in (a, b, c):
            (d / "__init__.py").write_text("", encoding="utf-8")
        mod = c / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == a.resolve()

    def test_no_init_files_returns_parent_dir(self, tmp_path: Path) -> None:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        mod = scripts / "run.py"
        mod.write_text("print('hi')\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == scripts.resolve()

    def test_gap_in_init_chain(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (inner / "__init__.py").write_text("", encoding="utf-8")
        mod = inner / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == inner.resolve()

    def test_file_directly_in_pyproject_dir(self, tmp_path: Path) -> None:
        mod = tmp_path / "standalone.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == tmp_path.resolve()

    def test_src_layout(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        mod = pkg / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        result = infer_module_root_from_file(mod, tmp_path)
        assert result is not None
        assert result.resolve() == pkg.resolve()


class TestTryCorrectModuleRoot:
    def _make_optimizer_stub(
        self,
        file_path: Path,
        module_root: Path,
        project_root: Path,
        original_module_path: str = "pkg.mod",
    ) -> MagicMock:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        optimizer = MagicMock(spec=PythonFunctionOptimizer)
        optimizer.function_to_optimize = MagicMock()
        optimizer.function_to_optimize.file_path = file_path
        optimizer.args = MagicMock()
        optimizer.args.module_root = module_root
        optimizer.args.project_root = project_root
        optimizer.project_root = project_root
        optimizer.original_module_path = original_module_path
        return optimizer

    def test_returns_false_when_pyproject_not_found(self, tmp_path: Path) -> None:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        mod = tmp_path / "pkg" / "mod.py"
        mod.parent.mkdir()
        mod.write_text("x = 1\n", encoding="utf-8")
        optimizer = self._make_optimizer_stub(mod, tmp_path / "pkg", tmp_path)

        with patch(
            "codeflash.languages.python.function_optimizer.find_pyproject_toml",
            side_effect=ValueError("not found"),
        ):
            result = PythonFunctionOptimizer.try_correct_module_root(optimizer)
        assert result is False

    def test_returns_false_when_inferred_same_as_current(self, tmp_path: Path) -> None:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        mod = pkg / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.codeflash]\nmodule-root = "pkg"\n', encoding="utf-8")

        optimizer = self._make_optimizer_stub(mod, pkg, tmp_path)

        with patch(
            "codeflash.languages.python.function_optimizer.find_pyproject_toml",
            return_value=pyproject,
        ):
            result = PythonFunctionOptimizer.try_correct_module_root(optimizer)
        assert result is False

    def test_corrects_module_root_and_updates_pyproject(self, tmp_path: Path) -> None:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        pkg = tmp_path / "pkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (sub / "__init__.py").write_text("", encoding="utf-8")
        mod = sub / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.codeflash]\nmodule-root = "pkg/sub"\n', encoding="utf-8")

        optimizer = self._make_optimizer_stub(
            file_path=mod,
            module_root=sub,
            project_root=tmp_path,
            original_module_path="sub.mod",
        )

        with (
            patch(
                "codeflash.languages.python.function_optimizer.find_pyproject_toml",
                return_value=pyproject,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.project_root_from_module_root",
                return_value=tmp_path,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.module_name_from_file_path",
                return_value="pkg.sub.mod",
            ),
            patch(
                "codeflash.languages.python.function_optimizer.validate_module_import",
                return_value=(True, ""),
            ),
        ):
            result = PythonFunctionOptimizer.try_correct_module_root(optimizer)

        assert result is True
        data = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
        assert data["tool"]["codeflash"]["module-root"] == os.path.relpath(pkg.resolve(), tmp_path)

    def test_returns_false_when_import_validation_fails(self, tmp_path: Path) -> None:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        pkg = tmp_path / "pkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (sub / "__init__.py").write_text("", encoding="utf-8")
        mod = sub / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.codeflash]\nmodule-root = "pkg/sub"\n', encoding="utf-8")

        optimizer = self._make_optimizer_stub(
            file_path=mod,
            module_root=sub,
            project_root=tmp_path,
        )

        with (
            patch(
                "codeflash.languages.python.function_optimizer.find_pyproject_toml",
                return_value=pyproject,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.project_root_from_module_root",
                return_value=tmp_path,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.module_name_from_file_path",
                return_value="pkg.sub.mod",
            ),
            patch(
                "codeflash.languages.python.function_optimizer.validate_module_import",
                return_value=(False, "Module not found"),
            ),
        ):
            result = PythonFunctionOptimizer.try_correct_module_root(optimizer)

        assert result is False

    def test_returns_false_when_module_name_raises(self, tmp_path: Path) -> None:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        pkg = tmp_path / "pkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (sub / "__init__.py").write_text("", encoding="utf-8")
        mod = sub / "mod.py"
        mod.write_text("x = 1\n", encoding="utf-8")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.codeflash]\nmodule-root = "pkg/sub"\n', encoding="utf-8")

        optimizer = self._make_optimizer_stub(
            file_path=mod,
            module_root=sub,
            project_root=tmp_path,
        )

        with (
            patch(
                "codeflash.languages.python.function_optimizer.find_pyproject_toml",
                return_value=pyproject,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.project_root_from_module_root",
                return_value=tmp_path,
            ),
            patch(
                "codeflash.languages.python.function_optimizer.module_name_from_file_path",
                side_effect=ValueError("cannot derive module name"),
            ),
        ):
            result = PythonFunctionOptimizer.try_correct_module_root(optimizer)

        assert result is False
