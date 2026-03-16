"""Integration tests for JavaScriptFunctionOptimizer validation methods."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from codeflash.code_utils.config_js import PACKAGE_JSON_DATA_CACHE
from codeflash.either import Success, is_successful
from codeflash.languages.javascript.function_optimizer import JavaScriptFunctionOptimizer

if TYPE_CHECKING:
    from pathlib import Path


def _make_optimizer(
    tmp_path: Path, source_rel: str = "src/index.js", module_root_rel: str = "src"
) -> JavaScriptFunctionOptimizer:
    """Build a JavaScriptFunctionOptimizer with minimal mocked internals."""
    project_root = tmp_path / "project"
    project_root.mkdir(exist_ok=True)
    src = project_root / "src"
    src.mkdir(exist_ok=True)
    source_file = project_root / source_rel
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("export function foo() {}", encoding="utf-8")
    (project_root / "package.json").write_text("{}", encoding="utf-8")

    module_root = (project_root / module_root_rel).resolve()

    fto = MagicMock()
    fto.file_path = source_file.resolve()
    fto.function_name = "foo"
    fto.qualified_name_with_modules_from_root.return_value = "src/index.foo"

    args = Namespace(module_root=module_root, project_root=project_root.resolve(), no_gen_tests=False)

    opt = object.__new__(JavaScriptFunctionOptimizer)
    opt.function_to_optimize = fto
    opt.project_root = project_root.resolve()
    opt.args = args

    return opt


class TestTryCorrectModuleRoot:
    def test_noop_when_config_is_correct(self, tmp_path: Path) -> None:
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel="src")
        PACKAGE_JSON_DATA_CACHE.clear()

        corrected = opt.try_correct_module_root()
        assert corrected is False

    def test_corrects_module_root_and_updates_package_json(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel=".")
        PACKAGE_JSON_DATA_CACHE.clear()

        # The source file is in src/ but module_root points to project root,
        # which is valid because src/ IS within project root.
        # To test correction, we need module_root to NOT contain the source file.
        wrong_root = (project_root / "lib").resolve()
        wrong_root.mkdir(exist_ok=True)
        opt.args.module_root = wrong_root

        corrected = opt.try_correct_module_root()
        assert corrected is True

        # Module root should now be src/
        expected = (project_root / "src").resolve()
        assert opt.args.module_root == expected

        # package.json should have been updated
        pkg = json.loads((project_root / "package.json").read_text(encoding="utf-8"))
        assert pkg["codeflash"]["moduleRoot"] == "src"

    def test_returns_false_when_inferred_root_doesnt_contain_source(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel="src")
        PACKAGE_JSON_DATA_CACHE.clear()

        # Point module root somewhere invalid and mock infer to return something that also doesn't contain source
        wrong_root = (project_root / "lib").resolve()
        wrong_root.mkdir(exist_ok=True)
        opt.args.module_root = wrong_root

        with patch(
            "codeflash.languages.javascript.function_optimizer.infer_js_module_root",
            return_value=(project_root / "other").resolve(),
        ):
            corrected = opt.try_correct_module_root()
        assert corrected is False


class TestCanBeOptimized:
    def test_returns_failure_when_source_outside_module_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel="src")
        PACKAGE_JSON_DATA_CACHE.clear()

        # Point module root somewhere that doesn't contain the source
        other = (project_root / "other").resolve()
        other.mkdir(exist_ok=True)
        opt.args.module_root = other

        # Also mock infer to return 'other' so correction still fails
        with patch("codeflash.languages.javascript.function_optimizer.infer_js_module_root", return_value=other):
            result = opt.can_be_optimized()

        assert not is_successful(result)
        assert "not within module root" in result.failure()

    def test_delegates_to_super_when_valid(self, tmp_path: Path) -> None:
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel="src")
        PACKAGE_JSON_DATA_CACHE.clear()

        mock_result = Success((False, MagicMock(), {}))
        with patch.object(JavaScriptFunctionOptimizer.__bases__[0], "can_be_optimized", return_value=mock_result):
            result = opt.can_be_optimized()

        assert is_successful(result)

    def test_auto_corrects_then_succeeds(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        opt = _make_optimizer(tmp_path, source_rel="src/index.js", module_root_rel="src")
        PACKAGE_JSON_DATA_CACHE.clear()

        # Start with wrong module root
        wrong_root = (project_root / "lib").resolve()
        wrong_root.mkdir(exist_ok=True)
        opt.args.module_root = wrong_root

        mock_result = Success((False, MagicMock(), {}))
        with patch.object(JavaScriptFunctionOptimizer.__bases__[0], "can_be_optimized", return_value=mock_result):
            result = opt.can_be_optimized()

        assert is_successful(result)
        # Verify module root was corrected
        expected = (project_root / "src").resolve()
        assert opt.args.module_root == expected
