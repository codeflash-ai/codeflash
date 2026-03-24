from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash_core.config import TestConfig


@pytest.fixture
def js_support() -> JavaScriptSupport:
    return JavaScriptSupport()


def make_test_config(project_root: Path) -> TestConfig:
    return TestConfig(
        tests_root=project_root / "tests",
        project_root=project_root,
        tests_project_rootdir=project_root,
    )


class TestSetupTestConfigNodeModulesSymlink:
    def test_symlinks_node_modules_when_worktree_and_original_exists(self, js_support: JavaScriptSupport, tmp_path: Path) -> None:
        original_root = tmp_path / "original"
        worktree_root = tmp_path / "worktree"

        original_root.mkdir()
        worktree_root.mkdir()

        # Create package.json in worktree so find_node_project_root finds it
        (worktree_root / "package.json").write_text("{}", encoding="utf-8")

        # Create node_modules in original
        original_node_modules = original_root / "node_modules"
        original_node_modules.mkdir()
        (original_node_modules / "some_package").mkdir()

        test_cfg = make_test_config(worktree_root)
        file_path = worktree_root / "src" / "index.js"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        with (
            patch("codeflash.languages.javascript.support.git_root_dir", return_value=worktree_root),
            patch("codeflash.languages.javascript.support.mirror_path", return_value=original_root),
            patch("codeflash.languages.javascript.optimizer.verify_js_requirements"),
        ):
            js_support.setup_test_config(test_cfg, file_path, current_worktree=original_root)

        worktree_node_modules = test_cfg.js_project_root / "node_modules"
        assert worktree_node_modules.is_symlink()
        assert worktree_node_modules.resolve() == original_node_modules.resolve()

    def test_no_symlink_when_worktree_is_none(self, js_support: JavaScriptSupport, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "package.json").write_text("{}", encoding="utf-8")

        # Create node_modules in project (should remain a real directory, not a symlink)
        (project_root / "node_modules").mkdir()

        test_cfg = make_test_config(project_root)
        file_path = project_root / "index.js"
        file_path.touch()

        with patch("codeflash.languages.javascript.optimizer.verify_js_requirements"):
            js_support.setup_test_config(test_cfg, file_path, current_worktree=None)

        assert not (project_root / "node_modules").is_symlink()

    def test_no_symlink_when_original_node_modules_missing(self, js_support: JavaScriptSupport, tmp_path: Path) -> None:
        original_root = tmp_path / "original"
        worktree_root = tmp_path / "worktree"

        original_root.mkdir()
        worktree_root.mkdir()
        (worktree_root / "package.json").write_text("{}", encoding="utf-8")

        # Do NOT create node_modules in original
        test_cfg = make_test_config(worktree_root)
        file_path = worktree_root / "index.js"
        file_path.touch()

        with (
            patch("codeflash.languages.javascript.support.git_root_dir", return_value=worktree_root),
            patch("codeflash.languages.javascript.support.mirror_path", return_value=original_root),
            patch("codeflash.languages.javascript.optimizer.verify_js_requirements"),
        ):
            js_support.setup_test_config(test_cfg, file_path, current_worktree=original_root)

        worktree_node_modules = test_cfg.js_project_root / "node_modules"
        assert not worktree_node_modules.exists()
        assert not worktree_node_modules.is_symlink()

    def test_no_symlink_when_worktree_node_modules_already_exists(self, js_support: JavaScriptSupport, tmp_path: Path) -> None:
        original_root = tmp_path / "original"
        worktree_root = tmp_path / "worktree"

        original_root.mkdir()
        worktree_root.mkdir()
        (worktree_root / "package.json").write_text("{}", encoding="utf-8")

        # Create node_modules in both
        (original_root / "node_modules").mkdir()
        (worktree_root / "node_modules").mkdir()

        test_cfg = make_test_config(worktree_root)
        file_path = worktree_root / "index.js"
        file_path.touch()

        with (
            patch("codeflash.languages.javascript.support.git_root_dir", return_value=worktree_root),
            patch("codeflash.languages.javascript.support.mirror_path", return_value=original_root),
            patch("codeflash.languages.javascript.optimizer.verify_js_requirements"),
        ):
            js_support.setup_test_config(test_cfg, file_path, current_worktree=original_root)

        worktree_node_modules = test_cfg.js_project_root / "node_modules"
        assert not worktree_node_modules.is_symlink()
        assert worktree_node_modules.is_dir()

    def test_symlink_points_to_correct_target(self, js_support: JavaScriptSupport, tmp_path: Path) -> None:
        original_root = tmp_path / "original"
        worktree_root = tmp_path / "worktree"

        original_root.mkdir()
        worktree_root.mkdir()
        (worktree_root / "package.json").write_text("{}", encoding="utf-8")

        original_node_modules = original_root / "node_modules"
        original_node_modules.mkdir()
        # Add a marker file to verify the symlink target is correct
        (original_node_modules / "marker.txt").write_text("test", encoding="utf-8")

        test_cfg = make_test_config(worktree_root)
        file_path = worktree_root / "index.js"
        file_path.touch()

        with (
            patch("codeflash.languages.javascript.support.git_root_dir", return_value=worktree_root),
            patch("codeflash.languages.javascript.support.mirror_path", return_value=original_root),
            patch("codeflash.languages.javascript.optimizer.verify_js_requirements"),
        ):
            js_support.setup_test_config(test_cfg, file_path, current_worktree=original_root)

        worktree_node_modules = test_cfg.js_project_root / "node_modules"
        assert worktree_node_modules.is_symlink()
        assert (worktree_node_modules / "marker.txt").read_text(encoding="utf-8") == "test"
