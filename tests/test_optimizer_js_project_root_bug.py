"""Test that test_cfg.js_project_root caching bug is demonstrated and bypassed by the fix."""

from pathlib import Path
from unittest.mock import patch

from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash.verification.verification_utils import TestConfig


@patch("codeflash.languages.javascript.optimizer.verify_js_requirements")
def test_js_project_root_cached_in_test_cfg(mock_verify: object, tmp_path: Path) -> None:
    """Demonstrates that test_cfg.js_project_root is set once per setup_test_config call.

    This test shows the root cause: test_cfg caches the project root from the first function.
    The fix bypasses this cache in FunctionOptimizer.get_js_project_root() instead of
    changing how test_cfg stores the value.
    """
    mock_verify.return_value = []  # type: ignore[attr-defined]

    # Create main project
    main_project = (tmp_path / "project").resolve()
    main_project.mkdir()
    (main_project / "package.json").write_text('{"name": "main"}', encoding="utf-8")
    (main_project / "src").mkdir()
    (main_project / "test").mkdir()
    (main_project / "node_modules").mkdir()

    # Create extension with its own package.json
    extension_dir = (main_project / "extensions" / "discord").resolve()
    extension_dir.mkdir(parents=True)
    (extension_dir / "package.json").write_text('{"name": "discord-extension"}', encoding="utf-8")
    (extension_dir / "src").mkdir()
    (extension_dir / "node_modules").mkdir()

    test_cfg = TestConfig(
        tests_root=main_project / "test",
        project_root_path=main_project,
        tests_project_rootdir=main_project / "test",
    )
    test_cfg.set_language("javascript")

    js_support = JavaScriptSupport()

    extension_file = (extension_dir / "src" / "accounts.ts").resolve()
    extension_file.write_text("export function foo() {}", encoding="utf-8")

    success = js_support.setup_test_config(test_cfg, extension_file, current_worktree=None)
    assert success, "setup_test_config should succeed"
    # After setup for extension file, js_project_root is the extension directory
    assert test_cfg.js_project_root == extension_dir

    # test_cfg is NOT re-initialized for subsequent functions — js_project_root stays cached
    main_file = (main_project / "src" / "commands.ts").resolve()
    main_file.write_text("export function bar() {}", encoding="utf-8")

    # The cached value is still extension_dir, not main_project — this is the root cause
    assert test_cfg.js_project_root == extension_dir
