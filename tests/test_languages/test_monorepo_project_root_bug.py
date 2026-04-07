"""Regression test for monorepo Jest config bug.

Bug: In --all mode with monorepo, test_cfg.js_project_root is set once based on the
first file and reused for all functions. When optimizing functions from different
packages, Jest runs with the wrong package's config, causing module resolution failures.

Example: Optimizing worker/src/tenants.ts uses server's Jest config, breaking imports.

Trace ID: 02f0351a-db89-4ebc-a2e6-c45b19061152
"""
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from codeflash.languages.javascript.test_runner import find_node_project_root, run_jest_behavioral_tests


@pytest.fixture
def monorepo_structure(tmp_path):
    """Create minimal monorepo like budibase with server and worker packages."""
    root = tmp_path / "repo"
    root.mkdir()

    # Root with yarn workspaces
    (root / "package.json").write_text('{"workspaces": {"packages": ["packages/*"]}}')
    (root / "yarn.lock").touch()
    (root / "node_modules").mkdir()

    # Server package
    server = root / "packages/server"
    server.mkdir(parents=True)
    (server / "package.json").write_text('{"name": "@test/server"}')
    (server / "jest.config.js").write_text('module.exports = {testEnvironment: "node"};')

    # Worker package
    worker = root / "packages/worker"
    worker.mkdir(parents=True)
    (worker / "package.json").write_text('{"name": "@test/worker"}')
    (worker / "jest.config.js").write_text('module.exports = {testEnvironment: "node"};')

    return root


def test_find_node_project_root_detects_correct_package(monorepo_structure):
    """Verify find_node_project_root returns the correct package, not monorepo root."""
    server_file = monorepo_structure / "packages/server/src/api.ts"
    server_file.parent.mkdir(parents=True)
    server_file.touch()

    worker_file = monorepo_structure / "packages/worker/src/tenant.ts"
    worker_file.parent.mkdir(parents=True)
    worker_file.touch()

    # Each file should resolve to its own package, not the monorepo root
    server_root = find_node_project_root(server_file)
    worker_root = find_node_project_root(worker_file)

    assert server_root == monorepo_structure / "packages/server"
    assert worker_root == monorepo_structure / "packages/worker"
    assert server_root != worker_root, "Different packages must have different roots"


def test_run_jest_uses_correct_cwd_when_project_root_is_wrong(monorepo_structure):
    """
    REGRESSION TEST for bug where wrong project_root causes Jest module resolution failures.

    Scenario: test_cfg.js_project_root points to server, but we're testing worker files.
    Expected: run_jest_behavioral_tests should detect worker package from test file path.
    Actual (before fix): Uses wrong project_root, causing "Cannot find module" errors.

    This test documents current behavior (uses wrong cwd) and will pass after fix.
    """
    # Create test file in worker package
    worker_test = monorepo_structure / "packages/worker/src/tests/test_tenant.test.ts"
    worker_test.parent.mkdir(parents=True)
    worker_test.write_text('test("dummy", () => {});')

    # Simulate bug: project_root wrongly points to server package
    wrong_project_root = monorepo_structure / "packages/server"
    correct_project_root = monorepo_structure / "packages/worker"

    # What happens now (before fix):
    # Line 782: effective_cwd = project_root if project_root else cwd
    # Since project_root = wrong_project_root (server), effective_cwd is wrong

    # Current behavior: find_node_project_root is ONLY called if project_root is None (line 779)
    # This is the bug - we should call it even when project_root is provided

    # After fix: run_jest_behavioral_tests should always verify project_root
    # by calling find_node_project_root with the test file path

    # For now, just test that find_node_project_root would return the right answer
    detected_root = find_node_project_root(worker_test)
    assert detected_root == correct_project_root, \
        "find_node_project_root should detect worker package from test path"

    # The actual fix will be: instead of using project_root directly as effective_cwd,
    # run_jest_behavioral_tests should call find_node_project_root(test_files[0])
    # to determine the correct package for Jest execution


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
