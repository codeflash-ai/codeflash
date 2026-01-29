"""Tests for JavaScript/TypeScript GitHub Actions workflow helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeflash.cli_cmds.init_javascript import (
    JsPackageManager,
    get_js_codeflash_install_step,
    get_js_codeflash_run_command,
    get_js_dependency_installation_commands,
    get_js_runtime_setup_steps,
    is_codeflash_dependency,
)


class TestIsCodeflashDependency:
    """Tests for is_codeflash_dependency function."""

    def test_returns_true_when_in_dev_dependencies(self, tmp_path: Path) -> None:
        """Should return True when codeflash is in devDependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"devDependencies": {"codeflash": "^1.0.0"}}))

        assert is_codeflash_dependency(tmp_path) is True

    def test_returns_true_when_in_dependencies(self, tmp_path: Path) -> None:
        """Should return True when codeflash is in dependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"dependencies": {"codeflash": "^1.0.0"}}))

        assert is_codeflash_dependency(tmp_path) is True

    def test_returns_false_when_not_present(self, tmp_path: Path) -> None:
        """Should return False when codeflash is not in any dependencies."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"devDependencies": {"jest": "^29.0.0"}}))

        assert is_codeflash_dependency(tmp_path) is False

    def test_returns_false_when_no_package_json(self, tmp_path: Path) -> None:
        """Should return False when package.json doesn't exist."""
        assert is_codeflash_dependency(tmp_path) is False

    def test_returns_false_for_invalid_json(self, tmp_path: Path) -> None:
        """Should return False for invalid package.json."""
        package_json = tmp_path / "package.json"
        package_json.write_text("invalid json")

        assert is_codeflash_dependency(tmp_path) is False

    def test_handles_empty_dependencies(self, tmp_path: Path) -> None:
        """Should handle empty dependencies objects."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test", "dependencies": {}, "devDependencies": {}}))

        assert is_codeflash_dependency(tmp_path) is False


class TestGetJsRuntimeSetupSteps:
    """Tests for get_js_runtime_setup_steps function."""

    def test_npm_setup(self) -> None:
        """Should generate correct setup steps for npm."""
        result = get_js_runtime_setup_steps(JsPackageManager.NPM)

        assert "Setup Node.js" in result
        assert "actions/setup-node@v4" in result
        assert "node-version: '22'" in result
        assert "cache: 'npm'" in result

    def test_yarn_setup(self) -> None:
        """Should generate correct setup steps for yarn."""
        result = get_js_runtime_setup_steps(JsPackageManager.YARN)

        assert "Setup Node.js" in result
        assert "actions/setup-node@v4" in result
        assert "cache: 'yarn'" in result

    def test_pnpm_setup(self) -> None:
        """Should generate correct setup steps for pnpm."""
        result = get_js_runtime_setup_steps(JsPackageManager.PNPM)

        assert "Setup pnpm" in result
        assert "pnpm/action-setup@v4" in result
        assert "Setup Node.js" in result
        assert "cache: 'pnpm'" in result

    def test_bun_setup(self) -> None:
        """Should generate correct setup steps for bun."""
        result = get_js_runtime_setup_steps(JsPackageManager.BUN)

        assert "Setup Bun" in result
        assert "oven-sh/setup-bun@v2" in result
        assert "bun-version: latest" in result

    def test_unknown_defaults_to_npm(self) -> None:
        """Should default to npm setup for unknown package manager."""
        result = get_js_runtime_setup_steps(JsPackageManager.UNKNOWN)

        assert "cache: 'npm'" in result


class TestGetJsDependencyInstallationCommands:
    """Tests for get_js_dependency_installation_commands function."""

    def test_npm_install(self) -> None:
        """Should return npm ci for npm."""
        assert get_js_dependency_installation_commands(JsPackageManager.NPM) == "npm ci"

    def test_yarn_install(self) -> None:
        """Should return yarn install for yarn."""
        assert get_js_dependency_installation_commands(JsPackageManager.YARN) == "yarn install"

    def test_pnpm_install(self) -> None:
        """Should return pnpm install for pnpm."""
        assert get_js_dependency_installation_commands(JsPackageManager.PNPM) == "pnpm install"

    def test_bun_install(self) -> None:
        """Should return bun install for bun."""
        assert get_js_dependency_installation_commands(JsPackageManager.BUN) == "bun install"


class TestGetJsCodeflashInstallStep:
    """Tests for get_js_codeflash_install_step function."""

    def test_returns_empty_when_is_dependency(self) -> None:
        """Should return empty string when codeflash is a dependency."""
        result = get_js_codeflash_install_step(JsPackageManager.NPM, is_dependency=True)

        assert result == ""

    def test_npm_global_install(self) -> None:
        """Should generate npm global install when not a dependency."""
        result = get_js_codeflash_install_step(JsPackageManager.NPM, is_dependency=False)

        assert "Install Codeflash" in result
        assert "npm install -g codeflash" in result

    def test_yarn_global_install(self) -> None:
        """Should generate yarn global install when not a dependency."""
        result = get_js_codeflash_install_step(JsPackageManager.YARN, is_dependency=False)

        assert "yarn global add codeflash" in result

    def test_pnpm_global_install(self) -> None:
        """Should generate pnpm global install when not a dependency."""
        result = get_js_codeflash_install_step(JsPackageManager.PNPM, is_dependency=False)

        assert "pnpm add -g codeflash" in result

    def test_bun_global_install(self) -> None:
        """Should generate bun global install when not a dependency."""
        result = get_js_codeflash_install_step(JsPackageManager.BUN, is_dependency=False)

        assert "bun add -g codeflash" in result


class TestGetJsCodeflashRunCommand:
    """Tests for get_js_codeflash_run_command function."""

    def test_npm_with_dependency(self) -> None:
        """Should use npx when codeflash is a dependency."""
        result = get_js_codeflash_run_command(JsPackageManager.NPM, is_dependency=True)

        assert result == "npx codeflash"

    def test_npm_without_dependency(self) -> None:
        """Should use direct codeflash when globally installed."""
        result = get_js_codeflash_run_command(JsPackageManager.NPM, is_dependency=False)

        assert result == "codeflash"

    def test_yarn_with_dependency(self) -> None:
        """Should use yarn codeflash when it's a dependency."""
        result = get_js_codeflash_run_command(JsPackageManager.YARN, is_dependency=True)

        assert result == "yarn codeflash"

    def test_pnpm_with_dependency(self) -> None:
        """Should use pnpm exec when it's a dependency."""
        result = get_js_codeflash_run_command(JsPackageManager.PNPM, is_dependency=True)

        assert result == "pnpm exec codeflash"

    def test_bun_with_dependency(self) -> None:
        """Should use bun run when it's a dependency."""
        result = get_js_codeflash_run_command(JsPackageManager.BUN, is_dependency=True)

        assert result == "bun run codeflash"

    def test_all_global_installs_use_direct_command(self) -> None:
        """All package managers should use direct 'codeflash' when globally installed."""
        for pm in [JsPackageManager.NPM, JsPackageManager.YARN, JsPackageManager.PNPM, JsPackageManager.BUN]:
            result = get_js_codeflash_run_command(pm, is_dependency=False)
            assert result == "codeflash", f"Failed for {pm}"


class TestWorkflowTemplateIntegration:
    """Integration tests for workflow template generation."""

    def test_workflow_template_exists(self) -> None:
        """Verify the JS workflow template file exists."""
        from importlib.resources import files

        template_path = files("codeflash").joinpath("cli_cmds", "workflows", "codeflash-optimize-js.yaml")
        content = template_path.read_text(encoding="utf-8")

        # Check all placeholders exist
        assert "{{ codeflash_module_path }}" in content
        assert "{{ working_directory }}" in content
        assert "{{ setup_runtime_steps }}" in content
        assert "{{ install_dependencies_command }}" in content
        assert "{{ install_codeflash_step }}" in content
        assert "{{ codeflash_command }}" in content

    def test_workflow_template_has_correct_structure(self) -> None:
        """Verify the JS workflow template has the expected YAML structure."""
        from importlib.resources import files

        template_path = files("codeflash").joinpath("cli_cmds", "workflows", "codeflash-optimize-js.yaml")
        content = template_path.read_text(encoding="utf-8")

        # Check key sections
        assert "name: Codeflash" in content
        assert "pull_request:" in content
        assert "workflow_dispatch:" in content
        assert "concurrency:" in content
        assert "cancel-in-progress: true" in content
        assert "jobs:" in content
        assert "optimize:" in content
        assert "github.actor != 'codeflash-ai[bot]'" in content
        assert "CODEFLASH_API_KEY" in content
        assert "actions/checkout@v4" in content
        assert "fetch-depth: 0" in content
