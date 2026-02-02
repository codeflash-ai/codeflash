"""Tests for JavaScript/TypeScript project initialization and package manager detection."""

from pathlib import Path

import pytest

from codeflash.cli_cmds.init_javascript import (
    JsPackageManager,
    determine_js_package_manager,
    get_package_install_command,
)


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    return tmp_path


class TestDetermineJsPackageManager:
    """Tests for determine_js_package_manager function."""

    def test_detects_pnpm_from_lockfile(self, tmp_project: Path) -> None:
        """Should detect pnpm from pnpm-lock.yaml."""
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.PNPM

    def test_detects_yarn_from_lockfile(self, tmp_project: Path) -> None:
        """Should detect yarn from yarn.lock."""
        (tmp_project / "yarn.lock").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.YARN

    def test_detects_npm_from_lockfile(self, tmp_project: Path) -> None:
        """Should detect npm from package-lock.json."""
        (tmp_project / "package-lock.json").write_text("{}")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.NPM

    def test_detects_bun_from_lockfile(self, tmp_project: Path) -> None:
        """Should detect bun from bun.lockb."""
        (tmp_project / "bun.lockb").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.BUN

    def test_detects_bun_from_bun_lock(self, tmp_project: Path) -> None:
        """Should detect bun from bun.lock."""
        (tmp_project / "bun.lock").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.BUN

    def test_defaults_to_npm_with_package_json_only(self, tmp_project: Path) -> None:
        """Should default to npm when only package.json exists."""
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.NPM

    def test_returns_unknown_without_package_json(self, tmp_project: Path) -> None:
        """Should return UNKNOWN when no package.json exists."""
        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.UNKNOWN

    def test_pnpm_takes_precedence_over_npm(self, tmp_project: Path) -> None:
        """Should prefer pnpm when both lockfiles exist (migration scenario)."""
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package-lock.json").write_text("{}")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.PNPM

    def test_bun_takes_precedence_over_others(self, tmp_project: Path) -> None:
        """Should prefer bun when bun.lockb exists alongside others."""
        (tmp_project / "bun.lockb").write_text("")
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = determine_js_package_manager(tmp_project)

        assert result == JsPackageManager.BUN


class TestGetPackageInstallCommand:
    """Tests for get_package_install_command function."""

    def test_npm_install_command(self, tmp_project: Path) -> None:
        """Should return npm install command for npm projects."""
        (tmp_project / "package-lock.json").write_text("{}")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=True)

        assert result == ["npm", "install", "codeflash", "--save-dev"]

    def test_npm_install_command_non_dev(self, tmp_project: Path) -> None:
        """Should return npm install command without --save-dev when dev=False."""
        (tmp_project / "package-lock.json").write_text("{}")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=False)

        assert result == ["npm", "install", "codeflash"]

    def test_pnpm_add_command(self, tmp_project: Path) -> None:
        """Should return pnpm add command for pnpm projects."""
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=True)

        assert result == ["pnpm", "add", "codeflash", "--save-dev"]

    def test_pnpm_add_command_non_dev(self, tmp_project: Path) -> None:
        """Should return pnpm add command without --save-dev when dev=False."""
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=False)

        assert result == ["pnpm", "add", "codeflash"]

    def test_yarn_add_command(self, tmp_project: Path) -> None:
        """Should return yarn add command for yarn projects."""
        (tmp_project / "yarn.lock").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=True)

        assert result == ["yarn", "add", "codeflash", "--dev"]

    def test_yarn_add_command_non_dev(self, tmp_project: Path) -> None:
        """Should return yarn add command without --dev when dev=False."""
        (tmp_project / "yarn.lock").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=False)

        assert result == ["yarn", "add", "codeflash"]

    def test_bun_add_command(self, tmp_project: Path) -> None:
        """Should return bun add command for bun projects."""
        (tmp_project / "bun.lockb").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=True)

        assert result == ["bun", "add", "codeflash", "--dev"]

    def test_bun_add_command_non_dev(self, tmp_project: Path) -> None:
        """Should return bun add command without --dev when dev=False."""
        (tmp_project / "bun.lockb").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "codeflash", dev=False)

        assert result == ["bun", "add", "codeflash"]

    def test_defaults_to_npm_for_unknown(self, tmp_project: Path) -> None:
        """Should default to npm for unknown package manager."""
        # No lockfile, no package.json - unknown package manager
        result = get_package_install_command(tmp_project, "codeflash", dev=True)

        assert result == ["npm", "install", "codeflash", "--save-dev"]

    def test_different_package_name(self, tmp_project: Path) -> None:
        """Should work with different package names."""
        (tmp_project / "pnpm-lock.yaml").write_text("")
        (tmp_project / "package.json").write_text("{}")

        result = get_package_install_command(tmp_project, "typescript", dev=True)

        assert result == ["pnpm", "add", "typescript", "--save-dev"]
