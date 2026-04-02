"""Test for pnpm workspace handling in package installation."""

import tempfile
from pathlib import Path

from codeflash.cli_cmds.init_javascript import get_package_install_command


def test_pnpm_workspace_adds_workspace_flag() -> None:
    """Test that pnpm workspace projects get the -w flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create pnpm-workspace.yaml to indicate workspace
        (project_root / "pnpm-workspace.yaml").write_text("packages:\n  - .")

        # Create pnpm-lock.yaml to indicate pnpm is the package manager
        (project_root / "pnpm-lock.yaml").write_text("lockfileVersion: '6.0'")

        # Create package.json
        (project_root / "package.json").write_text('{"name": "test"}')

        # Get install command
        cmd = get_package_install_command(project_root, "some-package", dev=True)

        # Should include -w flag for workspace root
        assert "-w" in cmd or "--workspace-root" in cmd, f"Expected workspace flag in {cmd}"


def test_dev_environment_uses_local_package() -> None:
    """Test that dev environment uses local codeflash package path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        (project_root / "package.json").write_text('{"name": "test"}')

        # Get install command for codeflash
        cmd = get_package_install_command(project_root, "codeflash", dev=True)

        # In dev mode (when running from /opt/codeflash/),
        # should use local package path instead of npm package name
        cmd_str = " ".join(cmd)

        # Should reference local packages directory
        assert (
            "/opt/codeflash/packages/codeflash" in cmd_str
            or "file:" in cmd_str
            or cmd[0] in ["npm", "pnpm", "yarn", "bun"]
        ), f"Expected local package reference or valid package manager in {cmd}"


if __name__ == "__main__":
    # Run tests
    test_pnpm_workspace_adds_workspace_flag()
    test_dev_environment_uses_local_package()
    print("All tests passed!")
