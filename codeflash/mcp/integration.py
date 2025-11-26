#!/usr/bin/env python3
"""Claude Code integration utilities for automatic MCP server setup."""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from codeflash.cli_cmds.console import logger


class ClaudeCodeIntegration:
    """Handles Claude Code MCP server integration."""

    def __init__(self):
        self.system = platform.system().lower()
        self.home = Path.home()
        self.claude_config_paths = self._get_claude_config_paths()

    def _get_claude_config_paths(self) -> List[Path]:
        """Get possible Claude Code configuration file paths."""
        paths = []

        if self.system == "darwin":  # macOS
            paths.extend(
                [
                    self.home / ".claude" / "config.json",
                    self.home / "Library" / "Application Support" / "Claude Code" / "config.json",
                    self.home / ".config" / "claude" / "config.json",
                ]
            )
        elif self.system == "linux":
            paths.extend(
                [
                    self.home / ".claude" / "config.json",
                    self.home / ".config" / "claude" / "config.json",
                    Path("/etc/claude/config.json"),
                ]
            )
        elif self.system == "windows":
            appdata = os.getenv("APPDATA", str(self.home / "AppData" / "Roaming"))
            localappdata = os.getenv("LOCALAPPDATA", str(self.home / "AppData" / "Local"))
            paths.extend(
                [
                    Path(appdata) / "Claude Code" / "config.json",
                    Path(localappdata) / "Claude Code" / "config.json",
                    self.home / ".claude" / "config.json",
                ]
            )

        return paths

    def find_claude_config(self) -> Optional[Path]:
        """Find the Claude Code configuration file."""
        for path in self.claude_config_paths:
            if path.exists() and path.is_file():
                try:
                    with open(path) as f:
                        json.load(f)  # Validate it's valid JSON
                    return path
                except (OSError, json.JSONDecodeError):
                    continue
        return None

    def is_claude_code_installed(self) -> bool:
        """Check if Claude Code is installed and accessible."""
        try:
            # Try to find claude command
            claude_cmd = shutil.which("claude")
            if claude_cmd:
                return True

            # Check for config files
            return self.find_claude_config() is not None
        except Exception:
            return False

    def get_codeflash_mcp_executable(self) -> str:
        """Get the path to the codeflash MCP server executable."""
        # Check if we're in development mode
        if hasattr(sys, "_called_from_test") or "pytest" in sys.modules:
            return f"{sys.executable} -m codeflash.mcp.server"

        # Check if codeflash-mcp is in PATH (installed via pip)
        mcp_cmd = shutil.which("codeflash-mcp")
        if mcp_cmd:
            return mcp_cmd

        # Fallback to python module execution
        return f"{sys.executable} -m codeflash.mcp.server"

    def add_codeflash_to_claude_config(self, config_path: Path) -> bool:
        """Add codeflash MCP server to Claude Code configuration."""
        try:
            # Read existing config
            with open(config_path) as f:
                config = json.load(f)

            # Ensure mcpServers section exists
            if "mcpServers" not in config:
                config["mcpServers"] = {}

            # Add codeflash MCP server configuration
            codeflash_config = {
                "command": self.get_codeflash_mcp_executable(),
                "args": [],
                "env": {},
                "disabled": False,
            }

            config["mcpServers"]["codeflash"] = codeflash_config

            # Create backup of original config
            backup_path = config_path.with_suffix(".json.backup")
            shutil.copy2(config_path, backup_path)

            # Write updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"✅ Added codeflash MCP server to Claude Code config: {config_path}")
            logger.info(f"📁 Backup created at: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update Claude Code config: {e}")
            return False

    def create_claude_config(self) -> Optional[Path]:
        """Create a new Claude Code configuration file."""
        # Try to create config in the most appropriate location
        if self.system == "darwin":
            config_path = self.home / ".claude" / "config.json"
        elif self.system == "linux":
            config_path = self.home / ".config" / "claude" / "config.json"
        else:  # windows
            appdata = os.getenv("APPDATA", str(self.home / "AppData" / "Roaming"))
            config_path = Path(appdata) / "Claude Code" / "config.json"

        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Create minimal config with codeflash MCP server
            config = {
                "mcpServers": {
                    "codeflash": {
                        "command": self.get_codeflash_mcp_executable(),
                        "args": [],
                        "env": {},
                        "disabled": False,
                    }
                }
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"✅ Created Claude Code config with codeflash MCP server: {config_path}")
            return config_path

        except Exception as e:
            logger.error(f"❌ Failed to create Claude Code config: {e}")
            return None

    def setup_integration(self, force: bool = False) -> Tuple[bool, str]:
        """Set up codeflash integration with Claude Code.

        Args:
            force: Force setup even if already configured

        Returns:
            Tuple of (success, message)

        """
        if not force and not self.is_claude_code_installed():
            return False, (
                "Claude Code not found. Please install Claude Code first:\n"
                "Visit: https://docs.anthropic.com/en/docs/claude-code\n"
                "Then run: codeflash setup claude-code"
            )

        config_path = self.find_claude_config()

        if config_path:
            # Check if codeflash is already configured
            try:
                with open(config_path) as f:
                    config = json.load(f)

                if not force and "mcpServers" in config and "codeflash" in config["mcpServers"]:
                    return True, f"✅ Codeflash MCP server already configured in: {config_path}"

                # Add codeflash to existing config
                if self.add_codeflash_to_claude_config(config_path):
                    return True, f"✅ Successfully added codeflash to Claude Code config: {config_path}"
                return False, "❌ Failed to update Claude Code configuration"

            except Exception as e:
                return False, f"❌ Error reading Claude Code config: {e}"
        else:
            # Create new config file
            config_path = self.create_claude_config()
            if config_path:
                return True, f"✅ Created new Claude Code config with codeflash: {config_path}"
            return False, "❌ Failed to create Claude Code configuration"

    def remove_integration(self) -> Tuple[bool, str]:
        """Remove codeflash integration from Claude Code."""
        config_path = self.find_claude_config()

        if not config_path:
            return True, "No Claude Code configuration found"

        try:
            with open(config_path) as f:
                config = json.load(f)

            if "mcpServers" not in config or "codeflash" not in config["mcpServers"]:
                return True, "Codeflash not found in Claude Code configuration"

            # Remove codeflash MCP server
            del config["mcpServers"]["codeflash"]

            # Create backup
            backup_path = config_path.with_suffix(".json.backup")
            shutil.copy2(config_path, backup_path)

            # Write updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            return True, f"✅ Removed codeflash from Claude Code config: {config_path}"

        except Exception as e:
            return False, f"❌ Failed to remove codeflash from Claude Code config: {e}"

    def get_status(self) -> Dict[str, any]:
        """Get current integration status."""
        config_path = self.find_claude_config()
        is_installed = self.is_claude_code_installed()
        is_configured = False

        if config_path:
            try:
                with open(config_path) as f:
                    config = json.load(f)
                is_configured = "mcpServers" in config and "codeflash" in config["mcpServers"]
            except Exception:
                pass

        return {
            "claude_code_installed": is_installed,
            "config_file": str(config_path) if config_path else None,
            "codeflash_configured": is_configured,
            "mcp_executable": self.get_codeflash_mcp_executable(),
            "system": self.system,
        }


def setup_claude_code_integration(force: bool = False) -> bool:
    """Main function to set up Claude Code integration."""
    integration = ClaudeCodeIntegration()
    success, message = integration.setup_integration(force=force)

    print(message)

    if success:
        print("\n🎉 Codeflash is now available as a Claude Code subagent!")
        print("\nUsage in Claude Code:")
        print("- 'Optimize the bubble_sort function using codeflash'")
        print("- 'Use codeflash to trace and optimize my main.py script'")
        print("- 'Initialize codeflash in this project'")
        print("\nRestart Claude Code to load the new MCP server.")

    return success


def main():
    """Command-line interface for Claude Code integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up codeflash integration with Claude Code")
    parser.add_argument("--force", action="store_true", help="Force setup even if already configured")
    parser.add_argument("--status", action="store_true", help="Show current integration status")
    parser.add_argument("--remove", action="store_true", help="Remove codeflash integration")

    args = parser.parse_args()

    integration = ClaudeCodeIntegration()

    if args.status:
        status = integration.get_status()
        print(json.dumps(status, indent=2))
    elif args.remove:
        success, message = integration.remove_integration()
        print(message)
        sys.exit(0 if success else 1)
    else:
        success = setup_claude_code_integration(force=args.force)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
