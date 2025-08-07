#!/usr/bin/env python3
"""Native Claude Code integration with subagent installation."""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from codeflash.cli_cmds.console import logger


class ClaudeCodeIntegration:
    """Handles native Claude Code integration with subagents and MCP tools."""

    def __init__(self):
        self.system = platform.system().lower()
        self.home = Path.home()
        self.claude_paths = self._get_claude_paths()
        self.agents_dir_user = self.home / ".claude" / "agents"

    def _get_claude_paths(self) -> Dict[str, List[Path]]:
        """Get Claude Code installation and configuration paths."""
        config_paths = []
        agents_paths = []

        if self.system == "darwin":  # macOS
            config_paths.extend(
                [
                    self.home / ".claude" / "config.json",
                    self.home / "Library" / "Application Support" / "Claude Code" / "config.json",
                ]
            )
            agents_paths.extend(
                [
                    self.home / ".claude" / "agents",
                    self.home / "Library" / "Application Support" / "Claude Code" / "agents",
                ]
            )
        elif self.system == "linux":
            config_paths.extend(
                [self.home / ".claude" / "config.json", self.home / ".config" / "claude" / "config.json"]
            )
            agents_paths.extend([self.home / ".claude" / "agents", self.home / ".config" / "claude" / "agents"])
        elif self.system == "windows":
            appdata = os.getenv("APPDATA", str(self.home / "AppData" / "Roaming"))
            config_paths.extend([Path(appdata) / "Claude Code" / "config.json", self.home / ".claude" / "config.json"])
            agents_paths.extend([Path(appdata) / "Claude Code" / "agents", self.home / ".claude" / "agents"])

        return {"config": config_paths, "agents": agents_paths}

    def find_claude_config(self) -> Optional[Path]:
        """Find existing Claude Code configuration."""
        for path in self.claude_paths["config"]:
            if path.exists() and path.is_file():
                try:
                    with open(path) as f:
                        json.load(f)
                    return path
                except (OSError, json.JSONDecodeError):
                    continue
        return None

    def find_agents_directory(self) -> Path:
        """Find or create Claude Code agents directory."""
        for path in self.claude_paths["agents"]:
            if path.exists():
                return path

        # Create the standard user-level agents directory
        self.agents_dir_user.mkdir(parents=True, exist_ok=True)
        return self.agents_dir_user

    def is_claude_code_available(self) -> bool:
        """Check if Claude Code is available."""
        # Check for claude command
        if shutil.which("claude"):
            return True

        # Check for config files
        return self.find_claude_config() is not None

    def install_subagents(self, subagents: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Install codeflash subagents."""
        if subagents is None:
            subagents = ["codeflash-optimizer", "codeflash-profiler", "codeflash-reviewer"]

        results = []
        overall_success = True

        try:
            # Find agents directory
            agents_dir = self.find_agents_directory()
            agents_source_dir = Path(__file__).parent.parent / "agents"

            for subagent in subagents:
                source_agent = agents_source_dir / f"{subagent}.md"
                target_agent = agents_dir / f"{subagent}.md"

                if not source_agent.exists():
                    results.append(f"❌ {subagent}: template not found")
                    overall_success = False
                    continue

                # Create backup if exists
                if target_agent.exists():
                    backup_path = target_agent.with_suffix(".md.backup")
                    shutil.copy2(target_agent, backup_path)
                    logger.info(f"📁 Backup created: {backup_path}")

                # Install the subagent
                shutil.copy2(source_agent, target_agent)
                results.append(f"✅ {subagent}: installed")

            return overall_success, "\n".join(results)

        except Exception as e:
            return False, f"❌ Failed to install subagents: {e}"

    def setup_mcp_tools(self) -> Tuple[bool, str]:
        """Set up MCP tools for the subagent to use."""
        config_path = self.find_claude_config()

        if not config_path:
            # Create new config
            config_path = self.home / ".claude" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {"mcpServers": {}}
        else:
            # Load existing config
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception as e:
                return False, f"❌ Failed to read config: {e}"

        # Ensure mcpServers section exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Add codeflash MCP server for tools
        config["mcpServers"]["codeflash"] = {
            "command": f"{sys.executable}",
            "args": ["-m", "codeflash.mcp.server"],
            "env": {},
            "disabled": False,
        }

        try:
            # Create backup
            if config_path.exists():
                backup_path = config_path.with_suffix(".json.backup")
                shutil.copy2(config_path, backup_path)

            # Write updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            return True, f"✅ MCP tools configured: {config_path}"

        except Exception as e:
            return False, f"❌ Failed to configure MCP tools: {e}"

    def create_project_subagent(self, project_path: Path) -> Tuple[bool, str]:
        """Create project-specific codeflash subagent."""
        try:
            project_agents_dir = project_path / ".claude" / "agents"
            project_agents_dir.mkdir(parents=True, exist_ok=True)

            # Copy the subagent with project-specific customizations
            source_agent = Path(__file__).parent.parent / "agents" / "codeflash-optimizer.md"
            target_agent = project_agents_dir / "codeflash-optimizer.md"

            if source_agent.exists():
                shutil.copy2(source_agent, target_agent)
                return True, f"✅ Project subagent created: {target_agent}"
            return False, "❌ Source subagent not found"

        except Exception as e:
            return False, f"❌ Failed to create project subagent: {e}"

    def get_integration_status(self) -> Dict[str, any]:
        """Get current integration status."""
        config_path = self.find_claude_config()
        agents_dir = self.find_agents_directory()

        # Check which subagents are installed
        subagents = ["codeflash-optimizer", "codeflash-profiler", "codeflash-reviewer"]
        installed_subagents = []
        for subagent in subagents:
            if (agents_dir / f"{subagent}.md").exists():
                installed_subagents.append(subagent)

        mcp_configured = False
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                mcp_configured = "mcpServers" in config and "codeflash" in config["mcpServers"]
            except Exception:
                pass

        return {
            "claude_available": self.is_claude_code_available(),
            "config_path": str(config_path) if config_path else None,
            "agents_directory": str(agents_dir),
            "installed_subagents": installed_subagents,
            "all_subagents_installed": len(installed_subagents) == len(subagents),
            "mcp_tools_configured": mcp_configured,
            "system": self.system,
        }

    def setup_complete_integration(
        self, include_project: bool = False, project_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """Set up complete Claude Code integration."""
        results = []
        overall_success = True

        # Step 1: Install subagents
        success, message = self.install_subagents()
        results.append("Subagent Installation:")
        results.append(message)
        if not success:
            overall_success = False

        # Step 2: Set up MCP tools
        success, message = self.setup_mcp_tools()
        results.append(message)
        if not success:
            overall_success = False

        # Step 3: Create project subagent if requested
        if include_project and project_path:
            success, message = self.create_project_subagent(project_path)
            results.append(message)

        return overall_success, "\n".join(results)

    def remove_integration(self) -> Tuple[bool, str]:
        """Remove codeflash integration."""
        results = []

        # Remove subagent
        agents_dir = self.find_agents_directory()
        subagent_path = agents_dir / "codeflash-optimizer.md"

        if subagent_path.exists():
            try:
                subagent_path.unlink()
                results.append("✅ Removed codeflash optimizer subagent")
            except Exception as e:
                results.append(f"❌ Failed to remove subagent: {e}")

        # Remove MCP configuration
        config_path = self.find_claude_config()
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                if "mcpServers" in config and "codeflash" in config["mcpServers"]:
                    del config["mcpServers"]["codeflash"]

                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                    results.append("✅ Removed MCP tools configuration")

            except Exception as e:
                results.append(f"❌ Failed to remove MCP config: {e}")

        return True, "\n".join(results)


def setup_claude_integration(
    include_project: bool = False, project_path: Optional[str] = None, force: bool = False
) -> bool:
    """Main setup function for Claude Code integration."""
    integration = ClaudeCodeIntegration()

    if not force and not integration.is_claude_code_available():
        print("⚠️  Claude Code not detected.")
        print("Install Claude Code first: https://docs.anthropic.com/en/docs/claude-code")
        print("Then run: codeflash integrate claude")
        return False

    project_path_obj = Path(project_path) if project_path else None
    success, message = integration.setup_complete_integration(
        include_project=include_project, project_path=project_path_obj
    )

    print(message)

    if success:
        print("\n🎉 Codeflash is now integrated with Claude Code!")
        print("\nHow to use:")
        print("1. In Claude Code, type: '/agents' to see available subagents")
        print("2. Invoke with: '@codeflash-optimizer optimize my bubble_sort function'")
        print("3. Or let Claude auto-invoke when discussing performance")
        print("\n📝 Restart Claude Code to ensure all changes take effect")

    return success
