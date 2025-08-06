"""Integration commands for codeflash with development tools."""

import sys
from pathlib import Path

from codeflash.integrations.claude_code import ClaudeCodeIntegration, setup_claude_integration


def handle_integration():
    """Handle integration subcommands."""
    import argparse
    
    # Get the arguments after 'codeflash integrate'
    args = sys.argv[2:]  # Skip 'codeflash integrate'
    
    if not args or args[0] == "status":
        show_integration_status()
    elif args[0] == "claude":
        handle_claude_integration(args[1:])  # Pass remaining args
    else:
        print(f"❌ Unknown integration target: {args[0]}")
        print("Available: claude, status")
        sys.exit(1)


def handle_claude_integration(args):
    """Handle Claude Code integration."""
    import argparse
    
    parser = argparse.ArgumentParser(prog="codeflash integrate claude")
    parser.add_argument("--project", action="store_true", 
                        help="Also create project-specific subagent")
    parser.add_argument("--force", action="store_true",
                        help="Force integration even if Claude Code not detected")
    parser.add_argument("--remove", action="store_true",
                        help="Remove Claude Code integration")
    
    parsed_args = parser.parse_args(args)
    
    integration = ClaudeCodeIntegration()
    
    if parsed_args.remove:
        success, message = integration.remove_integration()
        print(message)
        sys.exit(0 if success else 1)
    
    # Perform integration
    project_path = str(Path.cwd()) if parsed_args.project else None
    success = setup_claude_integration(
        include_project=parsed_args.project,
        project_path=project_path,
        force=parsed_args.force
    )
    
    sys.exit(0 if success else 1)


def show_integration_status():
    """Show current integration status."""
    print("🔍 Codeflash Integration Status")
    print("=" * 40)
    
    # Claude Code status
    integration = ClaudeCodeIntegration()
    status = integration.get_integration_status()
    
    print("\n📋 Claude Code Integration:")
    print(f"  Claude Available: {'✅' if status['claude_available'] else '❌'}")
    
    if status['config_path']:
        print(f"  Config File: {status['config_path']}")
    else:
        print("  Config File: ❌ Not found")
    
    print(f"  Agents Directory: {status['agents_directory']}")
    print(f"  Subagents Installed: {len(status['installed_subagents'])}/3")
    for subagent in status['installed_subagents']:
        print(f"    ✅ {subagent}")
    
    missing_subagents = set(["codeflash-optimizer", "codeflash-profiler", "codeflash-reviewer"]) - set(status['installed_subagents'])
    for subagent in missing_subagents:
        print(f"    ❌ {subagent}")
    
    print(f"  MCP Tools: {'✅' if status['mcp_tools_configured'] else '❌'}")
    print(f"  System: {status['system']}")
    
    # Overall status
    if status['claude_available'] and status['all_subagents_installed']:
        print("\n🎉 Complete Claude Code integration is active!")
        print("\nAvailable subagents:")
        print("  @codeflash-optimizer - AI-powered code optimization")
        print("  @codeflash-profiler - Performance profiling and analysis")
        print("  @codeflash-reviewer - Performance-focused code reviews")
        print("\nUsage examples:")
        print("  '@codeflash-optimizer optimize my slow function'")
        print("  '@codeflash-profiler find bottlenecks in my script'")
        print("  '@codeflash-reviewer review this code for performance'")
    elif status['claude_available'] and status['installed_subagents']:
        print(f"\n⚠️  Partial integration ({len(status['installed_subagents'])}/3 subagents)")
        print("  Run: codeflash integrate claude  # to install all subagents")
    elif status['claude_available']:
        print("\n⚠️  Claude Code available but subagents not installed")
        print("  Run: codeflash integrate claude")
    else:
        print("\n📖 Install Claude Code to enable integration:")
        print("  Visit: https://docs.anthropic.com/en/docs/claude-code")
        print("  Then run: codeflash integrate claude")
    
    # Check for project-specific integration
    project_agent = Path.cwd() / ".claude" / "agents" / "codeflash-optimizer.md"
    if project_agent.exists():
        print(f"\n🏠 Project subagent found: {project_agent}")
    else:
        print("\n💡 Tip: Use --project flag to create project-specific subagent")
        print("  codeflash integrate claude --project")


if __name__ == "__main__":
    handle_integration()