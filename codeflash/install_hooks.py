#!/usr/bin/env python3
"""Post-install hooks for codeflash to set up integrations automatically."""

import sys


def post_install_hook():
    """Run after codeflash installation to set up integrations."""
    try:
        # Only auto-setup in interactive environments
        if not sys.stdin.isatty():
            return

        from codeflash.integrations.claude_code import ClaudeCodeIntegration

        integration = ClaudeCodeIntegration()

        # Check if Claude Code is available
        if not integration.is_claude_code_available():
            print("\n📖 Claude Code integration available!")
            print("   Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
            print("   Then run: codeflash integrate claude")
            return

        # Check if already configured
        status = integration.get_integration_status()
        if status["subagent_installed"]:
            print("\n✅ Codeflash Claude Code integration is already active!")
            return

        print("\n🚀 Claude Code detected! Setting up codeflash subagent...")

        success, message = integration.setup_complete_integration()
        if success:
            print("✅ Codeflash subagent installed!")
            print("\nUsage in Claude Code:")
            print("  @codeflash-optimizer optimize my slow function")
            print("  Or let Claude auto-invoke for performance tasks")
            print("\n📝 Restart Claude Code to activate the subagent")
        else:
            print("⚠️  Auto-setup encountered issues:")
            print(message)
            print("\n   Run manually: codeflash integrate claude")

    except Exception as e:
        # Silently fail to avoid breaking installation
        print(f"\n⚠️  Auto-integration encountered an issue: {e}")
        print("   Run manually: codeflash integrate claude")


if __name__ == "__main__":
    post_install_hook()
