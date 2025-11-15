"""Setup command for codeflash integrations and configurations."""

import sys

import click

from codeflash.mcp.integration import ClaudeCodeIntegration


@click.command()
@click.argument("integration", type=click.Choice(["claude-code", "status"], case_sensitive=False))
@click.option("--force", is_flag=True, help="Force setup even if already configured")
@click.option("--remove", is_flag=True, help="Remove the integration")
def setup_integrations(integration: str, force: bool = False, remove: bool = False):
    """Set up codeflash integrations with various tools and editors.

    Available integrations:
    - claude-code: Set up as Claude Code MCP server subagent
    - status: Show current integration status
    """
    if integration.lower() == "claude-code":
        setup_claude_code(force=force, remove=remove)
    elif integration.lower() == "status":
        show_integration_status()
    else:
        click.echo(f"Unknown integration: {integration}")
        sys.exit(1)


def setup_claude_code(force: bool = False, remove: bool = False):
    """Set up Claude Code integration."""
    integration = ClaudeCodeIntegration()

    if remove:
        success, message = integration.remove_integration()
        click.echo(message)
        sys.exit(0 if success else 1)

    success, message = integration.setup_integration(force=force)
    click.echo(message)

    if success:
        click.echo("\n🎉 Codeflash is now available as a Claude Code subagent!")
        click.echo("\nUsage examples in Claude Code:")
        click.echo("• 'Optimize the bubble_sort function using codeflash'")
        click.echo("• 'Use codeflash to trace and optimize my main.py script'")
        click.echo("• 'Initialize codeflash in this project'")
        click.echo("• 'Run codeflash benchmarks to measure performance'")
        click.echo("\n📝 Restart Claude Code to load the new MCP server.")

        # Show quick verification steps
        click.echo("\n🔍 Verification steps:")
        click.echo("1. Restart Claude Code")
        click.echo("2. In Claude Code, type: 'Show me codeflash help'")
        click.echo("3. Claude Code should respond with codeflash functionality")

    sys.exit(0 if success else 1)


def show_integration_status():
    """Show current integration status for all supported tools."""
    integration = ClaudeCodeIntegration()
    status = integration.get_status()

    click.echo("🔍 Codeflash Integration Status\n")

    # Claude Code status
    click.echo("📋 Claude Code:")
    if status["claude_code_installed"]:
        click.echo("  ✅ Claude Code is installed")
    else:
        click.echo("  ❌ Claude Code not found")

    if status["config_file"]:
        click.echo(f"  📁 Config file: {status['config_file']}")
    else:
        click.echo("  📁 No config file found")

    if status["codeflash_configured"]:
        click.echo("  ✅ Codeflash MCP server is configured")
    else:
        click.echo("  ❌ Codeflash MCP server not configured")

    click.echo(f"  🔧 MCP executable: {status['mcp_executable']}")
    click.echo(f"  💻 System: {status['system']}")

    # Overall status
    if status["claude_code_installed"] and status["codeflash_configured"]:
        click.echo("\n🎉 All integrations are working!")
    elif status["claude_code_installed"]:
        click.echo("\n⚠️  Claude Code is installed but codeflash is not configured.")
        click.echo("   Run: codeflash setup claude-code")
    else:
        click.echo("\n📖 Install Claude Code to enable MCP integration:")
        click.echo("   Visit: https://docs.anthropic.com/en/docs/claude-code")


# Function that can be called from main CLI
def setup_claude_code_integration(force: bool = False):
    """Setup function that can be called from main CLI."""
    from codeflash.mcp.integration import setup_claude_code_integration as _setup

    return _setup(force=force)


if __name__ == "__main__":
    setup_integrations()
