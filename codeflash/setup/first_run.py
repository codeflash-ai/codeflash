"""First-run experience for Codeflash.

This module handles the seamless first-run experience:
1. Auto-detect project settings
2. Display detected settings
3. Quick confirmation
4. API key setup
5. Save config and continue

Usage:
    from codeflash.setup.first_run import handle_first_run, is_first_run

    if is_first_run():
        args = handle_first_run(args)
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.cli_cmds.console import console
from codeflash.setup.config_writer import write_config
from codeflash.setup.detector import detect_project, has_existing_config

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path


def is_first_run(project_root: Path | None = None) -> bool:
    """Check if this is the first run (no config exists).

    Args:
        project_root: Project root to check. Defaults to auto-detect.

    Returns:
        True if no Codeflash config exists.

    """
    if project_root is None:
        try:
            detected = detect_project()
            project_root = detected.project_root
        except Exception:
            return True

    has_config, _ = has_existing_config(project_root)
    return not has_config


def handle_first_run(
    args: Namespace | None = None, skip_confirm: bool = False, skip_api_key: bool = False
) -> Namespace | None:
    """Handle the first-run experience with auto-detection and quick confirm.

    This is the main entry point for the frictionless setup experience.

    Args:
        args: Optional CLI args namespace to update.
        skip_confirm: Skip confirmation prompt (--yes flag).
        skip_api_key: Skip API key prompt.

    Returns:
        Updated args namespace with detected settings, or None if user cancelled.

    """
    from argparse import Namespace

    # Auto-detect project
    try:
        detected = detect_project()
    except Exception as e:
        _show_detection_error(str(e))
        return None

    # Show welcome message
    _show_welcome()

    # Show detected settings
    _show_detected_settings(detected)

    # Get user confirmation
    if not skip_confirm:
        choice = _prompt_confirmation()
        if choice == "n":
            _show_cancelled()
            return None
        if choice == "customize":
            # TODO: Implement customize flow (redirect to codeflash init)
            console.print("\n💡 Run [cyan]codeflash init[/cyan] for full customization.\n")
            return None

    # Handle API key
    if not skip_api_key:
        api_key_ok = _handle_api_key()
        if not api_key_ok:
            return None

    # Save config
    success, message = write_config(detected)
    if success:
        console.print(f"\n✅ {message}\n")
    else:
        console.print(f"\n⚠️  {message}\n")
        console.print("Continuing with detected settings (not saved).\n")

    # Create/update args namespace
    if args is None:
        args = Namespace()

    # Populate args with detected values
    args.module_root = str(detected.module_root)
    args.tests_root = str(detected.tests_root) if detected.tests_root else None
    args.project_root = str(detected.project_root)
    args.formatter_cmds = detected.formatter_cmds
    args.ignore_paths = [str(p) for p in detected.ignore_paths]
    args.pytest_cmd = detected.test_runner
    args.language = detected.language

    # Set defaults for other common args
    if not hasattr(args, "disable_telemetry"):
        args.disable_telemetry = False
    if not hasattr(args, "git_remote"):
        args.git_remote = "origin"

    return args


def _show_welcome() -> None:
    """Show welcome message for first-time users."""
    welcome_panel = Panel(
        Text(
            "⚡ Welcome to Codeflash!\n\nI've auto-detected your project settings.\nThis will only take a moment.",
            style="bold cyan",
            justify="center",
        ),
        title="🚀 First-Time Setup",
        border_style="bright_cyan",
        padding=(1, 2),
    )
    console.print(welcome_panel)
    console.print()


def _show_detected_settings(detected: detect_project) -> None:
    """Display detected settings in a nice table."""
    from codeflash.setup.detector import DetectedProject

    if not isinstance(detected, DetectedProject):
        return

    # Create settings table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="cyan", width=15)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    display_dict = detected.to_display_dict()
    details = detected.detection_details

    for key, value in display_dict.items():
        source = details.get(key.lower().replace(" ", "_"), "")
        # Truncate long sources
        if len(source) > 30:
            source = source[:27] + "..."
        table.add_row(key, value, f"({source})" if source else "")

    settings_panel = Panel(table, title="🔍 Auto-Detected Settings", border_style="bright_blue", padding=(1, 2))
    console.print(settings_panel)
    console.print()


def _prompt_confirmation() -> str:
    """Prompt user for confirmation.

    Returns:
        "y" for yes, "n" for no, "customize" for customization.

    """
    # Check if we're in a non-interactive environment
    if not sys.stdin.isatty():
        console.print("⚠️  Non-interactive environment detected. Use --yes to skip confirmation.")
        return "n"

    console.print("? [bold]Proceed with these settings?[/bold]")
    console.print("  [green]Y[/green] - Yes, save and continue")
    console.print("  [yellow]n[/yellow] - No, cancel")
    console.print("  [cyan]c[/cyan] - Customize (run full setup)")
    console.print()

    try:
        choice = console.input("[bold]Your choice[/bold] [green][Y][/green]/n/c: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "n"

    if choice in ("", "y", "yes"):
        return "y"
    if choice in ("c", "customize"):
        return "customize"
    return "n"


def _handle_api_key() -> bool:
    """Handle API key setup if not already configured.

    Returns:
        True if API key is available, False if user cancelled.

    """
    from codeflash.code_utils.env_utils import get_codeflash_api_key

    # Check for existing API key
    try:
        existing_key = get_codeflash_api_key()
        if existing_key:
            display_key = f"{existing_key[:3]}****{existing_key[-4:]}"
            console.print(f"✅ Found API key: [green]{display_key}[/green]\n")
            return True
    except OSError:
        pass

    # Prompt for API key
    console.print("🔑 [bold]API Key Required[/bold]")
    console.print("   Get your API key at: [cyan]https://app.codeflash.ai/app/apikeys[/cyan]\n")

    try:
        api_key = console.input("   Enter API key (or press Enter to open browser): ").strip()
    except (KeyboardInterrupt, EOFError):
        return False

    if not api_key:
        # Open browser
        import click

        click.launch("https://app.codeflash.ai/app/apikeys")
        console.print("\n   Opening browser...")
        try:
            api_key = console.input("   Enter API key: ").strip()
        except (KeyboardInterrupt, EOFError):
            return False

    if not api_key:
        console.print("\n⚠️  API key required. Run [cyan]codeflash init[/cyan] to set up.\n")
        return False

    if not api_key.startswith("cf-"):
        console.print("\n⚠️  Invalid API key format. Should start with 'cf-'.\n")
        return False

    # Save API key to environment
    os.environ["CODEFLASH_API_KEY"] = api_key

    # Try to save to shell rc
    try:
        from codeflash.code_utils.shell_utils import save_api_key_to_rc
        from codeflash.either import is_successful

        result = save_api_key_to_rc(api_key)
        if is_successful(result):
            console.print(f"\n✅ API key saved. {result.unwrap()}\n")
        else:
            console.print(f"\n⚠️  Could not save to shell: {result.failure()}")
            console.print("   API key set for this session only.\n")
    except Exception:
        console.print("\n✅ API key set for this session.\n")

    return True


def _show_detection_error(error: str) -> None:
    """Show error message when detection fails."""
    error_panel = Panel(
        Text(
            f"❌ Could not auto-detect project settings.\n\n"
            f"Error: {error}\n\n"
            "Please run [cyan]codeflash init[/cyan] for manual setup.",
            style="red",
        ),
        title="⚠️ Detection Failed",
        border_style="red",
        padding=(1, 2),
    )
    console.print(error_panel)


def _show_cancelled() -> None:
    """Show cancellation message."""
    console.print("\n⏹️  Setup cancelled. Run [cyan]codeflash init[/cyan] when ready.\n")
