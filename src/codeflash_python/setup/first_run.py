"""First-run experience for Codeflash.

This module handles the seamless first-run experience:
1. Auto-detect project settings
2. Display detected settings
3. Quick confirmation
4. API key setup
5. Save config and continue

Usage:
    from codeflash_python.setup.first_run import handle_first_run, is_first_run

    if is_first_run():
        args = handle_first_run(args)
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

try:
    from codeflash_python.setup.config_writer import write_config
    from codeflash_python.setup.detector import detect_project, has_existing_config
except ImportError:
    # Stubs for missing modules
    def detect_project() -> Any:
        msg = "detector module not available"
        raise NotImplementedError(msg)

    def has_existing_config(project_root: Any) -> tuple[bool, None]:
        return False, None

    def write_config(detected: Any) -> tuple[bool, str]:
        return False, "config_writer not available"


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
        show_detection_error(str(e))
        return None

    # Show welcome message
    show_welcome()

    # Show detected settings
    show_detected_settings(detected)

    # Get user confirmation
    if not skip_confirm:
        choice = prompt_confirmation()
        if choice == "n":
            show_cancelled()
            return None
        if choice == "customize":
            print("\nRun codeflash init for full customization.\n")
            return None

    # Handle API key
    if not skip_api_key:
        api_key_ok = handle_api_key()
        if not api_key_ok:
            return None

    # Save config
    config_result = write_config(detected)
    # Handle Result type from write_config
    if hasattr(config_result, "is_ok") and config_result.is_ok():  # type: ignore[union-attr]
        print(f"\n{config_result.unwrap()}\n")  # type: ignore[union-attr]
    elif hasattr(config_result, "error"):
        print(f"\n{config_result.error}\n")
        print("Continuing with detected settings (not saved).\n")
    else:
        # Handle tuple fallback case
        success, message = config_result  # type: ignore[misc]
        if success:
            print(f"\n{message}\n")
        else:
            print(f"\n{message}\n")
            print("Continuing with detected settings (not saved).\n")

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


def show_welcome() -> None:
    """Show welcome message for first-time users."""
    print("First-Time Setup")
    print("Welcome to Codeflash!")
    print()
    print("I've auto-detected your project settings.")
    print("This will only take a moment.")
    print()


def show_detected_settings(detected: detect_project) -> None:  # type: ignore[valid-type]
    """Display detected settings in a nice table."""
    try:
        from codeflash_python.setup.detector import DetectedProject
    except ImportError:
        # Stub
        class DetectedProject:
            pass

    if not isinstance(detected, DetectedProject):
        return

    display_dict = detected.to_display_dict()
    details = detected.detection_details

    print("Auto-Detected Settings")
    print(f"  {'Setting':<15}  {'Value':<30}  {'Source'}")
    print(f"  {'-' * 15}  {'-' * 30}  {'-' * 20}")
    for key, value in display_dict.items():
        source = details.get(key.lower().replace(" ", "_"), "")
        if len(source) > 30:
            source = source[:27] + "..."
        source_str = f"({source})" if source else ""
        print(f"  {key:<15}  {value:<30}  {source_str}")
    print()


def prompt_confirmation() -> str:
    """Prompt user for confirmation.

    Returns:
        "y" for yes, "n" for no, "customize" for customization.

    """
    # Check if we're in a non-interactive environment
    if not sys.stdin.isatty():
        print("Non-interactive environment detected. Use --yes to skip confirmation.")
        return "n"

    print("? Proceed with these settings?")
    print("  Y - Yes, save and continue")
    print("  n - No, cancel")
    print("  c - Customize (run full setup)")
    print()

    try:
        choice = input("Your choice [Y]/n/c: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "n"

    if choice in ("", "y", "yes"):
        return "y"
    if choice in ("c", "customize"):
        return "customize"
    return "n"


def handle_api_key() -> bool:
    """Handle API key setup if not already configured.

    Returns:
        True if API key is available, False if user cancelled.

    """
    try:
        from codeflash_python.code_utils.env_utils import get_codeflash_api_key
    except ImportError:
        # Stub
        def get_codeflash_api_key() -> str | None:
            return os.getenv("CODEFLASH_API_KEY")

    # Check for existing API key
    try:
        existing_key = get_codeflash_api_key()
        if existing_key:
            display_key = f"{existing_key[:3]}****{existing_key[-4:]}"
            print(f"Found API key: {display_key}\n")
            return True
    except OSError:
        pass

    # Prompt for API key
    print("API Key Required")
    print("   Get your API key at: https://app.codeflash.ai/app/apikeys\n")

    try:
        api_key = input("   Enter API key (or press Enter to open browser): ").strip()
    except (KeyboardInterrupt, EOFError):
        return False

    if not api_key:
        # Open browser
        import click

        click.launch("https://app.codeflash.ai/app/apikeys")
        print("\n   Opening browser...")
        try:
            api_key = input("   Enter API key: ").strip()
        except (KeyboardInterrupt, EOFError):
            return False

    if not api_key:
        print("\nAPI key required. Run codeflash init to set up.\n")
        return False

    if not api_key.startswith("cf-"):
        print("\nInvalid API key format. Should start with 'cf-'.\n")
        return False

    # Save API key to environment
    os.environ["CODEFLASH_API_KEY"] = api_key

    # Try to save to shell rc
    try:
        from codeflash_python.code_utils.shell_utils import save_api_key_to_rc

        result = save_api_key_to_rc(api_key)
        if hasattr(result, "is_ok") and result.is_ok():
            print(f"\nAPI key saved. {result.unwrap()}\n")
        elif hasattr(result, "error"):
            print(f"\nCould not save to shell: {result.error}")
            print("   API key set for this session only.\n")
        else:
            print("\nAPI key set for this session.\n")
    except Exception:
        print("\nAPI key set for this session.\n")

    return True


def show_detection_error(error: str) -> None:
    """Show error message when detection fails."""
    print("Detection Failed")
    print("Could not auto-detect project settings.\n")
    print(f"Error: {error}\n")
    print("Please run codeflash init for manual setup.")


def show_cancelled() -> None:
    """Show cancellation message."""
    print("\nSetup cancelled. Run codeflash init when ready.\n")
