"""JavaScript/TypeScript project initialization for Codeflash."""

# TODO:{claude} move to language support directory
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Union

import click
import inquirer
from git import InvalidGitRepositoryError, Repo
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console
from codeflash.code_utils.code_utils import validate_relative_directory_path
from codeflash.code_utils.compat import LF
from codeflash.code_utils.git_utils import get_git_remotes
from codeflash.code_utils.shell_utils import get_shell_rc_path, is_powershell
from codeflash.telemetry.posthog_cf import ph


class ProjectLanguage(Enum):
    """Supported project languages."""

    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()


class JsPackageManager(Enum):
    """JavaScript/TypeScript package managers."""

    NPM = auto()
    YARN = auto()
    PNPM = auto()
    BUN = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class JSSetupInfo:
    """Setup info for JavaScript/TypeScript projects.

    Only stores values that override auto-detection or user preferences.
    Most config is auto-detected from package.json and project structure.
    """

    # Override values (None means use auto-detected value)
    module_root_override: Union[str, None] = None
    formatter_override: Union[list[str], None] = None

    # User preferences (stored in config only if non-default)
    git_remote: str = "origin"
    disable_telemetry: bool = False
    ignore_paths: list[str] | None = None
    benchmarks_root: Union[str, None] = None


# Import theme from cmd_init to avoid duplication
def _get_theme():
    """Get the CodeflashTheme - imported lazily to avoid circular imports."""
    from codeflash.cli_cmds.cmd_init import CodeflashTheme

    return CodeflashTheme()


def detect_project_language(project_root: Path | None = None) -> ProjectLanguage:
    """Detect the primary language of the project.

    Args:
        project_root: Root directory to check. Defaults to current directory.

    Returns:
        ProjectLanguage enum value

    """
    root = project_root or Path.cwd()

    has_pyproject = (root / "pyproject.toml").exists()
    has_setup_py = (root / "setup.py").exists()
    has_package_json = (root / "package.json").exists()
    has_tsconfig = (root / "tsconfig.json").exists()

    # TypeScript project (tsconfig.json is definitive)
    if has_tsconfig:
        return ProjectLanguage.TYPESCRIPT

    # JavaScript project - package.json without Python-specific files takes priority
    # Note: If both package.json and pyproject.toml exist, check for typical JS project indicators
    if has_package_json:
        # If no Python config files, it's definitely JavaScript
        if not has_pyproject and not has_setup_py:
            return ProjectLanguage.JAVASCRIPT

        # If package.json exists with Python files, check for JS-specific indicators
        # Common React/Node patterns indicate a JS project
        js_indicators = [
            (root / "node_modules").exists(),
            (root / ".npmrc").exists(),
            (root / "yarn.lock").exists(),
            (root / "package-lock.json").exists(),
            (root / "pnpm-lock.yaml").exists(),
            (root / "bun.lockb").exists(),
            (root / "bun.lock").exists(),
        ]
        if any(js_indicators):
            return ProjectLanguage.JAVASCRIPT

    # Python project (default)
    return ProjectLanguage.PYTHON


def determine_js_package_manager(project_root: Path) -> JsPackageManager:
    """Determine which JavaScript package manager is being used based on lock files."""
    if (project_root / "bun.lockb").exists() or (project_root / "bun.lock").exists():
        return JsPackageManager.BUN
    if (project_root / "pnpm-lock.yaml").exists():
        return JsPackageManager.PNPM
    if (project_root / "yarn.lock").exists():
        return JsPackageManager.YARN
    if (project_root / "package-lock.json").exists():
        return JsPackageManager.NPM
    # Default to npm if package.json exists but no lock file
    if (project_root / "package.json").exists():
        return JsPackageManager.NPM
    return JsPackageManager.UNKNOWN


def init_js_project(language: ProjectLanguage) -> None:
    """Initialize Codeflash for a JavaScript/TypeScript project."""
    from codeflash.cli_cmds.cmd_init import install_github_actions, install_github_app, prompt_api_key

    lang_name = "TypeScript" if language == ProjectLanguage.TYPESCRIPT else "JavaScript"

    lang_panel = Panel(
        Text(
            f"📦 Detected {lang_name} project!\n\nI'll help you set up Codeflash for your project.",
            style="cyan",
            justify="center",
        ),
        title=f"🟨 {lang_name} Setup",
        border_style="bright_yellow",
    )
    console.print(lang_panel)
    console.print()

    did_add_new_key = prompt_api_key()

    should_modify, _config = should_modify_package_json_config()

    # Default git remote
    git_remote = "origin"

    if should_modify:
        setup_info = collect_js_setup_info(language)
        git_remote = setup_info.git_remote or "origin"
        configured = configure_package_json(setup_info)
        if not configured:
            apologize_and_exit()

    install_github_app(git_remote)

    install_github_actions(override_formatter_check=True)

    # Show completion message
    usage_table = Table(show_header=False, show_lines=False, border_style="dim")
    usage_table.add_column("Command", style="cyan")
    usage_table.add_column("Description", style="white")

    usage_table.add_row("codeflash --file <path-to-file> --function <function-name>", "Optimize a specific function")
    usage_table.add_row("codeflash --all", "Optimize all functions in all files")
    usage_table.add_row("codeflash --help", "See all available options")

    completion_message = (
        f"⚡️ Codeflash is now set up for your {lang_name} project!\n\nYou can now run any of these commands:"
    )

    if did_add_new_key:
        completion_message += (
            "\n\n🐚 Don't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!"
        )
        if os.name == "nt":
            reload_cmd = f". {get_shell_rc_path()}" if is_powershell() else f"call {get_shell_rc_path()}"
        else:
            reload_cmd = f"source {get_shell_rc_path()}"
        completion_message += f"\nOr run: {reload_cmd}"

    completion_panel = Panel(
        Group(Text(completion_message, style="bold green"), Text(""), usage_table),
        title="🎉 Setup Complete!",
        border_style="bright_green",
        padding=(1, 2),
    )
    console.print(completion_panel)

    ph("cli-js-installation-successful", {"language": lang_name, "did_add_new_key": did_add_new_key})
    sys.exit(0)


def should_modify_package_json_config() -> tuple[bool, dict[str, Any] | None]:
    """Check if package.json has valid codeflash config for JS/TS projects."""
    from rich.prompt import Confirm

    package_json_path = Path.cwd() / "package.json"

    if not package_json_path.exists():
        click.echo("❌ No package.json found. Please run 'npm init' first.")
        apologize_and_exit()

    try:
        with package_json_path.open(encoding="utf8") as f:
            package_data = json.load(f)

        config = package_data.get("codeflash", {})

        if not config:
            return True, None

        # Check if module_root is valid (defaults to "." if not specified)
        module_root = config.get("moduleRoot", ".")
        if not Path(module_root).is_dir():
            return True, None

        # Config is valid - ask if user wants to reconfigure
        return Confirm.ask(
            "✅ A valid Codeflash config already exists in package.json. Do you want to re-configure it?",
            default=False,
            show_default=True,
        ), config
    except Exception:
        return True, None


def collect_js_setup_info(language: ProjectLanguage) -> JSSetupInfo:
    """Collect setup information for JavaScript/TypeScript projects.

    Uses auto-detection for most settings and only asks for overrides if needed.
    """
    from rich.prompt import Confirm

    from codeflash.cli_cmds.cmd_init import ask_for_telemetry, get_valid_subdirs
    from codeflash.code_utils.config_js import (
        detect_formatter,
        detect_module_root,
        detect_test_runner,
        get_package_json_data,
    )

    curdir = Path.cwd()

    if not os.access(curdir, os.W_OK):
        click.echo(f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}")
        sys.exit(1)

    lang_name = "TypeScript" if language == ProjectLanguage.TYPESCRIPT else "JavaScript"

    # Load package.json data for detection
    package_json_path = curdir / "package.json"
    package_data = get_package_json_data(package_json_path) or {}

    # Auto-detect values
    detected_module_root = detect_module_root(curdir, package_data)
    detected_test_runner = detect_test_runner(curdir, package_data)
    detected_formatter = detect_formatter(curdir, package_data)

    # Build detection summary
    formatter_display = detected_formatter[0] if detected_formatter else "none detected"
    detection_table = Table(show_header=False, box=None, padding=(0, 2))
    detection_table.add_column("Setting", style="cyan")
    detection_table.add_column("Value", style="green")
    detection_table.add_row("Module root", detected_module_root)
    detection_table.add_row("Test runner", detected_test_runner)
    detection_table.add_row("Formatter", formatter_display)

    detection_panel = Panel(
        Group(Text(f"Auto-detected settings for your {lang_name} project:\n", style="cyan"), detection_table),
        title="🔍 Auto-Detection Results",
        border_style="bright_blue",
    )
    console.print(detection_panel)
    console.print()

    # Ask if user wants to change any settings
    module_root_override = None
    formatter_override = None

    if Confirm.ask("Would you like to change any of these settings?", default=False):
        # Module root override
        valid_subdirs = get_valid_subdirs()
        curdir_option = f"current directory ({curdir})"
        custom_dir_option = "enter a custom directory…"
        keep_detected_option = f"✓ keep detected ({detected_module_root})"

        module_options = [
            keep_detected_option,
            *[d for d in valid_subdirs if d not in ("tests", "__tests__", "node_modules", detected_module_root)],
            curdir_option,
            custom_dir_option,
        ]

        module_questions = [
            inquirer.List(
                "module_root",
                message=f"Which directory contains your {lang_name} source code?",
                choices=module_options,
                default=keep_detected_option,
                carousel=True,
            )
        ]

        module_answers = inquirer.prompt(module_questions, theme=_get_theme())
        if not module_answers:
            apologize_and_exit()

        module_root_answer = module_answers["module_root"]
        if module_root_answer == keep_detected_option:
            pass  # Keep auto-detected value
        elif module_root_answer == curdir_option:
            module_root_override = "."
        elif module_root_answer == custom_dir_option:
            module_root_override = _prompt_custom_directory("module")
        else:
            module_root_override = module_root_answer

        ph("cli-js-module-root-provided", {"overridden": module_root_override is not None})

        # Formatter override
        formatter_questions = [
            inquirer.List(
                "formatter",
                message="Which code formatter do you use?",
                choices=[
                    (f"✓ keep detected ({formatter_display})", "keep"),
                    ("💅 prettier", "prettier"),
                    ("📐 eslint --fix", "eslint"),
                    ("🔧 other", "other"),
                    ("❌ don't use a formatter", "disabled"),
                ],
                default="keep",
                carousel=True,
            )
        ]

        formatter_answers = inquirer.prompt(formatter_questions, theme=_get_theme())
        if not formatter_answers:
            apologize_and_exit()

        formatter_choice = formatter_answers["formatter"]
        if formatter_choice != "keep":
            formatter_override = get_js_formatter_cmd(formatter_choice)

        ph("cli-js-formatter-provided", {"overridden": formatter_override is not None})

    # Git remote
    git_remote = _get_git_remote_for_setup()

    # Telemetry
    disable_telemetry = not ask_for_telemetry()

    return JSSetupInfo(
        module_root_override=module_root_override,
        formatter_override=formatter_override,
        git_remote=git_remote,
        disable_telemetry=disable_telemetry,
    )


def _prompt_custom_directory(dir_type: str) -> str:
    """Prompt for a custom directory path."""
    while True:
        custom_questions = [
            inquirer.Path(
                "custom_path",
                message=f"Enter the path to your {dir_type} directory",
                path_type=inquirer.Path.DIRECTORY,
                exists=True,
            )
        ]

        custom_answers = inquirer.prompt(custom_questions, theme=_get_theme())
        if not custom_answers:
            apologize_and_exit()

        custom_path_str = str(custom_answers["custom_path"])
        is_valid, error_msg = validate_relative_directory_path(custom_path_str)
        if is_valid:
            return custom_path_str

        click.echo(f"❌ Invalid path: {error_msg}")
        click.echo("Please enter a valid relative directory path.")
        console.print()


def _get_git_remote_for_setup() -> str:
    """Get git remote for project setup."""
    try:
        repo = Repo(Path.cwd(), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if not git_remotes:
            return ""

        if len(git_remotes) == 1:
            return git_remotes[0]

        git_panel = Panel(
            Text(
                "🔗 Configure Git Remote for Pull Requests.\n\nCodeflash will use this remote to create pull requests.",
                style="blue",
            ),
            title="🔗 Git Remote Setup",
            border_style="bright_blue",
        )
        console.print(git_panel)
        console.print()

        git_questions = [
            inquirer.List(
                "git_remote",
                message="Which git remote should Codeflash use?",
                choices=git_remotes,
                default="origin",
                carousel=True,
            )
        ]

        git_answers = inquirer.prompt(git_questions, theme=_get_theme())
        return git_answers["git_remote"] if git_answers else git_remotes[0]
    except InvalidGitRepositoryError:
        return ""


def get_js_formatter_cmd(formatter: str) -> list[str]:
    """Get formatter commands for JavaScript/TypeScript."""
    if formatter == "prettier":
        return ["npx prettier --write $file"]
    if formatter == "eslint":
        return ["npx eslint --fix $file"]
    if formatter == "other":
        click.echo("🔧 In package.json, please replace 'your-formatter' with your formatter command.")
        return ["your-formatter $file"]
    return ["disabled"]


def configure_package_json(setup_info: JSSetupInfo) -> bool:
    """Configure codeflash section in package.json for JavaScript/TypeScript projects.

    Only writes minimal config - values that override auto-detection or user preferences.
    Auto-detected values (language, moduleRoot, testRunner, formatter) are NOT stored
    unless explicitly overridden by the user.
    """
    package_json_path = Path.cwd() / "package.json"

    try:
        with package_json_path.open(encoding="utf8") as f:
            package_data = json.load(f)
    except FileNotFoundError:
        click.echo("❌ No package.json found. Please run 'npm init' first.")
        return False
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid package.json: {e}")
        return False

    # Build minimal codeflash config using camelCase (JS convention)
    # Only include values that override auto-detection or are user preferences
    codeflash_config: dict[str, Any] = {}

    # Module root override (only if user changed from auto-detected)
    if setup_info.module_root_override is not None:
        codeflash_config["moduleRoot"] = setup_info.module_root_override

    # Formatter override (only if user changed from auto-detected)
    if setup_info.formatter_override is not None:
        if setup_info.formatter_override != ["disabled"]:
            codeflash_config["formatterCmds"] = setup_info.formatter_override
        else:
            codeflash_config["formatterCmds"] = []

    # Git remote (only if not default "origin")
    if setup_info.git_remote and setup_info.git_remote not in ("", "origin"):
        codeflash_config["gitRemote"] = setup_info.git_remote

    # User preferences
    if setup_info.disable_telemetry:
        codeflash_config["disableTelemetry"] = True

    if setup_info.ignore_paths:
        codeflash_config["ignorePaths"] = setup_info.ignore_paths

    if setup_info.benchmarks_root:
        codeflash_config["benchmarksRoot"] = setup_info.benchmarks_root

    # Only write codeflash section if there's something to write
    if codeflash_config:
        package_data["codeflash"] = codeflash_config
        action = "Updated"
    else:
        # Remove codeflash section if empty (all auto-detected)
        if "codeflash" in package_data:
            del package_data["codeflash"]
        action = "Configured"

    try:
        with package_json_path.open("w", encoding="utf8") as f:
            json.dump(package_data, f, indent=2)
            f.write("\n")  # Trailing newline
    except OSError as e:
        click.echo(f"❌ Failed to update package.json: {e}")
        return False
    else:
        if codeflash_config:
            click.echo(f"✅ {action} Codeflash configuration in {package_json_path}")
        else:
            click.echo("✅ Using auto-detected configuration (no overrides needed)")
        click.echo()
        return True


# ============================================================================
# GitHub Actions Workflow Helpers for JS/TS
# ============================================================================


def is_codeflash_dependency(project_root: Path) -> bool:
    """Check if codeflash is listed as a dependency in package.json."""
    package_json_path = project_root / "package.json"
    if not package_json_path.exists():
        return False

    try:
        with package_json_path.open(encoding="utf8") as f:
            package_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    deps = package_data.get("dependencies", {})
    dev_deps = package_data.get("devDependencies", {})
    return "codeflash" in deps or "codeflash" in dev_deps


def get_js_runtime_setup_steps(pkg_manager: JsPackageManager) -> str:
    """Generate the appropriate Node.js/Bun setup steps for GitHub Actions.

    Returns properly indented YAML steps for the workflow template.
    """
    if pkg_manager == JsPackageManager.BUN:
        return """- name: 🥟 Setup Bun
        uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest"""

    if pkg_manager == JsPackageManager.PNPM:
        return """- name: 📦 Setup pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 9
      - name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'"""

    if pkg_manager == JsPackageManager.YARN:
        return """- name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'yarn'"""

    # NPM or UNKNOWN
    return """- name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'"""


def get_js_codeflash_install_step(pkg_manager: JsPackageManager, *, is_dependency: bool) -> str:
    """Generate the codeflash installation step if not already a dependency.

    Args:
        pkg_manager: The package manager being used.
        is_dependency: Whether codeflash is already in package.json dependencies.

    Returns:
        YAML step string for installing codeflash, or empty string if not needed.

    """
    if is_dependency:
        # Codeflash will be installed with other dependencies
        return ""

    # Need to install codeflash separately
    if pkg_manager == JsPackageManager.BUN:
        return """- name: 📥 Install Codeflash
        run: bun add -g codeflash"""

    if pkg_manager == JsPackageManager.PNPM:
        return """- name: 📥 Install Codeflash
        run: pnpm add -g codeflash"""

    if pkg_manager == JsPackageManager.YARN:
        return """- name: 📥 Install Codeflash
        run: yarn global add codeflash"""

    # NPM or UNKNOWN
    return """- name: 📥 Install Codeflash
        run: npm install -g codeflash"""


def get_js_codeflash_run_command(pkg_manager: JsPackageManager, *, is_dependency: bool) -> str:
    """Generate the codeflash run command for GitHub Actions.

    Args:
        pkg_manager: The package manager being used.
        is_dependency: Whether codeflash is in package.json dependencies.

    Returns:
        Command string to run codeflash.

    """
    if is_dependency:
        # Use package manager's run command for local dependency
        if pkg_manager == JsPackageManager.BUN:
            return "bun run codeflash"
        if pkg_manager == JsPackageManager.PNPM:
            return "pnpm exec codeflash"
        if pkg_manager == JsPackageManager.YARN:
            return "yarn codeflash"
        # NPM
        return "npx codeflash"

    # Globally installed - just run directly
    return "codeflash"


def get_js_runtime_setup_string(pkg_manager: JsPackageManager) -> str:
    """Generate the appropriate Node.js setup step for GitHub Actions.

    Deprecated: Use get_js_runtime_setup_steps instead.
    """
    return get_js_runtime_setup_steps(pkg_manager)


def get_js_dependency_installation_commands(pkg_manager: JsPackageManager) -> str:
    """Generate commands to install JavaScript/TypeScript dependencies."""
    if pkg_manager == JsPackageManager.BUN:
        return "bun install"
    if pkg_manager == JsPackageManager.PNPM:
        return "pnpm install"
    if pkg_manager == JsPackageManager.YARN:
        return "yarn install"
    # NPM or UNKNOWN
    return "npm ci"


def get_js_codeflash_command(pkg_manager: JsPackageManager) -> str:
    """Generate the appropriate codeflash command for JavaScript/TypeScript projects."""
    if pkg_manager == JsPackageManager.BUN:
        return "bunx codeflash"
    if pkg_manager == JsPackageManager.PNPM:
        return "pnpm dlx codeflash"
    if pkg_manager == JsPackageManager.YARN:
        return "yarn dlx codeflash"
    # NPM or UNKNOWN
    return "npx codeflash"
