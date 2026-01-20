"""JavaScript/TypeScript project initialization for Codeflash."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

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

if TYPE_CHECKING:
    pass


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

    tests_root is optional because Jest auto-discovers tests.
    """

    module_root: str
    tests_root: Union[str, None]  # Optional - Jest auto-discovers
    formatter: Union[str, list[str]]
    git_remote: str
    enable_telemetry: bool
    language: ProjectLanguage = ProjectLanguage.JAVASCRIPT


# Import theme from cmd_init to avoid duplication
def _get_theme():  # noqa: ANN202
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

    # TypeScript project
    if has_tsconfig:
        return ProjectLanguage.TYPESCRIPT

    # Pure JS project (has package.json but no Python files)
    if has_package_json and not has_pyproject and not has_setup_py:
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
    from codeflash.cli_cmds.cmd_init import (
        ask_for_telemetry,
        get_valid_subdirs,
        install_github_actions,
        install_github_app,
        prompt_api_key,
    )

    lang_name = "TypeScript" if language == ProjectLanguage.TYPESCRIPT else "JavaScript"

    lang_panel = Panel(
        Text(
            f"📦 Detected {lang_name} project!\n\n"
            "I'll help you set up Codeflash for your project.",
            style="cyan",
            justify="center",
        ),
        title=f"🟨 {lang_name} Setup",
        border_style="bright_yellow",
    )
    console.print(lang_panel)
    console.print()

    did_add_new_key = prompt_api_key()

    should_modify, config = should_modify_package_json_config()

    git_remote = config.get("gitRemote", "origin") if config else "origin"

    if should_modify:
        setup_info = collect_js_setup_info(language)
        git_remote = setup_info.git_remote
        configured = configure_package_json(setup_info)
        if not configured:
            apologize_and_exit()

    install_github_app(git_remote)

    install_github_actions(override_formatter_check=True)

    # Show completion message
    usage_table = Table(show_header=False, show_lines=False, border_style="dim")
    usage_table.add_column("Command", style="cyan")
    usage_table.add_column("Description", style="white")

    usage_table.add_row(
        "codeflash --file <path-to-file> --function <function-name>", "Optimize a specific function"
    )
    usage_table.add_row("codeflash --all", "Optimize all functions in all files")
    usage_table.add_row("codeflash --help", "See all available options")

    completion_message = f"⚡️ Codeflash is now set up for your {lang_name} project!\n\nYou can now run any of these commands:"

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
    """Collect setup information for JavaScript/TypeScript projects."""
    from codeflash.cli_cmds.cmd_init import ask_for_telemetry, get_valid_subdirs

    curdir = Path.cwd()

    if not os.access(curdir, os.W_OK):
        click.echo(f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}")
        sys.exit(1)

    lang_name = "TypeScript" if language == ProjectLanguage.TYPESCRIPT else "JavaScript"

    # Module root selection
    valid_subdirs = get_valid_subdirs()
    curdir_option = f"current directory ({curdir})"
    custom_dir_option = "enter a custom directory…"
    module_options = [
        *[d for d in valid_subdirs if d not in ("tests", "__tests__", "node_modules")],
        curdir_option,
        custom_dir_option,
    ]

    # Try to detect src directory
    default_module = "src" if "src" in valid_subdirs else module_options[0] if module_options else curdir_option

    info_panel = Panel(
        Text(
            f"📁 Let's identify your {lang_name} source directory.\n\n"
            "This is usually 'src' or the root directory containing your source code.\n",
            style="cyan",
        ),
        title="🔍 Source Discovery",
        border_style="bright_blue",
    )
    console.print(info_panel)
    console.print()

    questions = [
        inquirer.List(
            "module_root",
            message=f"Which directory contains your {lang_name} source code?",
            choices=module_options,
            default=default_module,
            carousel=True,
        )
    ]

    answers = inquirer.prompt(questions, theme=_get_theme())
    if not answers:
        apologize_and_exit()

    module_root_answer = answers["module_root"]
    if module_root_answer == curdir_option:
        module_root = "."
    elif module_root_answer == custom_dir_option:
        module_root = _prompt_custom_directory("module")
    else:
        module_root = module_root_answer

    ph("cli-js-module-root-provided")

    # Tests root - OPTIONAL for Jest
    tests_panel = Panel(
        Text(
            "🧪 Test Directory (Optional)\n\n"
            "Jest auto-discovers tests from patterns like *.test.js and __tests__/.\n"
            "You can specify a tests directory or skip this step.",
            style="green",
        ),
        title="🧪 Test Discovery",
        border_style="bright_green",
    )
    console.print(tests_panel)
    console.print()

    skip_option = "⏭️  Skip (Jest will auto-discover tests)"
    test_suggestions = [d for d in valid_subdirs if d in ("tests", "__tests__", "test", "spec")]
    test_options = [skip_option, *test_suggestions, custom_dir_option]

    tests_questions = [
        inquirer.List(
            "tests_root",
            message="Where are your tests located?",
            choices=test_options,
            default=skip_option,
            carousel=True,
        )
    ]

    tests_answers = inquirer.prompt(tests_questions, theme=_get_theme())
    if not tests_answers:
        apologize_and_exit()

    tests_root_answer = tests_answers["tests_root"]
    if tests_root_answer == skip_option:
        tests_root = None
    elif tests_root_answer == custom_dir_option:
        tests_root = _prompt_custom_directory("tests")
    else:
        tests_root = tests_root_answer

    ph("cli-js-tests-root-provided", {"skipped": tests_root is None})

    # Formatter selection
    formatter_panel = Panel(
        Text(
            "🎨 Let's configure your code formatter.\n\n" "Codeflash will use this to format optimized code.",
            style="magenta",
        ),
        title="🎨 Code Formatter",
        border_style="bright_magenta",
    )
    console.print(formatter_panel)
    console.print()

    formatter_questions = [
        inquirer.List(
            "formatter",
            message="Which code formatter do you use?",
            choices=[
                ("💅 prettier (Recommended)", "prettier"),
                ("📐 eslint --fix", "eslint"),
                ("🔧 other", "other"),
                ("❌ don't use a formatter", "disabled"),
            ],
            default="prettier",
            carousel=True,
        )
    ]

    formatter_answers = inquirer.prompt(formatter_questions, theme=_get_theme())
    if not formatter_answers:
        apologize_and_exit()

    formatter = formatter_answers["formatter"]

    # Git remote
    git_remote = _get_git_remote_for_setup(module_root)

    # Telemetry
    enable_telemetry = ask_for_telemetry()

    return JSSetupInfo(
        module_root=str(module_root),
        tests_root=str(tests_root) if tests_root else None,
        formatter=get_js_formatter_cmd(formatter),
        git_remote=str(git_remote),
        enable_telemetry=enable_telemetry,
        language=language,
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


def _get_git_remote_for_setup(module_root: str) -> str:
    """Get git remote for project setup."""
    try:
        repo = Repo(str(module_root), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if not git_remotes:
            return ""

        if len(git_remotes) == 1:
            return git_remotes[0]

        git_panel = Panel(
            Text(
                "🔗 Configure Git Remote for Pull Requests.\n\n"
                "Codeflash will use this remote to create pull requests.",
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
    """Configure codeflash section in package.json for JavaScript/TypeScript projects."""
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

    # Build codeflash config using camelCase (JS convention)
    codeflash_config: dict[str, Any] = {
        "moduleRoot": setup_info.module_root,
    }

    # testsRoot is optional - Jest auto-discovers tests
    if setup_info.tests_root:
        codeflash_config["testsRoot"] = setup_info.tests_root

    # Formatter
    if setup_info.formatter != ["disabled"]:
        codeflash_config["formatterCmds"] = setup_info.formatter

    # Git remote (only if not default)
    if setup_info.git_remote and setup_info.git_remote not in ("", "origin"):
        codeflash_config["gitRemote"] = setup_info.git_remote

    # Telemetry
    if not setup_info.enable_telemetry:
        codeflash_config["disableTelemetry"] = True

    # Language
    codeflash_config["language"] = "typescript" if setup_info.language == ProjectLanguage.TYPESCRIPT else "javascript"

    # Add/update codeflash section
    package_data["codeflash"] = codeflash_config

    try:
        with package_json_path.open("w", encoding="utf8") as f:
            json.dump(package_data, f, indent=2)
            f.write("\n")  # Trailing newline

        click.echo(f"✅ Added Codeflash configuration to {package_json_path}")
        click.echo()
        return True
    except OSError as e:
        click.echo(f"❌ Failed to update package.json: {e}")
        return False


# ============================================================================
# GitHub Actions Workflow Helpers for JS/TS
# ============================================================================


def get_js_runtime_setup_string(pkg_manager: JsPackageManager) -> str:
    """Generate the appropriate Node.js setup step for GitHub Actions."""
    if pkg_manager == JsPackageManager.BUN:
        return """name: 🥟 Setup Bun
        uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest"""
    if pkg_manager == JsPackageManager.PNPM:
        return """name: 📦 Setup pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 9
      - name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'"""
    if pkg_manager == JsPackageManager.YARN:
        return """name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'yarn'"""
    # NPM or UNKNOWN
    return """name: 🟢 Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'"""


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
