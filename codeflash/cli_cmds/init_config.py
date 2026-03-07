from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import click
import inquirer
import inquirer.themes
import tomlkit
from pydantic.dataclasses import dataclass

from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import check_formatter_installed
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.telemetry.posthog_cf import ph


@dataclass(frozen=True)
class CLISetupInfo:
    """Setup info for Python projects."""

    module_root: str
    tests_root: str
    benchmarks_root: Union[str, None]
    ignore_paths: list[str]
    formatter: Union[str, list[str]]
    git_remote: str
    enable_telemetry: bool


@dataclass(frozen=True)
class VsCodeSetupInfo:
    """Setup info for VSCode extension initialization."""

    module_root: str
    tests_root: str
    formatter: Union[str, list[str]]


# Custom theme for better UX
class CodeflashTheme(inquirer.themes.Default):
    def __init__(self) -> None:
        super().__init__()
        self.Question.mark_color = inquirer.themes.term.yellow
        self.Question.brackets_color = inquirer.themes.term.bright_blue
        self.Question.default_color = inquirer.themes.term.bright_cyan
        self.List.selection_color = inquirer.themes.term.bright_blue
        self.Checkbox.selection_color = inquirer.themes.term.bright_blue
        self.Checkbox.selected_icon = "✅"
        self.Checkbox.unselected_icon = "⬜"


# common sections between normal mode and lsp mode
class CommonSections(Enum):
    module_root = "module_root"
    tests_root = "tests_root"
    formatter_cmds = "formatter_cmds"

    def get_toml_key(self) -> str:
        return self.value.replace("_", "-")


ignore_subdirs = {
    "venv",
    "node_modules",
    "dist",
    "build",
    "build_temp",
    "build_scripts",
    "env",
    "logs",
    "tmp",
    "__pycache__",
}


@lru_cache(maxsize=1)
def get_valid_subdirs(current_dir: Optional[Path] = None) -> list[str]:

    path_str = str(current_dir) if current_dir else "."
    return [
        entry.name
        for entry in os.scandir(path_str)
        if entry.is_dir() and not entry.name.startswith((".", "__")) and entry.name not in ignore_subdirs
    ]


def get_suggestions(section: str) -> tuple[list[str], Optional[str]]:
    valid_subdirs = get_valid_subdirs()
    if section == CommonSections.module_root:
        return [d for d in valid_subdirs if d != "tests"], None
    if section == CommonSections.tests_root:
        default = "tests" if "tests" in valid_subdirs else None
        return valid_subdirs, default
    if section == CommonSections.formatter_cmds:
        return ["disabled", "ruff", "black"], "disabled"
    msg = f"Unknown section: {section}"
    raise ValueError(msg)


def config_found(pyproject_toml_path: Union[str, Path]) -> tuple[bool, str]:
    pyproject_toml_path = Path(pyproject_toml_path)

    if not pyproject_toml_path.exists():
        return False, f"Configuration file not found: {pyproject_toml_path}"

    if not pyproject_toml_path.is_file():
        return False, f"Configuration file is not a file: {pyproject_toml_path}"

    if pyproject_toml_path.suffix != ".toml":
        return False, f"Configuration file is not a .toml file: {pyproject_toml_path}"

    return True, ""


def is_valid_pyproject_toml(pyproject_toml_path: Union[str, Path]) -> tuple[bool, dict[str, Any] | None, str]:
    pyproject_toml_path = Path(pyproject_toml_path)
    try:
        config, _ = parse_config_file(pyproject_toml_path)
    except Exception as e:
        return False, None, f"Failed to parse configuration: {e}"

    module_root = config.get("module_root")
    if not module_root:
        return False, config, "Missing required field: 'module_root'"

    if not Path(module_root).is_dir():
        return False, config, f"Invalid 'module_root': directory does not exist at {module_root}"

    tests_root = config.get("tests_root")
    if not tests_root:
        return False, config, "Missing required field: 'tests_root'"

    if not Path(tests_root).is_dir():
        return False, config, f"Invalid 'tests_root': directory does not exist at {tests_root}"

    return True, config, ""


def should_modify_pyproject_toml() -> tuple[bool, dict[str, Any] | None]:
    """Check if the current directory contains a valid pyproject.toml file with codeflash config.

    If it does, ask the user if they want to re-configure it.
    """
    from rich.prompt import Confirm

    pyproject_toml_path = Path.cwd() / "pyproject.toml"

    found, _ = config_found(pyproject_toml_path)
    if not found:
        return True, None

    valid, config, _message = is_valid_pyproject_toml(pyproject_toml_path)
    if not valid:
        # needs to be re-configured
        return True, None

    return Confirm.ask(
        "✅ A valid Codeflash config already exists in this project. Do you want to re-configure it?",
        default=False,
        show_default=True,
    ), config


def get_formatter_cmds(formatter: str) -> list[str]:
    if formatter == "black":
        return ["black $file"]
    if formatter == "ruff":
        return ["ruff check --exit-zero --fix $file", "ruff format $file"]
    if formatter == "other":
        click.echo(
            "🔧 In pyproject.toml, please replace 'your-formatter' with the command you use to format your code."
        )
        return ["your-formatter $file"]
    if formatter in {"don't use a formatter", "disabled"}:
        return ["disabled"]
    if " && " in formatter:
        return formatter.split(" && ")
    return [formatter]


# Create or update the pyproject.toml file with the Codeflash dependency & configuration
def configure_pyproject_toml(
    setup_info: Union[VsCodeSetupInfo, CLISetupInfo], config_file: Optional[Path] = None
) -> bool:
    for_vscode = isinstance(setup_info, VsCodeSetupInfo)
    toml_path = config_file or Path.cwd() / "pyproject.toml"
    try:
        with toml_path.open(encoding="utf8") as pyproject_file:
            pyproject_data = tomlkit.parse(pyproject_file.read())
    except FileNotFoundError:
        click.echo(
            f"I couldn't find a pyproject.toml in the current directory.{LF}"
            f"Please create a new empty pyproject.toml file here, OR if you use poetry then run `poetry init`, OR run `codeflash init` again from a directory with an existing pyproject.toml file."
        )
        return False

    codeflash_section = tomlkit.table()
    codeflash_section.add(tomlkit.comment("All paths are relative to this pyproject.toml's directory."))

    if for_vscode:
        for section in CommonSections:
            if hasattr(setup_info, section.value):
                codeflash_section[section.get_toml_key()] = getattr(setup_info, section.value)
    else:
        codeflash_section["module-root"] = setup_info.module_root
        codeflash_section["tests-root"] = setup_info.tests_root
        codeflash_section["ignore-paths"] = setup_info.ignore_paths
        if not setup_info.enable_telemetry:
            codeflash_section["disable-telemetry"] = not setup_info.enable_telemetry
        if setup_info.git_remote not in ["", "origin"]:
            codeflash_section["git-remote"] = setup_info.git_remote

    formatter = setup_info.formatter

    formatter_cmds = formatter if isinstance(formatter, list) else get_formatter_cmds(formatter)

    check_formatter_installed(formatter_cmds, exit_on_failure=False)
    codeflash_section["formatter-cmds"] = formatter_cmds
    # Add the 'codeflash' section, ensuring 'tool' section exists
    tool_section = pyproject_data.get("tool", tomlkit.table())

    if for_vscode:
        # merge the existing codeflash section, instead of overwriting it
        existing_codeflash = tool_section.get("codeflash", tomlkit.table())

        for key, value in codeflash_section.items():
            existing_codeflash[key] = value
        tool_section["codeflash"] = existing_codeflash
    else:
        tool_section["codeflash"] = codeflash_section

    pyproject_data["tool"] = tool_section

    with toml_path.open("w", encoding="utf8") as pyproject_file:
        pyproject_file.write(tomlkit.dumps(pyproject_data))
    click.echo(f"Added Codeflash configuration to {toml_path}")
    click.echo()
    return True


def create_empty_pyproject_toml(pyproject_toml_path: Path) -> None:
    ph("cli-create-pyproject-toml")
    lsp_mode = is_LSP_enabled()
    # Define a minimal pyproject.toml content
    new_pyproject_toml = tomlkit.document()
    new_pyproject_toml["tool"] = {"codeflash": {}}
    try:
        pyproject_toml_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")

        # Check if the pyproject.toml file was created
        if pyproject_toml_path.exists() and not lsp_mode:
            from rich.panel import Panel
            from rich.text import Text

            success_panel = Panel(
                Text(
                    f"✅ Created a pyproject.toml file at {pyproject_toml_path}\n\n"
                    "Your project is now ready for Codeflash configuration!",
                    style="green",
                    justify="center",
                ),
                title="🎉 Success!",
                border_style="bright_green",
            )
            console.print(success_panel)
            console.print("\n📍 Press any key to continue...")
            console.input()
        ph("cli-created-pyproject-toml")
    except OSError:
        click.echo("❌ Failed to create pyproject.toml. Please check your disk permissions and available space.")
        apologize_and_exit()


def ask_for_telemetry() -> bool:
    """Prompt the user to enable or disable telemetry."""
    from rich.prompt import Confirm

    return Confirm.ask(
        "⚡️ Help us improve Codeflash by sharing anonymous usage data (e.g. errors encountered)?",
        default=True,
        show_default=True,
    )
