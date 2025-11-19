from __future__ import annotations

import ast
import os
import subprocess
import sys
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import click
import git
import tomlkit
from git import Repo
from pydantic.dataclasses import dataclass
from rich.panel import Panel
from rich.text import Text

from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import check_formatter_installed, get_codeflash_api_key
from codeflash.code_utils.github_utils import get_github_secrets_page_url
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.telemetry.posthog_cf import ph
from codeflash.version import __version__ as version

if TYPE_CHECKING:
    from argparse import Namespace

CODEFLASH_LOGO: str = (
    f"{LF}"
    r"                   _          ___  _               _     " + f"{LF}"
    r"                  | |        / __)| |             | |    " + f"{LF}"
    r"  ____   ___    _ | |  ____ | |__ | |  ____   ___ | | _  " + f"{LF}"
    r" / ___) / _ \  / || | / _  )|  __)| | / _  | /___)| || \ " + f"{LF}"
    r"( (___ | |_| |( (_| |( (/ / | |   | |( ( | ||___ || | | |" + f"{LF}"
    r" \____) \___/  \____| \____)|_|   |_| \_||_|(___/ |_| |_|" + f"{LF}"
    f"{('v' + version).rjust(66)}{LF}"
    f"{LF}"
)


@dataclass(frozen=True)
class CLISetupInfo:
    module_root: str
    tests_root: str
    benchmarks_root: Union[str, None]
    test_framework: str
    ignore_paths: list[str]
    formatter: Union[str, list[str]]
    git_remote: str
    enable_telemetry: bool


@dataclass(frozen=True)
class VsCodeSetupInfo:
    module_root: str
    tests_root: str
    test_framework: str
    formatter: Union[str, list[str]]


class DependencyManager(Enum):
    PIP = auto()
    POETRY = auto()
    UV = auto()
    UNKNOWN = auto()


def init_codeflash() -> None:
    try:
        from codeflash.cli_cmds.screens import CodeflashInit
        app = CodeflashInit()
        app.run()
        if app.config_saved:
            sys.exit(0)
    except KeyboardInterrupt:
        apologize_and_exit()


def ask_run_end_to_end_test(args: Namespace) -> None:
    from rich.prompt import Confirm

    run_tests = Confirm.ask(
        "⚡️ Do you want to run a sample optimization to make sure everything's set up correctly? (takes about 3 minutes)",
        choices=["y", "n"],
        default="y",
        show_choices=True,
        show_default=False,
        console=console,
    )

    console.rule()

    if run_tests:
        file_path = create_find_common_tags_file(args, "find_common_tags.py")
        run_end_to_end_test(args, file_path)


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




# common sections between normal mode and lsp mode
class CommonSections(Enum):
    module_root = "module_root"
    tests_root = "tests_root"
    test_framework = "test_framework"
    formatter_cmds = "formatter_cmds"

    def get_toml_key(self) -> str:
        return self.value.replace("_", "-")


@lru_cache(maxsize=1)
def get_valid_subdirs(current_dir: Optional[Path] = None) -> list[str]:
    ignore_subdirs = [
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
    ]
    path_str = str(current_dir) if current_dir else "."
    return [
        d
        for d in next(os.walk(path_str))[1]
        if not d.startswith(".") and not d.startswith("__") and d not in ignore_subdirs
    ]


def get_suggestions(section: str) -> tuple[list[str], Optional[str]]:
    valid_subdirs = get_valid_subdirs()
    if section == CommonSections.module_root:
        return [d for d in valid_subdirs if d != "tests"], None
    if section == CommonSections.tests_root:
        default = "tests" if "tests" in valid_subdirs else None
        return valid_subdirs, default
    if section == CommonSections.test_framework:
        auto_detected = detect_test_framework_from_config_files(Path.cwd())
        return ["pytest", "unittest"], auto_detected
    if section == CommonSections.formatter_cmds:
        return ["disabled", "ruff", "black"], "disabled"
    msg = f"Unknown section: {section}"
    raise ValueError(msg)




def detect_test_framework_from_config_files(curdir: Path) -> Optional[str]:
    test_framework = None
    pytest_files = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
    pytest_config_patterns = {
        "pytest.ini": "[pytest]",
        "pyproject.toml": "[tool.pytest.ini_options]",
        "tox.ini": "[pytest]",
        "setup.cfg": "[tool:pytest]",
    }
    for pytest_file in pytest_files:
        file_path = curdir / pytest_file
        if file_path.exists():
            with file_path.open(encoding="utf8") as file:
                contents = file.read()
                if pytest_config_patterns[pytest_file] in contents:
                    test_framework = "pytest"
                    break
        test_framework = "pytest"
    return test_framework


def detect_test_framework_from_test_files(tests_root: Path) -> Optional[str]:
    test_framework = None
    # Check if any python files contain a class that inherits from unittest.TestCase
    for filename in tests_root.iterdir():
        if filename.suffix == ".py":
            with filename.open(encoding="utf8") as file:
                contents = file.read()
                try:
                    node = ast.parse(contents)
                except SyntaxError:
                    continue
                if any(
                    isinstance(item, ast.ClassDef)
                    and any(
                        (isinstance(base, ast.Attribute) and base.attr == "TestCase")
                        or (isinstance(base, ast.Name) and base.id == "TestCase")
                        for base in item.bases
                    )
                    for item in node.body
                ):
                    test_framework = "unittest"
                    break
    return test_framework




def create_empty_pyproject_toml(pyproject_toml_path: Path) -> None:
    """Create an empty pyproject.toml with a minimal [tool.codeflash] section.

    Used by LSP mode when config file doesn't exist.
    """
    new_pyproject_toml = tomlkit.document()
    new_pyproject_toml["tool"] = {"codeflash": {}}
    pyproject_toml_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")


def install_github_actions(override_formatter_check: bool = False) -> None:  # noqa: FBT001, FBT002
    try:
        config, _config_file_path = parse_config_file(override_formatter_check=override_formatter_check)

        ph("cli-github-actions-install-started")
        try:
            repo = Repo(config["module_root"], search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            click.echo(
                "Skipping GitHub action installation for continuous optimization because you're not in a git repository."
            )
            return

        git_root = Path(repo.git.rev_parse("--show-toplevel"))
        workflows_path = git_root / ".github" / "workflows"
        optimize_yaml_path = workflows_path / "codeflash.yaml"

        actions_panel = Panel(
            Text(
                "🤖 GitHub Actions Setup\n\n"
                "GitHub Actions will automatically optimize your code in every pull request. "
                "This is the recommended way to use Codeflash for continuous optimization.",
                style="blue",
            ),
            title="🤖 Continuous Optimization",
            border_style="bright_blue",
        )
        console.print(actions_panel)
        console.print()

        # Check if the workflow file already exists
        if optimize_yaml_path.exists():
            from rich.prompt import Confirm

            confirm_overwrite = Confirm.ask(
                f"GitHub Actions workflow already exists at {optimize_yaml_path}. Overwrite?",
                default=False,
                console=console,
            )
            if not confirm_overwrite:
                skip_panel = Panel(
                    Text("⏩️ Skipping workflow creation.", style="yellow"), title="⏩️ Skipped", border_style="yellow"
                )
                console.print(skip_panel)
                ph("cli-github-workflow-skipped")
                return
            ph(
                "cli-github-optimization-confirm-workflow-overwrite",
                {"confirm_overwrite": confirm_overwrite},
            )

        from rich.prompt import Confirm

        confirm_creation = Confirm.ask(
            "Set up GitHub Actions for continuous optimization?",
            default=True,
            console=console,
        )
        if not confirm_creation:
            skip_panel = Panel(
                Text("⏩️ Skipping GitHub Actions setup.", style="yellow"), title="⏩️ Skipped", border_style="yellow"
            )
            console.print(skip_panel)
            ph("cli-github-workflow-skipped")
            return
        ph(
            "cli-github-optimization-confirm-workflow-creation",
            {"confirm_creation": confirm_creation},
        )
        workflows_path.mkdir(parents=True, exist_ok=True)
        from importlib.resources import files

        benchmark_mode = False
        benchmarks_root = config.get("benchmarks_root", "").strip()
        if benchmarks_root and benchmarks_root != "":
            benchmark_panel = Panel(
                Text(
                    "📊 Benchmark Mode Available\n\n"
                    "I noticed you've configured a benchmarks_root in your config. "
                    "Benchmark mode will show the performance impact of Codeflash's optimizations on your benchmarks.",
                    style="cyan",
                ),
                title="📊 Benchmark Mode",
                border_style="bright_cyan",
            )
            console.print(benchmark_panel)
            console.print()

            from rich.prompt import Confirm

            benchmark_mode = Confirm.ask(
                "Run GitHub Actions in benchmark mode?",
                default=True,
                console=console,
            )

        optimize_yml_content = (
            files("codeflash").joinpath("cli_cmds", "workflows", "codeflash-optimize.yaml").read_text(encoding="utf-8")
        )
        materialized_optimize_yml_content = customize_codeflash_yaml_content(
            optimize_yml_content, config, git_root, benchmark_mode
        )
        with optimize_yaml_path.open("w", encoding="utf8") as optimize_yml_file:
            optimize_yml_file.write(materialized_optimize_yml_content)
        # Success panel for workflow creation
        workflow_success_panel = Panel(
            Text(
                f"✅ Created GitHub action workflow at {optimize_yaml_path}\n\n"
                "Your repository is now configured for continuous optimization!",
                style="green",
                justify="center",
            ),
            title="🎉 Workflow Created!",
            border_style="bright_green",
        )
        console.print(workflow_success_panel)
        console.print()

        try:
            existing_api_key = get_codeflash_api_key()
        except OSError:
            existing_api_key = None

        # GitHub secrets setup panel
        secrets_message = (
            "🔐 Next Step: Add API Key as GitHub Secret\n\n"
            "You'll need to add your CODEFLASH_API_KEY as a secret to your GitHub repository.\n\n"
            "📋 Steps:\n"
            "1. Press Enter to open your repo's secrets page\n"
            "2. Click 'New repository secret'\n"
            "3. Add your API key with the variable name CODEFLASH_API_KEY"
        )

        if existing_api_key:
            secrets_message += f"\n\n🔑 Your API Key: {existing_api_key}"

        secrets_panel = Panel(
            Text(secrets_message, style="blue"), title="🔐 GitHub Secrets Setup", border_style="bright_blue"
        )
        console.print(secrets_panel)

        console.print(f"\n📍 Press Enter to open: {get_github_secrets_page_url(repo)}")
        console.input()

        click.launch(get_github_secrets_page_url(repo))

        # Post-launch message panel
        launch_panel = Panel(
            Text(
                "🐙 I opened your GitHub secrets page!\n\n"
                "Note: If you see a 404, you probably don't have access to this repo's secrets. "
                "Ask a repo admin to add it for you, or (not recommended) you can temporarily "
                "hard-code your API key into the workflow file.",
                style="cyan",
            ),
            title="🌐 Browser Opened",
            border_style="bright_cyan",
        )
        console.print(launch_panel)
        click.pause()
        click.echo()
        click.echo(
            f"Please edit, commit and push this GitHub actions file to your repo, and you're all set!{LF}"
            f"🚀 Codeflash is now configured to automatically optimize new Github PRs!{LF}"
        )
        ph("cli-github-workflow-created")
    except KeyboardInterrupt:
        apologize_and_exit()


def determine_dependency_manager(pyproject_data: dict[str, Any]) -> DependencyManager:  # noqa: PLR0911
    """Determine which dependency manager is being used based on pyproject.toml contents."""
    if (Path.cwd() / "poetry.lock").exists():
        return DependencyManager.POETRY
    if (Path.cwd() / "uv.lock").exists():
        return DependencyManager.UV
    if "tool" not in pyproject_data:
        return DependencyManager.PIP

    tool_section = pyproject_data["tool"]

    # Check for poetry
    if "poetry" in tool_section:
        return DependencyManager.POETRY

    # Check for uv
    if any(key.startswith("uv") for key in tool_section):
        return DependencyManager.UV

    # Look for pip-specific markers
    if "pip" in tool_section or "setuptools" in tool_section:
        return DependencyManager.PIP

    return DependencyManager.UNKNOWN


def get_codeflash_github_action_command(dep_manager: DependencyManager) -> str:
    """Generate the appropriate codeflash command based on the dependency manager."""
    if dep_manager == DependencyManager.POETRY:
        return """|
          poetry env use python
          poetry run codeflash"""
    if dep_manager == DependencyManager.UV:
        return "uv run codeflash"
    # PIP or UNKNOWN
    return "codeflash"


def get_dependency_installation_commands(dep_manager: DependencyManager) -> tuple[str, str]:
    """Generate commands to install the dependency manager and project dependencies."""
    if dep_manager == DependencyManager.POETRY:
        return """|
          python -m pip install --upgrade pip
          pip install poetry
          poetry install --all-extras"""
    if dep_manager == DependencyManager.UV:
        return "uv sync --all-extras"
    # PIP or UNKNOWN
    return """|
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install codeflash"""


def get_dependency_manager_installation_string(dep_manager: DependencyManager) -> str:
    py_version = sys.version_info
    python_version_string = f"'{py_version.major}.{py_version.minor}'"
    if dep_manager == DependencyManager.UV:
        return """name: 🐍 Setup UV
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true"""
    return f"""name: 🐍 Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: {python_version_string}"""


def get_github_action_working_directory(toml_path: Path, git_root: Path) -> str:
    if toml_path.parent == git_root:
        return ""
    working_dir = str(toml_path.parent.relative_to(git_root))
    return f"""defaults:
      run:
        working-directory: ./{working_dir}"""


def customize_codeflash_yaml_content(
    optimize_yml_content: str,
    config: tuple[dict[str, Any], Path],
    git_root: Path,
    benchmark_mode: bool = False,  # noqa: FBT001, FBT002
) -> str:
    module_path = str(Path(config["module_root"]).relative_to(git_root) / "**")
    optimize_yml_content = optimize_yml_content.replace("{{ codeflash_module_path }}", module_path)

    # Get dependency installation commands
    toml_path = Path.cwd() / "pyproject.toml"
    try:
        with toml_path.open(encoding="utf8") as pyproject_file:
            pyproject_data = tomlkit.parse(pyproject_file.read())
    except FileNotFoundError:
        click.echo(
            f"I couldn't find a pyproject.toml in the current directory.{LF}"
            f"Please create a new empty pyproject.toml file here, OR if you use poetry then run `poetry init`, OR run `codeflash init` again from a directory with an existing pyproject.toml file."
        )
        apologize_and_exit()

    working_dir = get_github_action_working_directory(toml_path, git_root)
    optimize_yml_content = optimize_yml_content.replace("{{ working_directory }}", working_dir)
    dep_manager = determine_dependency_manager(pyproject_data)

    python_depmanager_installation = get_dependency_manager_installation_string(dep_manager)
    optimize_yml_content = optimize_yml_content.replace(
        "{{ setup_python_dependency_manager }}", python_depmanager_installation
    )
    install_deps_cmd = get_dependency_installation_commands(dep_manager)

    optimize_yml_content = optimize_yml_content.replace("{{ install_dependencies_command }}", install_deps_cmd)

    # Add codeflash command
    codeflash_cmd = get_codeflash_github_action_command(dep_manager)

    if benchmark_mode:
        codeflash_cmd += " --benchmark"
    return optimize_yml_content.replace("{{ codeflash_command }}", codeflash_cmd)


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
        codeflash_section["test-framework"] = setup_info.test_framework
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
    click.echo(f"✅ Added Codeflash configuration to {toml_path}")
    click.echo()
    return True






def create_find_common_tags_file(args: Namespace, file_name: str) -> Path:
    find_common_tags_content = """def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0]["tags"]
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article["tags"]]
    return set(common_tags)
"""

    file_path = Path(args.module_root) / file_name
    lsp_enabled = is_LSP_enabled()
    if file_path.exists() and not lsp_enabled:
        from rich.prompt import Confirm

        overwrite = Confirm.ask(
            f"🤔 {file_path} already exists. Do you want to overwrite it?", default=True, show_default=False
        )
        if not overwrite:
            apologize_and_exit()
        console.rule()

    file_path.write_text(find_common_tags_content, encoding="utf8")
    logger.info(f"Created demo optimization file: {file_path}")

    return file_path


def create_bubble_sort_file_and_test(args: Namespace) -> tuple[str, str]:
    bubble_sort_content = """from typing import Union, List
def sorter(arr: Union[List[int],List[float]]) -> Union[List[int],List[float]]:
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr
"""
    if args.test_framework == "unittest":
        bubble_sort_test_content = f"""import unittest
from {os.path.basename(args.module_root)}.bubble_sort import sorter # Keep usage of os.path.basename to avoid pathlib potential incompatibility https://github.com/codeflash-ai/codeflash/pull/1066#discussion_r1801628022

class TestBubbleSort(unittest.TestCase):
    def test_sort(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter(input)
        self.assertEqual(output, [0, 1, 2, 3, 4, 5])

        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter(input)
        self.assertEqual(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        input = list(reversed(range(100)))
        output = sorter(input)
        self.assertEqual(output, list(range(100)))
"""  # noqa: PTH119
    elif args.test_framework == "pytest":
        bubble_sort_test_content = f"""from {Path(args.module_root).name}.bubble_sort import sorter

def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(500)))
    output = sorter(input)
    assert output == list(range(500))
"""

    bubble_sort_path = Path(args.module_root) / "bubble_sort.py"
    if bubble_sort_path.exists():
        from rich.prompt import Confirm

        overwrite = Confirm.ask(
            f"🤔 {bubble_sort_path} already exists. Do you want to overwrite it?", default=True, show_default=False
        )
        if not overwrite:
            apologize_and_exit()
        console.rule()

    bubble_sort_path.write_text(bubble_sort_content, encoding="utf8")

    bubble_sort_test_path = Path(args.tests_root) / "test_bubble_sort.py"
    bubble_sort_test_path.write_text(bubble_sort_test_content, encoding="utf8")

    for path in [bubble_sort_path, bubble_sort_test_path]:
        logger.info(f"✅ Created {path}")
        console.rule()

    return str(bubble_sort_path), str(bubble_sort_test_path)


def run_end_to_end_test(args: Namespace, find_common_tags_path: Path) -> None:
    try:
        check_formatter_installed(args.formatter_cmds)
    except Exception:
        logger.error(
            "Formatter not found. Review the formatter_cmds in your pyproject.toml file and make sure the formatter is installed."
        )
        return

    command = ["codeflash", "--file", "find_common_tags.py", "--function", "find_common_tags"]
    if args.no_pr:
        command.append("--no-pr")
    if args.verbose:
        command.append("--verbose")

    logger.info("Running sample optimization…")
    console.rule()

    try:
        output = []
        with subprocess.Popen(
            command, text=True, cwd=args.module_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            if process.stdout:
                for line in process.stdout:
                    stripped = line.strip()
                    console.out(stripped)
                    output.append(stripped)
            process.wait()
        console.rule()
        if process.returncode == 0:
            logger.info("End-to-end test passed. Codeflash has been correctly set up!")
        else:
            logger.error(
                "End-to-end test failed. Please check the logs above, and take a look at https://docs.codeflash.ai/getting-started/local-installation for help and troubleshooting."
            )
    finally:
        console.rule()
        # Delete the bubble_sort.py file after the test
        logger.info("🧹 Cleaning up…")
        find_common_tags_path.unlink(missing_ok=True)
        logger.info(f"🗑️  Deleted {find_common_tags_path}")


