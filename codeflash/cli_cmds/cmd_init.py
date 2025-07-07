from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import click
import git
import inquirer
import inquirer.themes
import tomlkit
from git import InvalidGitRepositoryError, Repo
from pydantic.dataclasses import dataclass
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.api.cfapi import is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import check_formatter_installed, get_codeflash_api_key
from codeflash.code_utils.git_utils import get_git_remotes, get_repo_owner_and_name
from codeflash.code_utils.github_utils import get_github_secrets_page_url
from codeflash.code_utils.shell_utils import get_shell_rc_path, save_api_key_to_rc
from codeflash.either import is_successful
from codeflash.telemetry.posthog_cf import ph
from codeflash.version import __version__ as version

if TYPE_CHECKING:
    from argparse import Namespace

CODEFLASH_LOGO: str = (
    f"{LF}"  # noqa: ISC003
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
class SetupInfo:
    module_root: str
    tests_root: str
    benchmarks_root: Union[str, None]
    test_framework: str
    ignore_paths: list[str]
    formatter: str
    git_remote: str


class DependencyManager(Enum):
    PIP = auto()
    POETRY = auto()
    UV = auto()
    UNKNOWN = auto()


def init_codeflash() -> None:
    try:
        welcome_panel = Panel(
            Text(
                "⚡️ Welcome to Codeflash!\n\nThis setup will take just a few minutes.",
                style="bold cyan",
                justify="center",
            ),
            title="🚀 Codeflash Setup",
            border_style="bright_cyan",
            padding=(1, 2),
        )
        console.print(welcome_panel)
        console.print()

        did_add_new_key = prompt_api_key()

        if should_modify_pyproject_toml():
            setup_info: SetupInfo = collect_setup_info()

            configure_pyproject_toml(setup_info)

        install_github_app()

        install_github_actions(override_formatter_check=True)

        module_string = ""
        if "setup_info" in locals():
            module_string = f" you selected ({setup_info.module_root})"

        usage_table = Table(show_header=False, show_lines=False, border_style="dim")
        usage_table.add_column("Command", style="cyan")
        usage_table.add_column("Description", style="white")

        usage_table.add_row(
            "codeflash --file <path-to-file> --function <function-name>", "Optimize a specific function within a file"
        )
        usage_table.add_row("codeflash optimize <myscript.py>", "Trace and find the best optimizations for a script")
        usage_table.add_row("codeflash --all", "Optimize all functions in all files")
        usage_table.add_row("codeflash --help", "See all available options")

        completion_message = "⚡️ Codeflash is now set up!\n\nYou can now run any of these commands:"

        if did_add_new_key:
            completion_message += (
                "\n\n🐚 Don't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!"
            )
            reload_cmd = f"call {get_shell_rc_path()}" if os.name == "nt" else f"source {get_shell_rc_path()}"
            completion_message += f"\nOr run: {reload_cmd}"

        completion_panel = Panel(
            Group(Text(completion_message, style="bold green"), Text(""), usage_table),
            title="🎉 Setup Complete!",
            border_style="bright_green",
            padding=(1, 2),
        )
        console.print(completion_panel)

        ph("cli-installation-successful", {"did_add_new_key": did_add_new_key})
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
        bubble_sort_path, bubble_sort_test_path = create_bubble_sort_file_and_test(args)
        run_end_to_end_test(args, bubble_sort_path, bubble_sort_test_path)


def should_modify_pyproject_toml() -> bool:
    """Check if the current directory contains a valid pyproject.toml file with codeflash config.

    If it does, ask the user if they want to re-configure it.
    """
    from rich.prompt import Confirm

    pyproject_toml_path = Path.cwd() / "pyproject.toml"
    if not pyproject_toml_path.exists():
        return True
    try:
        config, config_file_path = parse_config_file(pyproject_toml_path)
    except Exception:
        return True

    if "module_root" not in config or config["module_root"] is None or not Path(config["module_root"]).is_dir():
        return True
    if "tests_root" not in config or config["tests_root"] is None or not Path(config["tests_root"]).is_dir():
        return True

    return Confirm.ask(
        "✅ A valid Codeflash config already exists in this project. Do you want to re-configure it?",
        default=False,
        show_default=True,
    )


# Custom theme for better UX
class CodeflashTheme(inquirer.themes.Default):
    def __init__(self) -> None:
        super().__init__()
        self.Question.mark_color = inquirer.themes.term.yellow
        self.Question.brackets_color = inquirer.themes.term.bright_blue
        self.Question.default_color = inquirer.themes.term.bright_cyan
        self.List.selection_color = inquirer.themes.term.bright_blue
        self.List.selection_cursor = "⚡"
        self.Checkbox.selection_color = inquirer.themes.term.bright_blue
        self.Checkbox.selection_cursor = "⚡"
        self.Checkbox.selected_icon = "✅"
        self.Checkbox.unselected_icon = "⬜"


def collect_setup_info() -> SetupInfo:
    curdir = Path.cwd()
    # Check if the cwd is writable
    if not os.access(curdir, os.W_OK):
        click.echo(f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}")
        click.echo("It's likely you don't have write permissions for this folder.")
        sys.exit(1)

    # Check for the existence of pyproject.toml or setup.py
    project_name = check_for_toml_or_setup_file()

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
    valid_subdirs = [
        d for d in next(os.walk("."))[1] if not d.startswith(".") and not d.startswith("__") and d not in ignore_subdirs
    ]

    valid_module_subdirs = [d for d in valid_subdirs if d != "tests"]

    curdir_option = f"current directory ({curdir})"
    custom_dir_option = "enter a custom directory…"
    module_subdir_options = [*valid_module_subdirs, curdir_option, custom_dir_option]

    info_panel = Panel(
        Text(
            "📁 Let's identify your Python module directory.\n\n"
            "This is usually the top-level directory containing all your Python source code.\n",
            style="cyan",
        ),
        title="🔍 Module Discovery",
        border_style="bright_blue",
    )
    console.print(info_panel)
    console.print()
    questions = [
        inquirer.List(
            "module_root",
            message="Which Python module do you want me to optimize?",
            choices=module_subdir_options,
            default=(project_name if project_name in module_subdir_options else module_subdir_options[0]),
            carousel=True,
        )
    ]

    answers = inquirer.prompt(questions, theme=CodeflashTheme())
    if not answers:
        apologize_and_exit()
    module_root_answer = answers["module_root"]
    if module_root_answer == curdir_option:
        module_root = "."
    elif module_root_answer == custom_dir_option:
        custom_panel = Panel(
            Text(
                "📂 Enter a custom module directory path.\n\nPlease provide the path to your Python module directory.",
                style="yellow",
            ),
            title="📂 Custom Directory",
            border_style="bright_yellow",
        )
        console.print(custom_panel)
        console.print()

        custom_questions = [
            inquirer.Path(
                "custom_path",
                message="Enter the path to your module directory",
                path_type=inquirer.Path.DIRECTORY,
                exists=True,
            )
        ]

        custom_answers = inquirer.prompt(custom_questions, theme=CodeflashTheme())
        if custom_answers:
            module_root = Path(custom_answers["custom_path"])
        else:
            apologize_and_exit()
    else:
        module_root = module_root_answer
    ph("cli-project-root-provided")

    # Discover test directory
    default_tests_subdir = "tests"
    create_for_me_option = f"🆕 Create a new tests{os.pathsep} directory for me!"
    test_subdir_options = [sub_dir for sub_dir in valid_subdirs if sub_dir != module_root]
    if "tests" not in valid_subdirs:
        test_subdir_options.append(create_for_me_option)
    custom_dir_option = "📁 Enter a custom directory…"
    test_subdir_options.append(custom_dir_option)

    tests_panel = Panel(
        Text(
            "🧪 Now let's locate your test directory.\n\n"
            "This is where all your test files are stored. If you don't have tests yet, "
            "I can create a directory for you!",
            style="green",
        ),
        title="🧪 Test Discovery",
        border_style="bright_green",
    )
    console.print(tests_panel)
    console.print()

    tests_questions = [
        inquirer.List(
            "tests_root",
            message="Where are your tests located?",
            choices=test_subdir_options,
            default=(default_tests_subdir if default_tests_subdir in test_subdir_options else test_subdir_options[0]),
            carousel=True,
        )
    ]

    tests_answers = inquirer.prompt(tests_questions, theme=CodeflashTheme())
    if not tests_answers:
        apologize_and_exit()
    tests_root_answer = tests_answers["tests_root"]

    if tests_root_answer == create_for_me_option:
        tests_root = Path(curdir) / default_tests_subdir
        tests_root.mkdir()
        click.echo(f"✅ Created directory {tests_root}{os.path.sep}{LF}")
    elif tests_root_answer == custom_dir_option:
        custom_tests_panel = Panel(
            Text(
                "🧪 Enter a custom test directory path.\n\nPlease provide the path to your test directory, relative to the current directory.",
                style="yellow",
            ),
            title="🧪 Custom Test Directory",
            border_style="bright_yellow",
        )
        console.print(custom_tests_panel)
        console.print()

        custom_tests_questions = [
            inquirer.Path(
                "custom_tests_path", message="Enter the path to your tests directory", path_type=inquirer.Path.DIRECTORY
            )
        ]

        custom_tests_answers = inquirer.prompt(custom_tests_questions, theme=CodeflashTheme())
        if custom_tests_answers:
            tests_root = Path(curdir) / Path(custom_tests_answers["custom_tests_path"])
        else:
            apologize_and_exit()
    else:
        tests_root = Path(curdir) / Path(cast("str", tests_root_answer))

    tests_root = tests_root.relative_to(curdir)

    resolved_module_root = (Path(curdir) / Path(module_root)).resolve()
    resolved_tests_root = (Path(curdir) / Path(tests_root)).resolve()
    if resolved_module_root == resolved_tests_root:
        logger.warning(
            "It looks like your tests root is the same as your module root. This is not recommended and can lead to unexpected behavior."
        )

    ph("cli-tests-root-provided")

    autodetected_test_framework = detect_test_framework(curdir, tests_root)

    framework_message = "⚗️ Let's configure your test framework.\n\n"
    if autodetected_test_framework:
        framework_message += f"I detected that you're using {autodetected_test_framework}. "
    framework_message += "Please confirm or select a different one."

    framework_panel = Panel(Text(framework_message, style="blue"), title="⚗️ Test Framework", border_style="bright_blue")
    console.print(framework_panel)
    console.print()

    framework_questions = [
        inquirer.List(
            "test_framework",
            message="Which test framework do you use?",
            choices=[("🧪 pytest", "pytest"), ("🐍 unittest", "unittest")],
            default=autodetected_test_framework or "pytest",
            carousel=True,
        )
    ]

    framework_answers = inquirer.prompt(framework_questions, theme=CodeflashTheme())
    if not framework_answers:
        apologize_and_exit()
    test_framework = framework_answers["test_framework"]

    ph("cli-test-framework-provided", {"test_framework": test_framework})

    benchmarks_root = None

    # TODO: Implement other benchmark framework options
    # if benchmarks_root:
    #     benchmarks_root = benchmarks_root.relative_to(curdir)
    #
    #     # Ask about benchmark framework
    #     benchmark_framework_options = ["pytest-benchmark", "asv (Airspeed Velocity)", "custom/other"]
    #     benchmark_framework = inquirer_wrapper(
    #         inquirer.list_input,
    #         message="Which benchmark framework do you use?",
    #         choices=benchmark_framework_options,
    #         default=benchmark_framework_options[0],
    #         carousel=True,
    #     )

    formatter_panel = Panel(
        Text(
            "🎨 Let's configure your code formatter.\n\n"
            "Code formatters help maintain consistent code style. "
            "Codeflash will use this to format optimized code.",
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
                ("⚫ black", "black"),
                ("⚡ ruff", "ruff"),
                ("🔧 other", "other"),
                ("❌ don't use a formatter", "don't use a formatter"),
            ],
            default="black",
            carousel=True,
        )
    ]

    formatter_answers = inquirer.prompt(formatter_questions, theme=CodeflashTheme())
    if not formatter_answers:
        apologize_and_exit()
    formatter = formatter_answers["formatter"]

    git_remote = ""
    try:
        repo = Repo(str(module_root), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if git_remotes:  # Only proceed if there are remotes
            if len(git_remotes) > 1:
                git_panel = Panel(
                    Text(
                        "🔗 Configure Git Remote for Pull Requests.\n\n"
                        "Codeflash will use this remote to create pull requests with optimized code.",
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
                        message="Which git remote should Codeflash use for Pull Requests?",
                        choices=git_remotes,
                        default="origin",
                        carousel=True,
                    )
                ]

                git_answers = inquirer.prompt(git_questions, theme=CodeflashTheme())
                git_remote = git_answers["git_remote"] if git_answers else git_remotes[0]
            else:
                git_remote = git_remotes[0]
        else:
            click.echo(
                "No git remotes found. You can still use Codeflash locally, but you'll need to set up a remote "
                "repository to use GitHub features."
            )
    except InvalidGitRepositoryError:
        git_remote = ""

    ignore_paths: list[str] = []
    return SetupInfo(
        module_root=str(module_root),
        tests_root=str(tests_root),
        benchmarks_root=str(benchmarks_root) if benchmarks_root else None,
        test_framework=cast("str", test_framework),
        ignore_paths=ignore_paths,
        formatter=cast("str", formatter),
        git_remote=str(git_remote),
    )


def detect_test_framework(curdir: Path, tests_root: Path) -> str | None:
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
    else:
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


def check_for_toml_or_setup_file() -> str | None:
    click.echo()
    click.echo("Checking for pyproject.toml or setup.py…\r", nl=False)
    curdir = Path.cwd()
    pyproject_toml_path = curdir / "pyproject.toml"
    setup_py_path = curdir / "setup.py"
    project_name = None
    if pyproject_toml_path.exists():
        try:
            pyproject_toml_content = pyproject_toml_path.read_text(encoding="utf8")
            project_name = tomlkit.parse(pyproject_toml_content)["tool"]["poetry"]["name"]
            click.echo(f"✅ I found a pyproject.toml for your project {project_name}.")
            ph("cli-pyproject-toml-found-name")
        except Exception:
            click.echo("✅ I found a pyproject.toml for your project.")
            ph("cli-pyproject-toml-found")
    else:
        if setup_py_path.exists():
            setup_py_content = setup_py_path.read_text(encoding="utf8")
            project_name_match = re.search(r"setup\s*\([^)]*?name\s*=\s*['\"](.*?)['\"]", setup_py_content, re.DOTALL)
            if project_name_match:
                project_name = project_name_match.group(1)
                click.echo(f"✅ Found setup.py for your project {project_name}")
                ph("cli-setup-py-found-name")
            else:
                click.echo("✅ Found setup.py.")
                ph("cli-setup-py-found")
        toml_info_panel = Panel(
            Text(
                f"💡 No pyproject.toml found in {curdir}.\n\n"
                "This file is essential for Codeflash to store its configuration.\n"
                "Please ensure you are running `codeflash init` from your project's root directory.",
                style="yellow",
            ),
            title="📋 pyproject.toml Required",
            border_style="bright_yellow",
        )
        console.print(toml_info_panel)
        console.print()
        ph("cli-no-pyproject-toml-or-setup-py")

        # Create a pyproject.toml file because it doesn't exist
        toml_questions = [
            inquirer.Confirm("create_toml", message="Create pyproject.toml in the current directory?", default=True)
        ]

        toml_answers = inquirer.prompt(toml_questions, theme=CodeflashTheme())
        if not toml_answers:
            apologize_and_exit()
        create_toml = toml_answers["create_toml"]
        if create_toml:
            ph("cli-create-pyproject-toml")
            # Define a minimal pyproject.toml content
            new_pyproject_toml = tomlkit.document()
            new_pyproject_toml["tool"] = {"codeflash": {}}
            try:
                pyproject_toml_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")

                # Check if the pyproject.toml file was created
                if pyproject_toml_path.exists():
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
                click.echo(
                    "❌ Failed to create pyproject.toml. Please check your disk permissions and available space."
                )
                apologize_and_exit()
        else:
            click.echo("⏩️ Skipping pyproject.toml creation.")
            apologize_and_exit()
    click.echo()
    return cast("str", project_name)


def install_github_actions(override_formatter_check: bool = False) -> None:  # noqa: FBT001, FBT002
    try:
        config, config_file_path = parse_config_file(override_formatter_check=override_formatter_check)

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
            overwrite_questions = [
                inquirer.Confirm(
                    "confirm_overwrite",
                    message=f"GitHub Actions workflow already exists at {optimize_yaml_path}. Overwrite?",
                    default=False,
                )
            ]

            overwrite_answers = inquirer.prompt(overwrite_questions, theme=CodeflashTheme())
            if not overwrite_answers or not overwrite_answers["confirm_overwrite"]:
                skip_panel = Panel(
                    Text("⏩️ Skipping workflow creation.", style="yellow"), title="⏩️ Skipped", border_style="yellow"
                )
                console.print(skip_panel)
                ph("cli-github-workflow-skipped")
                return
            ph(
                "cli-github-optimization-confirm-workflow-overwrite",
                {"confirm_overwrite": overwrite_answers["confirm_overwrite"]},
            )

        creation_questions = [
            inquirer.Confirm(
                "confirm_creation", message="Set up GitHub Actions for continuous optimization?", default=True
            )
        ]

        creation_answers = inquirer.prompt(creation_questions, theme=CodeflashTheme())
        if not creation_answers or not creation_answers["confirm_creation"]:
            skip_panel = Panel(
                Text("⏩️ Skipping GitHub Actions setup.", style="yellow"), title="⏩️ Skipped", border_style="yellow"
            )
            console.print(skip_panel)
            ph("cli-github-workflow-skipped")
            return
        ph(
            "cli-github-optimization-confirm-workflow-creation",
            {"confirm_creation": creation_answers["confirm_creation"]},
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

            benchmark_questions = [
                inquirer.Confirm("benchmark_mode", message="Run GitHub Actions in benchmark mode?", default=True)
            ]

            benchmark_answers = inquirer.prompt(benchmark_questions, theme=CodeflashTheme())
            benchmark_mode = benchmark_answers["benchmark_mode"] if benchmark_answers else False

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


# Create or update the pyproject.toml file with the Codeflash dependency & configuration
def configure_pyproject_toml(setup_info: SetupInfo) -> None:
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

    enable_telemetry = ask_for_telemetry()

    codeflash_section = tomlkit.table()
    codeflash_section.add(tomlkit.comment("All paths are relative to this pyproject.toml's directory."))
    codeflash_section["module-root"] = setup_info.module_root
    codeflash_section["tests-root"] = setup_info.tests_root
    codeflash_section["test-framework"] = setup_info.test_framework
    codeflash_section["ignore-paths"] = setup_info.ignore_paths
    if not enable_telemetry:
        codeflash_section["disable-telemetry"] = not enable_telemetry
    if setup_info.git_remote not in ["", "origin"]:
        codeflash_section["git-remote"] = setup_info.git_remote
    formatter = setup_info.formatter
    formatter_cmds = []
    if formatter == "black":
        formatter_cmds.append("black $file")
    elif formatter == "ruff":
        formatter_cmds.extend(["ruff check --exit-zero --fix $file", "ruff format $file"])
    elif formatter == "other":
        formatter_cmds.append("your-formatter $file")
        click.echo(
            "🔧 In pyproject.toml, please replace 'your-formatter' with the command you use to format your code."
        )
    elif formatter == "don't use a formatter":
        formatter_cmds.append("disabled")
    check_formatter_installed(formatter_cmds, exit_on_failure=False)
    codeflash_section["formatter-cmds"] = formatter_cmds
    # Add the 'codeflash' section, ensuring 'tool' section exists
    tool_section = pyproject_data.get("tool", tomlkit.table())
    tool_section["codeflash"] = codeflash_section
    pyproject_data["tool"] = tool_section

    with toml_path.open("w", encoding="utf8") as pyproject_file:
        pyproject_file.write(tomlkit.dumps(pyproject_data))
    click.echo(f"✅ Added Codeflash configuration to {toml_path}")
    click.echo()


def install_github_app() -> None:
    try:
        git_repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        click.echo("Skipping GitHub app installation because you're not in a git repository.")
        return
    owner, repo = get_repo_owner_and_name(git_repo)

    if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
        click.echo("🐙 Looks like you've already installed the Codeflash GitHub app on this repository! Continuing…")

    else:
        click.prompt(
            f"Finally, you'll need to install the Codeflash GitHub app by choosing the repository you want to install Codeflash on.{LF}"
            f"I will attempt to open the github app page - https://github.com/apps/codeflash-ai/installations/select_target {LF}"
            f"Press Enter to open the page to let you install the app…{LF}",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )
        click.launch("https://github.com/apps/codeflash-ai/installations/select_target")
        click.prompt(
            f"Press Enter once you've finished installing the github app from https://github.com/apps/codeflash-ai/installations/select_target…{LF}",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )

        count = 2
        while not is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
            if count == 0:
                click.echo(
                    f"❌ It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo}.{LF}"
                    f"You won't be able to create PRs with Codeflash until you install the app.{LF}"
                    f"In the meantime you can make local only optimizations by using the '--no-pr' flag with codeflash.{LF}"
                )
                break
            click.prompt(
                f"❌ It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo}.{LF}"
                f"Please install it from https://github.com/apps/codeflash-ai/installations/select_target {LF}"
                f"Press Enter to continue once you've finished installing the github app…{LF}",
                default="",
                type=click.STRING,
                prompt_suffix="",
                show_default=False,
            )
            count -= 1


class CFAPIKeyType(click.ParamType):
    name = "cfapi-key"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> str | None:
        value = value.strip()
        if not value.startswith("cf-") and value != "":
            self.fail(
                f"That key [{value}] seems to be invalid. It should start with a 'cf-' prefix. Please try again.",
                param,
                ctx,
            )
        return value


# Returns True if the user entered a new API key, False if they used an existing one
def prompt_api_key() -> bool:
    try:
        existing_api_key = get_codeflash_api_key()
    except OSError:
        existing_api_key = None
    if existing_api_key:
        display_key = f"{existing_api_key[:3]}****{existing_api_key[-4:]}"
        api_key_panel = Panel(
            Text(
                f"🔑 I found a CODEFLASH_API_KEY in your environment [{display_key}]!\n\n"
                "✅ You're all set with API authentication!",
                style="green",
                justify="center",
            ),
            title="🔑 API Key Found",
            border_style="bright_green",
        )
        console.print(api_key_panel)
        console.print()
        return False

    enter_api_key_and_save_to_rc()
    ph("cli-new-api-key-entered")
    return True


def enter_api_key_and_save_to_rc() -> None:
    browser_launched = False
    api_key = ""
    while api_key == "":
        api_key = click.prompt(
            f"Enter your Codeflash API key{' [or press Enter to open your API key page]' if not browser_launched else ''}",
            hide_input=False,
            default="",
            type=CFAPIKeyType(),
            show_default=False,
        ).strip()
        if api_key:
            break
        if not browser_launched:
            click.echo(
                f"Opening your Codeflash API key page. Grab a key from there!{LF}"
                "You can also open this link manually: https://app.codeflash.ai/app/apikeys"
            )
            click.launch("https://app.codeflash.ai/app/apikeys")
            browser_launched = True  # This does not work on remote consoles
    shell_rc_path = get_shell_rc_path()
    if not shell_rc_path.exists() and os.name == "nt":
        # On Windows, create a batch file in the user's home directory (not auto-run, just used to store api key)
        shell_rc_path.touch()
        click.echo(f"✅ Created {shell_rc_path}")
    result = save_api_key_to_rc(api_key)
    if is_successful(result):
        click.echo(result.unwrap())
    else:
        click.echo(result.failure())
        click.pause()

    os.environ["CODEFLASH_API_KEY"] = api_key


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


def run_end_to_end_test(args: Namespace, bubble_sort_path: str, bubble_sort_test_path: str) -> None:
    try:
        check_formatter_installed(args.formatter_cmds)
    except Exception:
        logger.error(
            "Formatter not found. Review the formatter_cmds in your pyproject.toml file and make sure the formatter is installed."
        )
        return

    command = ["codeflash", "--file", "bubble_sort.py", "--function", "sorter"]
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
        for path in [bubble_sort_path, bubble_sort_test_path]:
            console.rule()
            Path(path).unlink(missing_ok=True)
            logger.info(f"🗑️  Deleted {path}")


def ask_for_telemetry() -> bool:
    """Prompt the user to enable or disable telemetry."""
    from rich.prompt import Confirm

    return Confirm.ask(
        "⚡️ Would you like to enable telemetry to help us improve the Codeflash experience?",
        default=True,
        show_default=True,
    )
