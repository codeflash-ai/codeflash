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

from codeflash.api.cfapi import is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit, inquirer_wrapper, inquirer_wrapper_path
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
        click.echo(f"⚡️ Welcome to Codeflash! Let's get you set up.{LF}")

        did_add_new_key = prompt_api_key()

        if should_modify_pyproject_toml():
            setup_info: SetupInfo = collect_setup_info()

            configure_pyproject_toml(setup_info)

        install_github_app()

        install_github_actions(override_formatter_check=True)

        module_string = ""
        if "setup_info" in locals():
            module_string = f" you selected ({setup_info.module_root})"

        click.echo(
            f"{LF}"
            f"⚡️ Codeflash is now set up! You can now run:{LF}"
            f"    codeflash --file <path-to-file> --function <function-name> to optimize a function within a file{LF}"
            f"    codeflash --file <path-to-file> to optimize all functions in a file{LF}"
            f"    codeflash --all to optimize all functions in all files in the module{module_string}{LF}"
            f"-or-{LF}"
            f"    codeflash --help to see all options{LF}"
        )
        if did_add_new_key:
            click.echo("🐚 Don't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!")
            click.echo("Or run the following command to reload:")
            if os.name == "nt":
                click.echo(f"  call {get_shell_rc_path()}")
            else:
                click.echo(f"  source {get_shell_rc_path()}")

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

    module_root_answer = inquirer_wrapper(
        inquirer.list_input,
        message="Which Python module do you want me to optimize going forward? (Usually the top-most directory with "
        "all of your Python source code). Use arrow keys to select",
        choices=module_subdir_options,
        default=(project_name if project_name in module_subdir_options else module_subdir_options[0]),
    )
    if module_root_answer == curdir_option:
        module_root = "."
    elif module_root_answer == custom_dir_option:
        custom_module_root_answer = inquirer_wrapper_path(
            "path",
            message=f"Enter the path to your module directory inside {Path(curdir).resolve()}{os.path.sep} ",
            path_type=inquirer.Path.DIRECTORY,
        )
        if custom_module_root_answer:
            module_root = Path(custom_module_root_answer["path"])
        else:
            apologize_and_exit()
    else:
        module_root = module_root_answer
    ph("cli-project-root-provided")

    # Discover test directory
    default_tests_subdir = "tests"
    create_for_me_option = f"okay, create a tests{os.pathsep} directory for me!"
    test_subdir_options = [sub_dir for sub_dir in valid_subdirs if sub_dir != module_root]
    if "tests" not in valid_subdirs:
        test_subdir_options.append(create_for_me_option)
    custom_dir_option = "enter a custom directory…"
    test_subdir_options.append(custom_dir_option)
    tests_root_answer = inquirer_wrapper(
        inquirer.list_input,
        message="Where are your tests located? "
        f"(If you don't have any tests yet, I can create an empty tests{os.pathsep} directory for you)",
        choices=test_subdir_options,
        default=(default_tests_subdir if default_tests_subdir in test_subdir_options else test_subdir_options[0]),
    )

    if tests_root_answer == create_for_me_option:
        tests_root = Path(curdir) / default_tests_subdir
        tests_root.mkdir()
        click.echo(f"✅ Created directory {tests_root}{os.path.sep}{LF}")
    elif tests_root_answer == custom_dir_option:
        custom_tests_root_answer = inquirer_wrapper_path(
            "path",
            message=f"Enter the path to your tests directory inside {Path(curdir).resolve()}{os.path.sep} ",
            path_type=inquirer.Path.DIRECTORY,
        )
        if custom_tests_root_answer:
            tests_root = Path(curdir) / Path(custom_tests_root_answer["path"])
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

    # Autodiscover test framework
    autodetected_test_framework = detect_test_framework(curdir, tests_root)
    autodetected_suffix = (
        f" (seems to me you're using {autodetected_test_framework})" if autodetected_test_framework else ""
    )
    test_framework = inquirer_wrapper(
        inquirer.list_input,
        message="Which test framework do you use?" + autodetected_suffix,
        choices=["pytest", "unittest"],
        default=autodetected_test_framework or "pytest",
        carousel=True,
    )

    ph("cli-test-framework-provided", {"test_framework": test_framework})

    # Get benchmarks root directory
    default_benchmarks_subdir = "benchmarks"
    create_benchmarks_option = f"okay, create a {default_benchmarks_subdir}{os.path.sep} directory for me!"
    no_benchmarks_option = "I don't need benchmarks"

    # Check if benchmarks directory exists inside tests directory
    tests_subdirs = []
    if tests_root.exists():
        tests_subdirs = [d.name for d in tests_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    benchmarks_options = []
    benchmarks_options.append(no_benchmarks_option)
    if default_benchmarks_subdir in tests_subdirs:
        benchmarks_options.append(default_benchmarks_subdir)
    benchmarks_options.extend([d for d in tests_subdirs if d != default_benchmarks_subdir and d not in ignore_subdirs])
    benchmarks_options.append(create_benchmarks_option)
    benchmarks_options.append(custom_dir_option)

    benchmarks_answer = inquirer_wrapper(
        inquirer.list_input,
        message="Where are your performance benchmarks located? (benchmarks must be a sub directory of your tests root directory)",
        choices=benchmarks_options,
        default=(
            default_benchmarks_subdir if default_benchmarks_subdir in benchmarks_options else benchmarks_options[0]
        ),
    )

    if benchmarks_answer == create_benchmarks_option:
        benchmarks_root = tests_root / default_benchmarks_subdir
        benchmarks_root.mkdir(exist_ok=True)
        click.echo(f"✅ Created directory {benchmarks_root}{os.path.sep}{LF}")
    elif benchmarks_answer == custom_dir_option:
        custom_benchmarks_answer = inquirer_wrapper_path(
            "path",
            message=f"Enter the path to your benchmarks directory inside {tests_root}{os.path.sep} ",
            path_type=inquirer.Path.DIRECTORY,
        )
        if custom_benchmarks_answer:
            benchmarks_root = tests_root / Path(custom_benchmarks_answer["path"])
        else:
            apologize_and_exit()
    elif benchmarks_answer == no_benchmarks_option:
        benchmarks_root = None
    else:
        benchmarks_root = tests_root / Path(cast("str", benchmarks_answer))

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

    formatter = inquirer_wrapper(
        inquirer.list_input,
        message="Which code formatter do you use?",
        choices=["black", "ruff", "other", "don't use a formatter"],
        default="black",
        carousel=True,
    )

    git_remote = ""
    try:
        repo = Repo(str(module_root), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if git_remotes:  # Only proceed if there are remotes
            if len(git_remotes) > 1:
                git_remote = inquirer_wrapper(
                    inquirer.list_input,
                    message="What git remote do you want Codeflash to use for new Pull Requests? ",
                    choices=git_remotes,
                    default="origin",
                    carousel=True,
                )
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
        click.echo(
            f"💡 I couldn't find a pyproject.toml in the current directory ({curdir}).{LF}"
            f"(make sure you're running `codeflash init` from your project's root directory!){LF}"
            f"I need this file to store my configuration settings."
        )
        ph("cli-no-pyproject-toml-or-setup-py")

        # Create a pyproject.toml file because it doesn't exist
        create_toml = inquirer_wrapper(
            inquirer.confirm,
            message="Do you want me to create a pyproject.toml file in the current directory?",
            default=True,
            show_default=False,
        )
        if create_toml:
            ph("cli-create-pyproject-toml")
            # Define a minimal pyproject.toml content
            new_pyproject_toml = tomlkit.document()
            new_pyproject_toml["tool"] = {"codeflash": {}}
            try:
                pyproject_toml_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")

                # Check if the pyproject.toml file was created
                if pyproject_toml_path.exists():
                    click.echo(f"✅ Created a pyproject.toml file at {pyproject_toml_path}")
                    click.pause()
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

        # Check if the workflow file already exists
        if optimize_yaml_path.exists():
            confirm_overwrite = inquirer_wrapper(
                inquirer.confirm,
                message=f"⚡️ GitHub Actions workflow already exists at {optimize_yaml_path}. Overwrite?",
                default=False,  # Don't overwrite by default
            )
            ph("cli-github-optimization-confirm-workflow-overwrite", {"confirm_overwrite": confirm_overwrite})
            if not confirm_overwrite:
                click.echo("⏩️ Skipping workflow creation.")
                ph("cli-github-workflow-skipped")
                return

        confirm_creation_yes = inquirer_wrapper(
            inquirer.confirm,
            message="⚡️Shall I set up a GitHub action that will continuously optimize all new code in GitHub PRs"
            " for you? This is the main way of using Codeflash so we highly recommend it",
            default=True,
        )
        ph("cli-github-optimization-confirm-workflow-creation", {"confirm_creation": confirm_creation_yes})
        if not confirm_creation_yes:
            click.echo("⏩️ Exiting workflow creation.")
            ph("cli-github-workflow-skipped")
            return
        workflows_path.mkdir(parents=True, exist_ok=True)
        from importlib.resources import files

        benchmark_mode = False
        benchmarks_root = config.get("benchmarks_root", "").strip()
        if benchmarks_root and benchmarks_root != "":
            benchmark_mode = inquirer_wrapper(
                inquirer.confirm,
                message="⚡️It looks like you've configured a benchmarks_root in your config. Would you like to run the Github action in benchmark mode? "
                " This will show the impact of Codeflash's suggested optimizations on your benchmarks",
                default=True,
            )

        optimize_yml_content = (
            files("codeflash").joinpath("cli_cmds", "workflows", "codeflash-optimize.yaml").read_text(encoding="utf-8")
        )
        materialized_optimize_yml_content = customize_codeflash_yaml_content(
            optimize_yml_content, config, git_root, benchmark_mode
        )
        with optimize_yaml_path.open("w", encoding="utf8") as optimize_yml_file:
            optimize_yml_file.write(materialized_optimize_yml_content)
        click.echo(f"{LF}✅ Created GitHub action workflow at {optimize_yaml_path}{LF}")
        try:
            existing_api_key = get_codeflash_api_key()
        except OSError:
            existing_api_key = None
        click.prompt(
            f"Next, you'll need to add your CODEFLASH_API_KEY as a secret to your GitHub repo.{LF}"
            f"Press Enter to open your repo's secrets page at {get_github_secrets_page_url(repo)} {LF}"
            f"Then, click 'New repository secret' to add your api key with the variable name CODEFLASH_API_KEY.{LF}"
            f"{'Here is your CODEFLASH_API_KEY: ' + existing_api_key + ' ' + LF}"
            if existing_api_key
            else "",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )
        click.launch(get_github_secrets_page_url(repo))
        click.echo(
            "🐙 I opened your Github secrets page! Note: if you see a 404, you probably don't have access to this "
            "repo's secrets; ask a repo admin to add it for you, or (not super recommended) you can temporarily "
            f"hard-code your api key into the workflow file.{LF}"
        )
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
    codeflash_section["benchmarks-root"] = setup_info.benchmarks_root if setup_info.benchmarks_root else ""
    codeflash_section["ignore-paths"] = setup_info.ignore_paths
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

    if is_github_app_installed_on_repo(owner, repo):
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
        while not is_github_app_installed_on_repo(owner, repo):
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
        click.echo(f"🔑 I found a CODEFLASH_API_KEY in your environment [{display_key}]!")

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
                    console.print(stripped)
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
