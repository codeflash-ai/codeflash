from __future__ import annotations

import os
import re
import subprocess
import sys
import webbrowser
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import git
import tomlkit
from git import InvalidGitRepositoryError, Repo
from inquirer_textual.common.Choice import Choice
from inquirer_textual.widgets.InquirerConfirm import InquirerConfirm
from inquirer_textual.widgets.InquirerSelect import InquirerSelect
from pydantic.dataclasses import dataclass

from codeflash.api.cfapi import get_user_id, is_github_app_installed_on_repo
from codeflash.cli_cmds import themed_prompts as prompts
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, logger
from codeflash.cli_cmds.extension import install_vscode_extension
from codeflash.cli_cmds.validators import (
    APIKeyValidator,
    NotEqualPathValidator,
    PathExistsValidator,
    PyprojectTomlValidator,
    RelativePathValidator,
    TomlFileValidator,
)
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import check_formatter_installed, get_codeflash_api_key
from codeflash.code_utils.git_utils import get_git_remotes, get_repo_owner_and_name
from codeflash.code_utils.github_utils import get_github_secrets_page_url
from codeflash.code_utils.oauth_handler import perform_oauth_signin
from codeflash.code_utils.shell_utils import get_shell_rc_path, is_powershell, save_api_key_to_rc
from codeflash.either import is_successful
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
    ignore_paths: list[str]
    formatter: Union[str, list[str]]
    git_remote: str
    enable_telemetry: bool


@dataclass(frozen=True)
class ConfigAnswers:
    """Parsed answers from the config widgets."""

    module_root: Path
    tests_root: Path
    formatter: str
    enable_telemetry: bool
    git_remote: str


@dataclass(frozen=True)
class VsCodeSetupInfo:
    module_root: str
    tests_root: str
    formatter: Union[str, list[str]]


class DependencyManager(Enum):
    PIP = auto()
    POETRY = auto()
    UV = auto()
    UNKNOWN = auto()


def collect_config(curdir: Path, auth_status: str, project_name: str | None, git_remotes: list[str]) -> ConfigAnswers:
    """Build config widgets, prompt user, and process results. Exits on cancellation."""
    # Build options for module root
    valid_module_subdirs, _ = get_suggestions(CommonSections.module_root)
    curdir_option = f"current directory ({curdir})"
    custom_module_dir_option = "enter a custom directory…"
    module_subdir_options = [*valid_module_subdirs, curdir_option, custom_module_dir_option]
    default_module_choice = project_name if project_name in module_subdir_options else module_subdir_options[0]

    # Build options for tests root
    tests_suggestions, default_tests_subdir = get_suggestions(CommonSections.tests_root)
    create_for_me_option = f"🆕 Create a new tests{os.pathsep} directory for me!"
    test_subdir_options: list[str | Choice] = list(tests_suggestions)
    if "tests" not in tests_suggestions:
        test_subdir_options.append(create_for_me_option)
    custom_tests_dir_option = "📁 Enter a custom directory…"
    test_subdir_options.append(custom_tests_dir_option)

    # Formatter choices
    formatter_choices = [
        Choice("⚫ black", data="black"),
        Choice("⚡ ruff", data="ruff"),
        Choice("🔧 other", data="other"),
        Choice("❌ don't use a formatter", data="don't use a formatter"),
    ]

    # Build widgets
    widgets: list = [
        InquirerSelect(
            "Which Python module do you want me to optimize?",
            choices=module_subdir_options,
            default=default_module_choice,
            mandatory=True,
        ),
        InquirerSelect(
            "Where are your tests located?",
            choices=test_subdir_options,
            default=(default_tests_subdir or test_subdir_options[0]),
            mandatory=True,
        ),
        InquirerSelect(
            "Which code formatter do you use?", choices=formatter_choices, default=formatter_choices[0], mandatory=True
        ),
        InquirerConfirm("Help us improve Codeflash by sharing anonymous usage data?", default=True),
    ]

    has_multiple_remotes = len(git_remotes) > 1
    if has_multiple_remotes:
        git_remote_choices: list[str | Choice] = list(git_remotes)
        widgets.append(
            InquirerSelect(
                "Which git remote should Codeflash use for Pull Requests?",
                choices=git_remote_choices,
                default="origin" if "origin" in git_remotes else git_remotes[0],
                mandatory=True,
            )
        )

    # Run prompt
    header = f"{auth_status}\n\n⚙️ Project Configuration" if auth_status else "⚙️ Project Configuration"
    result = prompts.multi(widgets, header=header)
    if result.command is None:
        apologize_and_exit()

    # Unpack results
    if has_multiple_remotes:
        module_root_answer, tests_root_answer, formatter_answer, enable_telemetry, git_remote = result.value
    else:
        module_root_answer, tests_root_answer, formatter_answer, enable_telemetry = result.value
        git_remote = git_remotes[0] if len(git_remotes) == 1 else ""

    # Process module root
    module_root = process_module_root_answer(module_root_answer, curdir_option, custom_module_dir_option)
    ph("cli-project-root-provided")

    # Process tests root
    tests_root = process_tests_root_answer(
        tests_root_answer, create_for_me_option, custom_tests_dir_option, curdir, default_tests_subdir, module_root
    )
    tests_root = tests_root.relative_to(curdir)
    ph("cli-tests-root-provided")

    # Extract formatter value
    formatter = formatter_answer.data if isinstance(formatter_answer, Choice) else formatter_answer

    return ConfigAnswers(
        module_root=Path(module_root),
        tests_root=tests_root,
        formatter=cast("str", formatter),
        enable_telemetry=enable_telemetry,
        git_remote=str(git_remote),
    )


def init_codeflash() -> None:
    try:
        curdir = Path.cwd()

        try:
            existing_api_key = get_codeflash_api_key()
        except OSError:
            existing_api_key = None

        did_add_new_key = False
        auth_status = ""
        if existing_api_key:
            auth_status = f"🔑 API Key found [{existing_api_key[:3]}****{existing_api_key[-4:]}]"
        else:
            auth_choices = [Choice("🔐 Login with Codeflash", data="oauth"), Choice("🔑 Use API key", data="api_key")]
            method = prompts.select_or_exit(
                "How would you like to authenticate?",
                choices=auth_choices,
                default=auth_choices[0],
                header="⚡️ Welcome to Codeflash!\n\nThis setup will take just a few minutes.",
            )
            if method == "api_key":
                enter_api_key_and_save_to_rc()
                ph("cli-new-api-key-entered")
            else:
                api_key = perform_oauth_signin()
                if not api_key:
                    apologize_and_exit()
                    return  # unreachable, satisfies type checker
                save_api_key_and_set_env(api_key)
                ph("cli-oauth-signin-completed")
            auth_status = "✅ Signed in successfully!"
            did_add_new_key = True

        should_modify, config = should_modify_pyproject_toml()
        git_remote = config.get("git_remote", "origin") if config else "origin"

        if should_modify:
            if not os.access(curdir, os.W_OK):
                console.print(
                    f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}"
                )
                console.print("It's likely you don't have write permissions for this folder.")
                sys.exit(1)

            project_name = check_for_toml_or_setup_file()

            try:
                repo = Repo(curdir, search_parent_directories=True)
                git_remotes = get_git_remotes(repo)
            except InvalidGitRepositoryError:
                git_remotes = []

            answers = collect_config(curdir, auth_status, project_name, git_remotes)
            git_remote = answers.git_remote

            setup_info = CLISetupInfo(
                module_root=str(answers.module_root),
                tests_root=str(answers.tests_root),
                benchmarks_root=None,
                ignore_paths=[],
                formatter=answers.formatter,
                git_remote=answers.git_remote,
                enable_telemetry=answers.enable_telemetry,
            )
            if not configure_pyproject_toml(setup_info):
                apologize_and_exit()

        try:
            git.Repo(search_parent_directories=True)
            in_git_repo = True
        except git.InvalidGitRepositoryError:
            in_git_repo = False

        integration_options = [Choice("📦 VSCode Extension", data="vscode")]
        if in_git_repo:
            integration_options = [
                Choice("🐙 GitHub App", data="github_app"),
                Choice("⚙️ GitHub Actions", data="github_actions"),
                *integration_options,
            ]

        integrations_header = "🔧 Optional Integrations\n\n💡 SPACE to select, ENTER to confirm."
        if not should_modify and auth_status:
            integrations_header = f"{auth_status}\n\n{integrations_header}"

        selected = prompts.checkbox_or_default(
            "Which integrations would you like to install?",
            choices=integration_options,
            default_on_cancel=[],
            header=integrations_header,
        )
        for choice in selected:
            if choice.data == "github_app":
                install_github_app(git_remote)
            elif choice.data == "github_actions":
                install_github_actions(override_formatter_check=True)
            elif choice.data == "vscode":
                install_vscode_extension()

        header = "🎉 Codeflash is now set up!\n\nCommands:\n  codeflash --file <path> --function <name>\n  codeflash optimize <script.py>\n  codeflash --all\n  codeflash --help"
        if did_add_new_key:
            rc = get_shell_rc_path()
            cmd = f". {rc}" if is_powershell() else f"call {rc}" if os.name == "nt" else f"source {rc}"
            header += f"\n\n🐚 Restart your shell or run: {cmd}"
        prompts.confirm("Ready to start optimizing?", default=True, header=header)

        ph("cli-installation-successful", {"did_add_new_key": did_add_new_key})
        sys.exit(0)
    except KeyboardInterrupt:
        apologize_and_exit()


def ask_run_end_to_end_test(args: Namespace) -> None:
    result = prompts.select(
        "⚡️ Do you want to run a sample optimization to make sure everything's set up correctly? (takes about 3 minutes)",
        choices=["Yes", "No"],
        default="Yes",
    )

    console.rule()

    if result.command is not None and result.value == "Yes":
        file_path = create_find_common_tags_file(args, "find_common_tags.py")
        run_end_to_end_test(args, file_path)


def should_modify_pyproject_toml() -> tuple[bool, dict[str, Any] | None]:
    """Check if the current directory contains a valid pyproject.toml file with codeflash config.

    If it does, ask the user if they want to re-configure it.
    """
    pyproject_toml_path = Path.cwd() / "pyproject.toml"

    toml_validator = TomlFileValidator()
    toml_result = toml_validator.validate(str(pyproject_toml_path))
    if not toml_result.is_valid:
        return True, None  # File doesn't exist or isn't valid, needs configuration

    config_validator = PyprojectTomlValidator()
    config_result = config_validator.validate(str(pyproject_toml_path))
    if not config_result.is_valid:
        return True, None  # Configuration invalid, needs re-configuration

    try:
        config, _ = parse_config_file(pyproject_toml_path)
    except Exception:
        return True, None

    answer = prompts.select_or_exit(
        "✅ A valid Codeflash config already exists in this project. Do you want to re-configure it?",
        choices=["Yes", "No"],
        default="No",
    )

    return answer == "Yes", config


# common sections between normal mode and lsp mode
class CommonSections(Enum):
    module_root = "module_root"
    tests_root = "tests_root"
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


def get_suggestions(section: CommonSections | str) -> tuple[list[str], Optional[str]]:
    valid_subdirs = get_valid_subdirs()
    section_value = section.value if isinstance(section, CommonSections) else section
    if section_value == CommonSections.module_root.value:
        return [d for d in valid_subdirs if d != "tests"], None
    if section_value == CommonSections.tests_root.value:
        default = "tests" if "tests" in valid_subdirs else None
        return valid_subdirs, default
    if section_value == CommonSections.formatter_cmds.value:
        return ["disabled", "ruff", "black"], "disabled"
    msg = f"Unknown section: {section}"
    raise ValueError(msg)


def prompt_custom_directory(message: str, title: str, additional_validators: list | None = None) -> Path | None:
    validators = [PathExistsValidator(), RelativePathValidator()]
    if additional_validators:
        validators.extend(additional_validators)

    result = prompts.text("Enter the path to your directory", validators=validators, header=f"{title}\n\n{message}")

    if result.command is None:
        return None

    return Path(result.value)


def process_module_root_answer(answer: str, curdir_option: str, custom_option: str) -> str | Path:
    """Process module root answer and return the path. Exits on cancellation."""
    if answer == curdir_option:
        return "."
    if answer == custom_option:
        custom_path = prompt_custom_directory(
            "📂 Enter a custom module directory path.\n\nPlease provide the path to your Python module directory.",
            "📂 Custom Directory",
        )
        if custom_path is None:
            apologize_and_exit()
        return custom_path  # type: ignore[return-value]  # apologize_and_exit() never returns
    return answer


def process_tests_root_answer(
    answer: str,
    create_option: str,
    custom_option: str,
    curdir: Path,
    default_tests_subdir: str | None,
    module_root: str | Path,
) -> Path:
    """Process tests root answer and return the path. Exits on cancellation."""
    if answer == create_option:
        tests_root = curdir / (default_tests_subdir or "tests")
        tests_root.mkdir()
        console.print(f"✅ Created directory {tests_root}{os.path.sep}{LF}")
        return tests_root

    if answer == custom_option:
        resolved_module_root = (curdir / Path(module_root)).resolve()
        custom_path = prompt_custom_directory(
            "🧪 Enter a custom test directory path.\n\nPlease provide the path to your test directory, relative to the current directory.",
            "🧪 Custom Test Directory",
            additional_validators=[NotEqualPathValidator(resolved_module_root)],
        )
        if custom_path is None:
            apologize_and_exit()
        return curdir / custom_path  # type: ignore[operator]  # apologize_and_exit() never returns

    return curdir / Path(answer)


def detect_project_name(pyproject_path: Path, setup_py_path: Path) -> str | None:
    """Detect project name from pyproject.toml or setup.py."""
    if pyproject_path.exists():
        try:
            content = pyproject_path.read_text(encoding="utf8")
            parsed = tomlkit.parse(content)
            name = parsed.get("tool", {}).get("poetry", {}).get("name")  # type: ignore[union-attr]
            if name:
                console.print(f"✅ I found a pyproject.toml for your project {name}.")
                ph("cli-pyproject-toml-found-name")
                return cast("str", name)
        except Exception:  # noqa: S110 - Intentionally silent, project name is optional
            pass
        console.print("✅ I found a pyproject.toml for your project.")
        ph("cli-pyproject-toml-found")
        return None

    if setup_py_path.exists():
        content = setup_py_path.read_text(encoding="utf8")
        match = re.search(r"setup\s*\([^)]*?name\s*=\s*['\"](.*?)['\"]", content, re.DOTALL)
        if match:
            name = match.group(1)
            console.print(f"✅ Found setup.py for your project {name}")
            ph("cli-setup-py-found-name")
            return name
        console.print("✅ Found setup.py.")
        ph("cli-setup-py-found")
    return None


def check_for_toml_or_setup_file() -> str | None:
    """Check for pyproject.toml or setup.py and return project name if found."""
    curdir = Path.cwd()
    pyproject_toml_path = curdir / "pyproject.toml"
    setup_py_path = curdir / "setup.py"

    project_name = detect_project_name(pyproject_toml_path, setup_py_path)

    # If no pyproject.toml exists, prompt to create one
    if not pyproject_toml_path.exists():
        ph("cli-no-pyproject-toml-or-setup-py")

        answer = prompts.select_or_exit(
            "Create pyproject.toml in the current directory?",
            choices=["Yes", "No"],
            default="Yes",
            header=(
                f"📋 pyproject.toml Required\n\n"
                f"💡 No pyproject.toml found in {curdir}.\n\n"
                "This file is essential for Codeflash to store its configuration.\n"
                "Please ensure you are running `codeflash init` from your project's root directory."
            ),
        )
        if answer == "No":
            apologize_and_exit()
        create_empty_pyproject_toml(pyproject_toml_path)

    return project_name


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
            prompts.confirm(
                "Continue?",
                default=True,
                header=(
                    f"🎉 Success!\n\n"
                    f"✅ Created a pyproject.toml file at {pyproject_toml_path}\n\n"
                    "Your project is now ready for Codeflash configuration!"
                ),
            )
        ph("cli-created-pyproject-toml")
    except OSError:
        console.print("❌ Failed to create pyproject.toml. Please check your disk permissions and available space.")
        apologize_and_exit()


def install_github_actions(override_formatter_check: bool = False) -> None:  # noqa: FBT001, FBT002
    try:
        config, _config_file_path = parse_config_file(override_formatter_check=override_formatter_check)
        ph("cli-github-actions-install-started")

        try:
            repo = Repo(config["module_root"], search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            console.print(
                "Skipping GitHub action installation for continuous optimization because you're not in a git repository."
            )
            return

        git_root = Path(repo.git.rev_parse("--show-toplevel"))
        workflows_path = git_root / ".github" / "workflows"
        optimize_yaml_path = workflows_path / "codeflash.yaml"

        # Check if workflow exists
        if optimize_yaml_path.exists():
            answer = prompts.select_or_exit(
                f"GitHub Actions workflow already exists at {optimize_yaml_path}. Overwrite?",
                choices=["No", "Yes"],
                default="No",
            )
            if answer == "No":
                ph("cli-github-workflow-skipped")
                return
            ph("cli-github-optimization-confirm-workflow-overwrite", {"confirm_overwrite": True})

        # Confirm setup
        answer = prompts.select_or_exit(
            "Set up GitHub Actions for continuous optimization?",
            choices=["Yes", "No"],
            default="Yes",
            header=(
                "🤖 GitHub Actions Setup\n\n"
                "GitHub Actions will automatically optimize your code in every pull request. "
                "This is the recommended way to use Codeflash for continuous optimization."
            ),
        )
        if answer == "No":
            ph("cli-github-workflow-skipped")
            return

        ph("cli-github-optimization-confirm-workflow-creation", {"confirm_creation": True})
        workflows_path.mkdir(parents=True, exist_ok=True)

        # Check for benchmark mode
        benchmark_mode = False
        benchmarks_root = config.get("benchmarks_root", "").strip()
        if benchmarks_root:
            answer = prompts.select_or_exit(
                "Run GitHub Actions in benchmark mode?",
                choices=["Yes", "No"],
                default="Yes",
                header=(
                    "📊 Benchmark Mode Available\n\n"
                    "I noticed you've configured a benchmarks_root in your config. "
                    "Benchmark mode will show the performance impact of Codeflash's optimizations on your benchmarks."
                ),
            )
            benchmark_mode = answer == "Yes"

        # Create workflow file
        from importlib.resources import files

        optimize_yml_content = (files("codeflash") / "cli_cmds" / "workflows" / "codeflash-optimize.yaml").read_text(
            encoding="utf-8"
        )
        materialized_content = customize_codeflash_yaml_content(optimize_yml_content, config, git_root, benchmark_mode)
        optimize_yaml_path.write_text(materialized_content, encoding="utf8")

        # Guide user to add GitHub secret
        try:
            existing_api_key = get_codeflash_api_key()
        except OSError:
            existing_api_key = None

        secrets_url = get_github_secrets_page_url(repo)
        secrets_header = (
            f"🎉 Workflow Created!\n\n"
            f"✅ Created GitHub action workflow at {optimize_yaml_path}\n\n"
            "Your repository is now configured for continuous optimization!\n\n"
            "🔐 Next Step: Add API Key as GitHub Secret\n\n"
            "You'll need to add your CODEFLASH_API_KEY as a secret to your GitHub repository.\n\n"
            "📋 Steps:\n"
            "1. Select Yes to open your repo's secrets page\n"
            "2. Click 'New repository secret'\n"
            "3. Add your API key with the variable name CODEFLASH_API_KEY"
        )
        if existing_api_key:
            secrets_header += f"\n\n🔑 Your API Key: {existing_api_key}"

        open_secrets = prompts.select_or_exit(
            f"Open GitHub secrets page? ({secrets_url})", choices=["Yes", "No"], default="Yes", header=secrets_header
        )
        if open_secrets == "Yes":
            webbrowser.open(secrets_url)

        prompts.confirm(
            "Continue?",
            default=True,
            header=(
                "🚀 Almost done!\n\n"
                "Please edit, commit and push this GitHub actions file to your repo, and you're all set!\n\n"
                "Note: If you see a 404 on the secrets page, you probably don't have access to this repo's secrets. "
                "Ask a repo admin to add it for you."
            ),
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


def get_dependency_installation_commands(dep_manager: DependencyManager) -> str:
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
    config: dict[str, Any],
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
        console.print(
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
        console.print(
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
        console.print(
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
        cli_info = cast("CLISetupInfo", setup_info)
        codeflash_section["module-root"] = cli_info.module_root
        codeflash_section["tests-root"] = cli_info.tests_root
        codeflash_section["ignore-paths"] = cli_info.ignore_paths
        if not cli_info.enable_telemetry:
            codeflash_section["disable-telemetry"] = not cli_info.enable_telemetry
        if cli_info.git_remote not in ["", "origin"]:
            codeflash_section["git-remote"] = cli_info.git_remote

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
    console.print(f"Added Codeflash configuration to {toml_path}")
    console.print()
    return True


def prompt_github_app_install(owner: str, repo: str) -> None:
    """Prompt user to install GitHub app and wait for confirmation."""
    app_url = "https://github.com/apps/codeflash-ai/installations/select_target"

    open_page = prompts.select_or_exit(
        f"Open GitHub App installation page? ({app_url})",
        choices=["Yes", "No"],
        default="Yes",
        header=(
            f"🐙 GitHub App Installation\n\n"
            f"You'll need to install the Codeflash GitHub app for {owner}/{repo}.\n\n"
            "I'll open the installation page where you can select your repository."
        ),
    )
    if open_page == "Yes":
        webbrowser.open(app_url)

    prompts.confirm("Continue once you've completed the installation?", default=True)


def install_github_app(git_remote: str) -> None:
    try:
        git_repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        console.print("Skipping GitHub app installation because you're not in a git repository.")
        return

    if git_remote not in get_git_remotes(git_repo):
        console.print(f"Skipping GitHub app installation, remote ({git_remote}) does not exist in this repository.")
        return

    owner, repo = get_repo_owner_and_name(git_repo, git_remote)

    if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
        console.print(
            f"🐙 Looks like you've already installed the Codeflash GitHub app on this repository ({owner}/{repo})! Continuing…"
        )
        return

    # Not installed - prompt for installation
    try:
        prompt_github_app_install(owner, repo)

        # Verify installation with retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
                console.print(f"✅ GitHub App installed successfully for {owner}/{repo}!")
                break

            if attempt == max_retries:
                console.print(
                    f"❌ GitHub App not detected on {owner}/{repo}.\n"
                    f"You won't be able to create PRs until you install it.\n"
                    f"Use the '--no-pr' flag for local-only optimizations."
                )
                break

            console.print(
                f"❌ GitHub App not detected on {owner}/{repo}.\nPress Enter to check again after installation..."
            )
            console.input()
    except (KeyboardInterrupt, EOFError):
        console.print()  # Clean line for next prompt


def save_api_key_and_set_env(api_key: str) -> None:
    """Save API key to shell RC file and set environment variable."""
    shell_rc_path = get_shell_rc_path()
    if not shell_rc_path.exists() and os.name == "nt":
        shell_rc_path.parent.mkdir(parents=True, exist_ok=True)
        shell_rc_path.touch()
        console.print(f"✅ Created {shell_rc_path}")

    result = save_api_key_to_rc(api_key)
    if is_successful(result):
        console.print(result.unwrap())
    else:
        console.print(result.failure())
        console.input("Press Enter to continue...")

    os.environ["CODEFLASH_API_KEY"] = api_key


def enter_api_key_and_save_to_rc() -> None:
    """Prompt for API key with validation and save to shell RC file."""
    browser_launched = False
    api_key = ""

    while api_key == "":
        result = prompts.text(
            f"Enter your Codeflash API key{' [or press Enter to open your API key page]' if not browser_launched else ''}",
            validators=APIKeyValidator(),
        )
        if result.command is None:
            apologize_and_exit()
            return

        api_key = result.value.strip()

        # If empty, open browser and try again
        if not api_key and not browser_launched:
            console.print(
                f"Opening your Codeflash API key page. Grab a key from there!{LF}"
                "You can also open this link manually: https://app.codeflash.ai/app/apikeys"
            )
            webbrowser.open("https://app.codeflash.ai/app/apikeys")
            browser_launched = True  # This does not work on remote consoles
    get_user_id(api_key=api_key)  # Used to verify whether the API key is valid.
    save_api_key_and_set_env(api_key)


find_common_tags_content = """from __future__ import annotations


def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0].get("tags", [])
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article.get("tags", [])]
    return set(common_tags)
"""


def create_find_common_tags_file(args: Namespace, file_name: str) -> Path:
    file_path = Path(args.module_root) / file_name
    lsp_enabled = is_LSP_enabled()
    if file_path.exists() and not lsp_enabled:
        answer = prompts.select_or_exit(
            f"🤔 {file_path} already exists. Do you want to overwrite it?", choices=["Yes", "No"], default="Yes"
        )
        if answer == "No":
            apologize_and_exit()
        console.rule()

    file_path.write_text(find_common_tags_content, encoding="utf8")
    logger.info(f"Created demo optimization file: {file_path}")

    return file_path


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


def create_bubble_sort_file_and_test(args: Namespace) -> tuple[str, str]:
    # Always use pytest for tests
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
        answer = prompts.select_or_exit(
            f"🤔 {bubble_sort_path} already exists. Do you want to overwrite it?", choices=["Yes", "No"], default="Yes"
        )
        if answer == "No":
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
