from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
import inquirer
import tomlkit
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, logger
from codeflash.cli_cmds.extension import install_vscode_extension
from codeflash.cli_cmds.github_workflow import install_github_actions
from codeflash.cli_cmds.init_auth import install_github_app, prompt_api_key
from codeflash.cli_cmds.init_config import (
    CLISetupInfo,
    CodeflashTheme,
    CommonSections,
    ask_for_telemetry,
    configure_pyproject_toml,
    create_empty_pyproject_toml,
    get_suggestions,
    should_modify_pyproject_toml,
)
from codeflash.cli_cmds.init_javascript import ProjectLanguage, detect_project_language, init_js_project
from codeflash.code_utils.code_utils import validate_relative_directory_path
from codeflash.code_utils.compat import LF
from codeflash.code_utils.env_utils import check_formatter_installed
from codeflash.code_utils.git_utils import get_git_remotes
from codeflash.code_utils.shell_utils import get_shell_rc_path, is_powershell
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.telemetry.posthog_cf import ph

if TYPE_CHECKING:
    from argparse import Namespace


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

        # Detect project language
        project_language = detect_project_language()

        if project_language in (ProjectLanguage.JAVASCRIPT, ProjectLanguage.TYPESCRIPT):
            init_js_project(project_language)
            return

        # Python project flow
        did_add_new_key = prompt_api_key()

        should_modify, config = should_modify_pyproject_toml()

        git_remote = config.get("git_remote", "origin") if config else "origin"

        if should_modify:
            setup_info: CLISetupInfo = collect_setup_info()
            git_remote = setup_info.git_remote
            configured = configure_pyproject_toml(setup_info)
            if not configured:
                apologize_and_exit()

        install_github_app(git_remote)

        install_github_actions(override_formatter_check=True)

        install_vscode_extension()

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
        file_path = create_find_common_tags_file(args, "find_common_tags.py")
        run_end_to_end_test(args, file_path)


def collect_setup_info() -> CLISetupInfo:
    from git import InvalidGitRepositoryError, Repo

    curdir = Path.cwd()
    # Check if the cwd is writable
    if not os.access(curdir, os.W_OK):
        click.echo(f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}")
        click.echo("It's likely you don't have write permissions for this folder.")
        sys.exit(1)

    # Check for the existence of pyproject.toml or setup.py
    project_name = check_for_toml_or_setup_file()
    valid_module_subdirs, _ = get_suggestions(CommonSections.module_root)

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

        # Retry loop for custom module root path
        custom_module_root: Path | None = None
        while custom_module_root is None:
            custom_questions = [
                inquirer.Path(
                    "custom_path",
                    message="Enter the path to your module directory",
                    path_type=inquirer.Path.DIRECTORY,
                    exists=True,
                )
            ]

            custom_answers = inquirer.prompt(custom_questions, theme=CodeflashTheme())
            if not custom_answers:
                apologize_and_exit()

            custom_path_str = str(custom_answers["custom_path"])
            # Validate the path is safe
            is_valid, error_msg = validate_relative_directory_path(custom_path_str)
            if not is_valid:
                click.echo(f"❌ Invalid path: {error_msg}")
                click.echo("Please enter a valid relative directory path.")
                console.print()  # Add spacing before retry
                continue  # Retry the prompt
            custom_module_root = Path(custom_path_str)
        module_root = str(custom_module_root)
    else:
        module_root = module_root_answer
    ph("cli-project-root-provided")

    # Discover test directory
    create_for_me_option = f"🆕 Create a new tests{os.pathsep} directory for me!"
    tests_suggestions, default_tests_subdir = get_suggestions(CommonSections.tests_root)
    test_subdir_options = [sub_dir for sub_dir in tests_suggestions if sub_dir != module_root]
    if "tests" not in tests_suggestions:
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
            default=(default_tests_subdir or test_subdir_options[0]),
            carousel=True,
        )
    ]

    tests_answers = inquirer.prompt(tests_questions, theme=CodeflashTheme())
    if not tests_answers:
        apologize_and_exit()
    tests_root_answer = tests_answers["tests_root"]

    if tests_root_answer == create_for_me_option:
        tests_root = Path(curdir) / (default_tests_subdir or "tests")
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

        # Retry loop for custom tests root path
        custom_tests_root: Path | None = None
        while custom_tests_root is None:
            custom_tests_questions = [
                inquirer.Path(
                    "custom_tests_path",
                    message="Enter the path to your tests directory",
                    path_type=inquirer.Path.DIRECTORY,
                    exists=True,
                )
            ]

            custom_tests_answers = inquirer.prompt(custom_tests_questions, theme=CodeflashTheme())
            if not custom_tests_answers:
                apologize_and_exit()

            custom_tests_path_str = str(custom_tests_answers["custom_tests_path"])
            # Validate the path is safe
            is_valid, error_msg = validate_relative_directory_path(custom_tests_path_str)
            if not is_valid:
                click.echo(f"❌ Invalid path: {error_msg}")
                click.echo("Please enter a valid relative directory path.")
                console.print()  # Add spacing before retry
                continue  # Retry the prompt
            custom_tests_root = Path(curdir) / Path(custom_tests_path_str)
        tests_root = custom_tests_root
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

    benchmarks_root = None

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

    enable_telemetry = ask_for_telemetry()

    ignore_paths: list[str] = []
    return CLISetupInfo(
        module_root=str(module_root),
        tests_root=str(tests_root),
        benchmarks_root=str(benchmarks_root) if benchmarks_root else None,
        ignore_paths=ignore_paths,
        formatter=cast("str", formatter),
        git_remote=str(git_remote),
        enable_telemetry=enable_telemetry,
    )


def check_for_toml_or_setup_file() -> str | None:
    click.echo()
    click.echo("Checking for pyproject.toml or setup.py…\r", nl=False)
    curdir = Path.cwd()
    pyproject_toml_path = curdir / "pyproject.toml"
    setup_py_path = curdir / "setup.py"
    package_json_path = curdir / "package.json"
    project_name = None

    # Check if this might be a JavaScript/TypeScript project that wasn't detected
    if package_json_path.exists() and not pyproject_toml_path.exists() and not setup_py_path.exists():
        js_redirect_panel = Panel(
            Text(
                f"📦 I found a package.json in {curdir}.\n\n"
                "This looks like a JavaScript/TypeScript project!\n"
                "Redirecting to JavaScript setup...",
                style="cyan",
            ),
            title="🟨 JavaScript Project Detected",
            border_style="bright_yellow",
        )
        console.print(js_redirect_panel)
        console.print()
        ph("cli-js-project-redirect")

        # Redirect to JS init
        from codeflash.cli_cmds.init_javascript import ProjectLanguage, detect_project_language, init_js_project

        project_language = detect_project_language()
        if project_language in (ProjectLanguage.JAVASCRIPT, ProjectLanguage.TYPESCRIPT):
            init_js_project(project_language)
            sys.exit(0)  # init_js_project handles its own exit, but ensure we don't continue

    if pyproject_toml_path.exists():
        try:
            pyproject_toml_content = pyproject_toml_path.read_text(encoding="utf8")
            project_name = tomlkit.parse(pyproject_toml_content)["tool"]["poetry"]["name"]  # type: ignore[index]
            click.echo(f"✅ I found a pyproject.toml for your project {project_name}.")
            ph("cli-pyproject-toml-found-name")
        except Exception:
            click.echo("✅ I found a pyproject.toml for your project.")
            ph("cli-pyproject-toml-found")
    elif setup_py_path.exists():
        setup_py_content = setup_py_path.read_text(encoding="utf8")
        project_name_match = re.search(r"setup\s*\([^)]*?name\s*=\s*['\"](.*?)['\"]", setup_py_content, re.DOTALL)
        if project_name_match:
            project_name = project_name_match.group(1)
            click.echo(f"✅ Found setup.py for your project {project_name}")
            ph("cli-setup-py-found-name")
        else:
            click.echo("✅ Found setup.py.")
            ph("cli-setup-py-found")
    else:
        # No Python config files found - show appropriate message
        # Check again if this might be a JS project
        if package_json_path.exists():
            js_hint_panel = Panel(
                Text(
                    f"📦 I found a package.json but no pyproject.toml in {curdir}.\n\n"
                    "If this is a JavaScript/TypeScript project, please run:\n"
                    "  codeflash init\n\n"
                    "from the project root directory.",
                    style="yellow",
                ),
                title="🤔 Mixed Project?",
                border_style="bright_yellow",
            )
            console.print(js_hint_panel)
        else:
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
            create_empty_pyproject_toml(pyproject_toml_path)
    click.echo()
    return cast("str", project_name)


def create_find_common_tags_file(args: Namespace, file_name: str) -> Path:
    find_common_tags_content = """from __future__ import annotations


def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0].get("tags", [])
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article.get("tags", [])]
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
