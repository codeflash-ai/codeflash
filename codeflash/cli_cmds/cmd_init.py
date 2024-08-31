from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys
from argparse import Namespace
from typing import Optional

import click
import git
import inquirer
import inquirer.themes
import tomlkit
from git import Repo
from pydantic.dataclasses import dataclass
from returns.pipeline import is_successful

from codeflash.api.cfapi import is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit, inquirer_wrapper, inquirer_wrapper_path
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import (
    get_codeflash_api_key,
)
from codeflash.code_utils.git_utils import get_repo_owner_and_name
from codeflash.code_utils.github_utils import (
    get_github_secrets_page_url,
    require_github_app_or_exit,
)
from codeflash.code_utils.shell_utils import (
    get_shell_rc_path,
    save_api_key_to_rc,
)
from codeflash.telemetry.posthog import ph
from codeflash.version import __version__ as version

CODEFLASH_LOGO: str = (
    f"{LF}"
    r"              __    _____         __ " + f"{LF}"
    r" _______  ___/ /__ / _/ /__ ____ / / " + f"{LF}"
    r"/ __/ _ \/ _  / -_) _/ / _ `(_-</ _ \ " + f"{LF}"
    r"\__/\___/\_,_/\__/_//_/\_,_/___/_//_/" + f"{LF}"
    f"{('v'+version).rjust(46)}{LF}"
    f"                          https://codeflash.ai{LF}"
    f"{LF}"
)


@dataclass(frozen=True)
class SetupInfo:
    module_root: str
    tests_root: str
    test_framework: str
    ignore_paths: list[str]
    formatter: str


def init_codeflash() -> None:
    try:
        click.echo(f"⚡️ Welcome to Codeflash! Let's get you set up.{LF}")

        did_add_new_key = prompt_api_key()

        setup_info: SetupInfo = collect_setup_info()

        configure_pyproject_toml(setup_info)

        install_github_app()

        click.echo(
            f"{LF}"
            f"⚡️ Codeflash is now set up! You can now run:{LF}"
            f"    codeflash --file <path-to-file> --function <function-name> to optimize a function within a file{LF}"
            f"    codeflash --file <path-to-file> to optimize all functions in a file{LF}"
            f"    codeflash --all to optimize all functions in all files in the module you selected ({setup_info.module_root}){LF}"
            # f"    codeflash --pr <pr-number> to optimize a PR{LF}"
            f"-or-{LF}"
            f"    codeflash --help to see all options{LF}",
        )
        if did_add_new_key:
            click.echo(
                "🐚 Don't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!",
            )
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
    run_tests = inquirer_wrapper(
        inquirer.confirm,
        message="⚡️ Do you want to run a sample optimization to make sure everything's set up correctly? (takes about 3 minutes)",
        default=True,
    )
    if run_tests:
        bubble_sort_path, bubble_sort_test_path = create_bubble_sort_file_and_test(args)
        run_end_to_end_test(args, bubble_sort_path, bubble_sort_test_path)


def collect_setup_info() -> SetupInfo:
    curdir = os.getcwd()
    # Check if the cwd is writable
    if not os.access(curdir, os.W_OK):
        click.echo(
            f"❌ The current directory isn't writable, please check your folder permissions and try again.{LF}",
        )
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
    ]
    valid_subdirs = [
        d
        for d in next(os.walk("."))[1]
        if not d.startswith(".") and not d.startswith("__") and d not in ignore_subdirs
    ]

    valid_module_subdirs = [d for d in valid_subdirs if d != "tests"]

    curdir_option = "current directory (" + curdir + ")"
    module_subdir_options = valid_module_subdirs + [curdir_option]

    module_root_answer = inquirer_wrapper(
        inquirer.list_input,
        message="Which Python module do you want me to optimize going forward? (Usually the top-most directory with "
        "all of your Python source code). Use arrow keys to select",
        choices=module_subdir_options,
        default=(project_name if project_name in module_subdir_options else module_subdir_options[0]),
    )
    module_root = "." if module_root_answer == curdir_option else module_root_answer
    ph("cli-project-root-provided")

    # Discover test directory
    default_tests_subdir = "tests"
    create_for_me_option = f"okay, create a tests{os.pathsep} directory for me!"
    test_subdir_options = valid_subdirs
    if "tests" not in valid_subdirs:
        test_subdir_options.append(create_for_me_option)
    custom_dir_option = "enter a custom directory..."
    test_subdir_options.append(custom_dir_option)
    tests_root_answer = inquirer_wrapper(
        inquirer.list_input,
        message="Where are your tests located? "
        f"(If you don't have any tests yet, I can create an empty tests{os.pathsep} directory for you)",
        choices=test_subdir_options,
        default=(
            default_tests_subdir if default_tests_subdir in test_subdir_options else test_subdir_options[0]
        ),
    )

    if tests_root_answer == create_for_me_option:
        tests_root = os.path.join(curdir, default_tests_subdir)
        os.mkdir(tests_root)
        click.echo(f"✅ Created directory {tests_root}{os.pathsep}{LF}")
    elif tests_root_answer == custom_dir_option:
        custom_tests_root_answer = inquirer_wrapper_path(
            "path",
            message=f"Enter the path to your tests directory inside {os.path.abspath('.') + os.path.sep} ",
            path_type=inquirer.Path.DIRECTORY,
            exists=True,
        )
        tests_root = custom_tests_root_answer["path"] if custom_tests_root_answer else apologize_and_exit()
    else:
        tests_root = tests_root_answer
    tests_root = os.path.relpath(tests_root, curdir)
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

    formatter = inquirer_wrapper(
        inquirer.list_input,
        message="Which code formatter do you use?",
        choices=["black", "ruff", "other", "don't use a formatter"],
        default="black",
        carousel=True,
    )

    # Ask for paths to ignore and update the setup_info dictionary
    # ignore_paths_input = click.prompt("Are there any paths Codeflash should ignore? (comma-separated, no spaces)",
    #                                   default='', show_default=False)
    # ignore_paths = ignore_paths_input.split(',') if ignore_paths_input else [f'tests{os.pathsep}']
    ignore_paths: list[str] = []

    return SetupInfo(
        module_root=module_root,
        tests_root=tests_root,
        test_framework=test_framework,
        ignore_paths=ignore_paths,
        formatter=formatter,
    )


def detect_test_framework(curdir, tests_root) -> Optional[str]:
    test_framework = None
    pytest_files = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
    pytest_config_patterns = {
        "pytest.ini": "[pytest]",
        "pyproject.toml": "[tool.pytest.ini_options]",
        "tox.ini": "[pytest]",
        "setup.cfg": "[tool:pytest]",
    }
    for pytest_file in pytest_files:
        file_path = os.path.join(curdir, pytest_file)
        if os.path.exists(file_path):
            with open(file_path, encoding="utf8") as file:
                contents = file.read()
                if pytest_config_patterns[pytest_file] in contents:
                    test_framework = "pytest"
                    break
        test_framework = "pytest"
    else:
        # Check if any python files contain a class that inherits from unittest.TestCase
        for filename in os.listdir(tests_root):
            if filename.endswith(".py"):
                with open(os.path.join(tests_root, filename), encoding="utf8") as file:
                    contents = file.read()
                    try:
                        node = ast.parse(contents)
                    except SyntaxError:
                        continue
                    if any(
                        isinstance(item, ast.ClassDef)
                        and any(
                            isinstance(base, ast.Attribute)
                            and base.attr == "TestCase"
                            or isinstance(base, ast.Name)
                            and base.id == "TestCase"
                            for base in item.bases
                        )
                        for item in node.body
                    ):
                        test_framework = "unittest"
                        break
    return test_framework


def check_for_toml_or_setup_file() -> Optional[str]:
    click.echo()
    click.echo("Checking for pyproject.toml or setup.py ...\r", nl=False)
    curdir = os.getcwd()
    pyproject_toml_path = os.path.join(curdir, "pyproject.toml")
    setup_py_path = os.path.join(curdir, "setup.py")
    project_name = None
    if os.path.exists(pyproject_toml_path):
        try:
            with open(pyproject_toml_path, encoding="utf8") as f:
                pyproject_toml_content = f.read()
            project_name = tomlkit.parse(pyproject_toml_content)["tool"]["poetry"]["name"]
            click.echo(f"✅ I found a pyproject.toml for your project {project_name}.")
            ph("cli-pyproject-toml-found-name")
        except Exception:
            click.echo("✅ I found a pyproject.toml for your project.")
            ph("cli-pyproject-toml-found")
    else:
        if os.path.exists(setup_py_path):
            with open(setup_py_path, encoding="utf8") as f:
                setup_py_content = f.read()
            project_name_match = re.search(
                r"setup\s*\([^)]*?name\s*=\s*['\"](.*?)['\"]",
                setup_py_content,
                re.DOTALL,
            )
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
            f"I need this file to store my configuration settings.",
        )
        ph("cli-no-pyproject-toml-or-setup-py")

        # Create a pyproject.toml file because it doesn't exist
        create_toml = inquirer_wrapper(
            inquirer.confirm,
            "Do you want me to create a pyproject.toml file in the current directory?",
            default=True,
            show_default=False,
        )
        if create_toml:
            ph("cli-create-pyproject-toml")
            # Define a minimal pyproject.toml content
            new_pyproject_toml = tomlkit.document()
            new_pyproject_toml["tool"] = {"codeflash": {}}
            try:
                with open(pyproject_toml_path, "w", encoding="utf8") as pyproject_file:
                    pyproject_file.write(tomlkit.dumps(new_pyproject_toml))

                # Check if the pyproject.toml file was created
                if os.path.exists(pyproject_toml_path):
                    click.echo(
                        f"✅ Created a pyproject.toml file at {pyproject_toml_path}",
                    )
                    click.pause()
                ph("cli-created-pyproject-toml")
            except OSError:
                click.echo(
                    "❌ Failed to create pyproject.toml. Please check your disk permissions and available space.",
                )
                apologize_and_exit()
        else:
            click.echo("⏩️ Skipping pyproject.toml creation.")
            apologize_and_exit()
    click.echo()
    return project_name


def install_github_actions() -> None:
    try:
        click.echo(
            "⚡️ Codeflash can automatically optimize new Github PRs for you when they're opened. Let's get that set up!",
        )
        config, config_file_path = parse_config_file()

        ph("cli-github-actions-install-started")
        repo = Repo(config["module_root"], search_parent_directories=True)

        owner, repo_name = get_repo_owner_and_name(repo)
        require_github_app_or_exit(owner, repo_name)

        git_root = repo.git.rev_parse("--show-toplevel")
        workflows_path = os.path.join(git_root, ".github", "workflows")
        optimize_yaml_path = os.path.join(workflows_path, "codeflash-optimize.yaml")

        confirm_creation_yes = inquirer_wrapper(
            inquirer.confirm,
            message=f"I'm going to create a new GitHub actions workflow file at {optimize_yaml_path} ... is this OK?",
            default=True,
        )
        ph(
            "cli-github-optimization-confirm-workflow-creation",
            {"confirm_creation": confirm_creation_yes},
        )
        if not confirm_creation_yes:
            click.echo("⏩️ Exiting workflow creation.")
            ph("cli-github-workflow-skipped")
            apologize_and_exit()
        os.makedirs(workflows_path, exist_ok=True)
        from importlib.resources import files

        py_version = sys.version_info
        python_version_string = f"'{py_version.major}.{py_version.minor}'"
        optimize_yml_content = (
            files("codeflash")
            .joinpath("cli_cmds", "workflows", "codeflash-optimize.yaml")
            .read_text(encoding="utf-8")
        )
        optimize_yml_content = optimize_yml_content.replace(
            "{{ python_version }}",
            python_version_string,
        )
        with open(optimize_yaml_path, "w", encoding="utf8") as optimize_yml_file:
            optimize_yml_file.write(optimize_yml_content)
        click.echo(f"✅ Created {optimize_yaml_path}{LF}")
        click.prompt(
            f"Next, you'll need to add your CODEFLASH_API_KEY as a secret to your GitHub repo.{LF}"
            + f"Press Enter to open your repo's secrets page at {get_github_secrets_page_url(repo)} ...{LF}"
            + f"Then, click 'New repository secret' to add your api key with the variable name CODEFLASH_API_KEY.{LF}",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )
        click.launch(get_github_secrets_page_url(repo))
        click.echo(
            "🐙 I opened your Github secrets page! Note: if you see a 404, you probably don't have access to this "
            + "repo's secrets; ask a repo admin to add it for you, or (not super recommended) you can temporarily "
            f"hard-code your api key into the workflow file.{LF}",
        )
        click.pause()
        click.echo()
        click.prompt(
            f"Finally, for the workflow to work, you'll need to edit the workflow file to install the right "
            f"Python version and any project dependencies.{LF}"
            + f"Press Enter to open {optimize_yaml_path} in your editor.{LF}",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )
        click.launch(optimize_yaml_path)
        click.echo(
            "📝 I opened the workflow file in your editor! You'll need to edit the steps that install the right Python "
            + f"version and any project dependencies. See the comments in the file for more details.{LF}",
        )
        click.pause()
        click.echo()
        click.echo(
            f"Please commit and push this GitHub actions file to your repo, and you're all set!{LF}"
            f"🚀 Codeflash is now configured to automatically optimize new Github PRs!{LF}",
        )
        ph("cli-github-workflow-created")
    except KeyboardInterrupt:
        apologize_and_exit()


# Create or update the pyproject.toml file with the Codeflash dependency & configuration
def configure_pyproject_toml(setup_info: SetupInfo) -> None:
    toml_path = os.path.join(os.getcwd(), "pyproject.toml")
    try:
        with open(toml_path, encoding="utf8") as pyproject_file:
            pyproject_data = tomlkit.parse(pyproject_file.read())
    except FileNotFoundError:
        click.echo(
            f"I couldn't find a pyproject.toml in the current directory.{LF}"
            f"Please create a new empty pyproject.toml file here, OR if you use poetry then run `poetry init`, OR run `codeflash init` again from a directory with an existing pyproject.toml file.",
        )
        apologize_and_exit()

    codeflash_section = tomlkit.table()
    codeflash_section.add(
        tomlkit.comment("All paths are relative to this pyproject.toml's directory."),
    )
    codeflash_section["module-root"] = setup_info.module_root
    codeflash_section["tests-root"] = setup_info.tests_root
    codeflash_section["test-framework"] = setup_info.test_framework
    codeflash_section["ignore-paths"] = setup_info.ignore_paths
    formatter = setup_info.formatter
    formatter_cmds = []
    if formatter == "black":
        formatter_cmds.append("black $file")
    elif formatter == "ruff":
        formatter_cmds.extend(["ruff check --exit-zero --fix $file", "ruff format $file"])
    elif formatter == "other":
        formatter_cmds.append("your-formatter $file")
    elif formatter == "don't use a formatter":
        formatter_cmds.append("disabled")
    codeflash_section["formatter-cmds"] = formatter_cmds
    # Add the 'codeflash' section, ensuring 'tool' section exists
    tool_section = pyproject_data.get("tool", tomlkit.table())
    tool_section["codeflash"] = codeflash_section
    pyproject_data["tool"] = tool_section

    click.echo("Writing Codeflash configuration ...\r", nl=False)
    with open(toml_path, "w", encoding="utf8") as pyproject_file:
        pyproject_file.write(tomlkit.dumps(pyproject_data))
    click.echo(f"✅ Added Codeflash configuration to {toml_path}")
    click.echo()


def install_github_app() -> None:
    git_repo = git.Repo(search_parent_directories=True)
    owner, repo = get_repo_owner_and_name(git_repo)

    if is_github_app_installed_on_repo(owner, repo):
        click.echo(
            "🐙 Looks like you've already installed the Codeflash GitHub app on this repository! Continuing…",
        )

    else:
        click.prompt(
            f"Finally, you'll need install the Codeflash GitHub app by choosing the repository you want to install Codeflash on.{LF}"
            + f"I will attempt to open the github app page - https://github.com/apps/codeflash-ai/installations/select_target {LF}"
            + f"Press Enter to open the page to let you install the app ...{LF}",
            default="",
            type=click.STRING,
            prompt_suffix="",
            show_default=False,
        )
        click.launch("https://github.com/apps/codeflash-ai/installations/select_target")
        click.prompt(
            f"Press Enter once you've finished installing the github app from https://github.com/apps/codeflash-ai/installations/select_target …{LF}",
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
                    f"In the meantime you can make local only optimizations by using the '--no-pr' flag with codeflash.{LF}",
                )
                break
            click.prompt(
                f"❌ It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo}.{LF}"
                f"Please install it from https://github.com/apps/codeflash-ai/installations/select_target {LF}"
                f"Press Enter to continue once you've finished installing the github app...{LF}",
                default="",
                type=click.STRING,
                prompt_suffix="",
                show_default=False,
            )
            count -= 1


class CFAPIKeyType(click.ParamType):
    name = "cfapi-key"

    def convert(self, value, param, ctx):
        value = value.strip()
        if value.startswith("cf-") or value == "":
            return value
        self.fail(
            f"That key [{value}] seems to be invalid. It should start with a 'cf-' prefix. Please try again.",
            param,
            ctx,
        )


# Returns True if the user entered a new API key, False if they used an existing one
def prompt_api_key() -> bool:
    try:
        existing_api_key = get_codeflash_api_key()
    except OSError:
        existing_api_key = None
    if existing_api_key:
        display_key = f"{existing_api_key[:3]}****{existing_api_key[-4:]}"
        click.echo(
            f"🔑 I found a CODEFLASH_API_KEY in your environment [{display_key}]!",
        )

        use_existing_key = inquirer_wrapper(
            inquirer.confirm,
            message="Do you want to use this key?",
            default=True,
            show_default=False,
        )
        if use_existing_key:
            ph("cli-existing-api-key-used")
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
                "You can also open this link manually: https://app.codeflash.ai/app/apikeys",
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


def create_bubble_sort_file_and_test(args: Namespace) -> None:
    bubble_sort_content = """def sorter(arr):
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
from {os.path.basename(args.module_root)}.bubble_sort import sorter

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
"""
    elif args.test_framework == "pytest":
        bubble_sort_test_content = f"""from {os.path.basename(args.module_root)}.bubble_sort import sorter

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
    else:
        click.echo(f"❌ Unsupported test framework: {args.test_framework}")
        apologize_and_exit()
    bubble_sort_path = os.path.join(args.module_root, "bubble_sort.py")
    with open(bubble_sort_path, "w", encoding="utf8") as bubble_sort_file:
        bubble_sort_file.write(bubble_sort_content)
    bubble_sort_test_path = os.path.join(args.tests_root, "test_bubble_sort.py")
    with open(bubble_sort_test_path, "w", encoding="utf8") as bubble_sort_test_file:
        bubble_sort_test_file.write(bubble_sort_test_content)
    click.echo(f"✅ Created {bubble_sort_path}")
    click.echo(f"✅ Created {bubble_sort_test_path}")
    return bubble_sort_path, bubble_sort_test_path


def run_end_to_end_test(args: Namespace, bubble_sort_path: str, bubble_sort_test_path: str) -> None:
    command = [
        "codeflash",
        "--file",
        "bubble_sort.py",
        "--function",
        "sorter",
    ]
    animation = "|/-\\"
    idx = 0
    sys.stdout.write("Running sample optimization... ")
    sys.stdout.flush()
    try:
        process = subprocess.run(
            command,
            text=True,
            cwd=args.module_root,
            check=False,
        )
    finally:
        # Delete the bubble_sort.py file after the test
        pathlib.Path(bubble_sort_path).unlink(missing_ok=True)
        pathlib.Path(bubble_sort_test_path).unlink(missing_ok=True)
        click.echo(f"{LF}🗑️ Deleted {bubble_sort_path}")
        click.echo(f"{LF}🗑️ Deleted {bubble_sort_test_path}")

    if process.returncode == 0:
        click.echo(
            f"{LF}✅ End-to-end test passed. Codeflash has been correctly set up!",
        )
    else:
        click.echo(
            f"{LF}❌ End-to-end test failed. Please check the logs above, and take a look at https://docs.codeflash.ai/getting-started/local-installation for help and troubleshooting.",
        )
