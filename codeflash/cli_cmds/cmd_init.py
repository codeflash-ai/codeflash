import os
import re
import subprocess
import sys
import time
from typing import Optional

import click
import tomlkit
from git import Repo

from codeflash.analytics.posthog import ph
from codeflash.cli_cmds.cli import CODEFLASH_LOGO
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.code_utils.git_utils import get_github_secrets_page_url


def init_codeflash():
    click.echo(CODEFLASH_LOGO)
    click.echo("⚡️ Welcome to CodeFlash! Let's get you set up.\n")

    prompt_api_key()

    setup_info: dict[str, str] = {}

    collect_setup_info(setup_info)

    configure_pyproject_toml(setup_info)

    prompt_github_action(setup_info)

    run_tests = click.confirm(
        "Do you want to run a sample optimization to ensure everything is set up correctly? This will take about 3 minutes.",
        default=True,
    )

    if run_tests:
        create_bubble_sort_file(setup_info)
        run_end_to_end_test(setup_info)

    click.echo(
        "\n"
        "⚡️ CodeFlash is now set up! You can now run:\n"
        "    codeflash --file <path-to-file> --function <function-name> to optimize a function within a file\n"
        "    codeflash --file <path-to-file> to optimize all functions in a file\n"
        # "    codeflash --pr <pr-number> to optimize a PR\n"
        "-or-\n"
        "    codeflash --help to see all options\n"
        "Please restart your shell to load the CODEFLASH_API_KEY environment variable."
    )

    ph("cli-installation-successful")


def create_bubble_sort_file(setup_info: dict[str, str]):
    bubble_sort_content = """def sorter(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr
"""
    bubble_sort_path = os.path.join(setup_info["project_root"], "bubble_sort.py")
    with open(bubble_sort_path, "w") as bubble_sort_file:
        bubble_sort_file.write(bubble_sort_content)
    click.echo(f"✅ Created {bubble_sort_path}")


def run_end_to_end_test(setup_info: dict[str, str]):
    command = [
        "codeflash",
        "--file",
        "bubble_sort.py",
        "--function",
        "sorter",
    ]
    animation = "|/-\\"
    idx = 0
    sys.stdout.write("Running end-to-end test... ")
    sys.stdout.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=setup_info["project_root"],
    )
    while process.poll() is None:
        sys.stdout.write(animation[idx % len(animation)])
        sys.stdout.flush()
        time.sleep(0.5)
        sys.stdout.write("\b")
        idx += 1

    sys.stdout.write(" ")  # Clear the last animation character
    sys.stdout.flush()
    stderr = process.stderr.read()
    if stderr:
        click.echo(stderr.strip())

    bubble_sort_path = os.path.join(setup_info["project_root"], "bubble_sort.py")
    if process.returncode == 0:
        click.echo("\n✅ End-to-end test passed. CodeFlash has been correctly set up!")
    else:
        click.echo("\n❌ End-to-end test failed. Please check the setup and try again.")

    # Delete the bubble_sort.py file after the test
    os.remove(bubble_sort_path)
    click.echo(f"🗑️ Deleted {bubble_sort_path}")


def collect_setup_info(setup_info: dict[str, str]):
    click.echo("\n⚡️ Let's set up CodeFlash for your project.")
    click.echo("Checking for pyproject.toml or setup.py ...")
    # Check for the existence of pyproject.toml or setup.py
    project_name = check_for_toml_or_setup_file()

    curdir = os.getcwd()
    subdirs = [d for d in next(os.walk("."))[1] if not d.startswith(".")]

    source_dir = click.prompt(
        "What's your project's source directory? I'll optimize all code in this directory.",
        type=click.Choice(
            [dir for dir in subdirs if dir != "tests"],
            case_sensitive=False,
        ),
        show_choices=True,
        default=project_name if project_name in subdirs else subdirs[0] if subdirs else ".",
    )
    setup_info["project_root"] = source_dir
    ph("cli-project-root-provided")

    setup_info["test_framework"] = click.prompt(
        "Which test framework do you use?",
        type=click.Choice(["pytest", "unittest"]),
        show_choices=True,
    )
    ph("cli-test-framework-provided", {"test_framework": setup_info["test_framework"]})

    # Discover test directory
    default_tests_subdir = "tests"
    if default_tests_subdir in subdirs:
        tests_root = click.prompt(
            "Where are your tests located?",
            default=os.path.join(curdir, default_tests_subdir),
        )
    else:
        tests_root = click.prompt(
            "Where are your tests located? If you don't have any tests yet, just press enter and I'll create an empty tests/ directory for you.",
            default="",
        )
        if tests_root == "":
            tests_root = os.path.join(curdir, default_tests_subdir)
            os.mkdir(tests_root)
    setup_info["tests_root"] = os.path.relpath(tests_root, curdir)
    ph("cli-tests-root-provided")

    # Ask for paths to ignore and update the setup_info dictionary
    # ignore_paths_input = click.prompt("Are there any paths CodeFlash should ignore? (comma-separated, no spaces)",
    #                                   default='', show_default=False)
    # ignore_paths = ignore_paths_input.split(',') if ignore_paths_input else ['tests/']
    ignore_paths = []
    setup_info["ignore_paths"] = ignore_paths


def check_for_toml_or_setup_file() -> Optional[str]:
    curdir = os.getcwd()
    pyproject_toml_path = os.path.join(curdir, "pyproject.toml")
    setup_py_path = os.path.join(curdir, "setup.py")
    project_name = None
    if os.path.exists(pyproject_toml_path):
        try:
            with open(pyproject_toml_path, "r") as f:
                pyproject_toml_content = f.read()
            project_name = tomlkit.parse(pyproject_toml_content)["tool"]["poetry"]["name"]
            click.echo(f"✅ Found a pyproject.toml for your project {project_name}")
            ph("cli-pyproject-toml-found-name")
        except Exception as e:
            click.echo(f"✅ Found a pyproject.toml.")
            ph("cli-pyproject-toml-found")
    elif os.path.exists(setup_py_path):
        with open(setup_py_path, "r") as f:
            setup_py_content = f.read()
        project_name_match = re.search(
            r"setup\s*\([^)]*?name\s*=\s*['\"](.*?)['\"]", setup_py_content, re.DOTALL
        )
        if project_name_match:
            project_name = project_name_match.group(1)
            click.echo(f"✅ Found setup.py for your project {project_name}")
            ph("cli-setup-py-found-name")
        else:
            click.echo(f"✅ Found setup.py.")
            ph("cli-setup-py-found")
        # Create a pyproject.toml file because it doesn't exist
        create_toml = (
            click.prompt(
                f"I need your project to have a pyproject.toml file to store CodeFlash configuration settings.\n"
                f"Do you want to run `poetry init` to create one?",
                default="y",
                type=click.STRING,
            )
            .lower()
            .strip()
        )
        if create_toml.startswith("y"):
            # Check if Poetry is installed, if not, install it using pip
            poetry_check = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
            if poetry_check.returncode != 0:
                click.echo("Poetry is not installed. Installing Poetry...")
                subprocess.run(["pip", "install", "poetry"], check=True)
            subprocess.run(["poetry", "init"], cwd=curdir)
            click.echo(f"✅ Created a pyproject.toml file at {pyproject_toml_path}")
            ph("cli-created-pyproject-toml")
    else:
        click.echo(
            f"❌ I couldn't find a pyproject.toml or a setup.py in the current directory ({curdir}).\n"
            "Please make sure you're running codeflash init from your project directory.\n"
            "See https://app.codeflash.ai/app/getting-started for more details!"
        )
        ph("cli-no-pyproject-toml-or-setup-py")
        sys.exit(1)
    return project_name


# Ask if the user wants CodeFlash to optimize new GitHub PRs
def prompt_github_action(setup_info: dict[str, str]):
    optimize_prs = (
        click.prompt(
            "Do you want CodeFlash to automatically optimize new Github PRs when they're opened (recommended)?",
            default="y",
            type=click.STRING,
        )
        .lower()
        .strip()
    )
    optimize_yes = optimize_prs.startswith("y")
    ph("cli-github-optimization-choice", {"optimize_prs": optimize_yes})
    if optimize_yes:
        repo = Repo(setup_info["project_root"], search_parent_directories=True)
        git_root = repo.git.rev_parse("--show-toplevel")
        workflows_path = os.path.join(git_root, ".github", "workflows")
        optimize_yml_path = os.path.join(workflows_path, "codeflash-optimize.yml")

        confirm_creation = (
            click.prompt(
                f"Great! We'll create a new workflow file [{optimize_yml_path}]. Is this OK?",
                default="y",
                type=click.STRING,
            )
            .lower()
            .strip()
        )
        confirm_creation_yes = confirm_creation.startswith("y")
        ph(
            "cli-github-optimization-confirm-workflow-creation",
            {"confirm_creation": confirm_creation_yes},
        )
        if confirm_creation_yes:
            os.makedirs(workflows_path, exist_ok=True)
            from importlib.resources import read_text

            optimize_yml_content = read_text(
                "codeflash.cli_cmds.workflows", "codeflash-optimize.yaml"
            )
            with open(optimize_yml_path, "w") as optimize_yml_file:
                optimize_yml_file.write(optimize_yml_content)
            click.echo(f"✅ Created {optimize_yml_path}")

            click.prompt(
                f"As a final step, you'll need to add your CODEFLASH_API_KEY as a secret to your GitHub repo.\n"
                + "Press enter to open your repo's secrets page, then "
                + "click 'New repository secret' and add your api key with the variable name CODEFLASH_API_KEY.",
                default="",
                type=click.STRING,
                prompt_suffix="",
                show_default=False,
            )
            click.launch(get_github_secrets_page_url(repo))
            ph("cli-github-workflow-created")
        else:
            click.echo("Skipping GitHub workflow creation.")
            ph("cli-github-workflow-skipped")


# Create or update the pyproject.toml file with the CodeFlash dependency & configuration
def configure_pyproject_toml(setup_info: dict[str, str]):
    toml_path = os.path.join(os.getcwd(), "pyproject.toml")
    try:
        with open(toml_path, "r") as pyproject_file:
            pyproject_data = tomlkit.parse(pyproject_file.read())
    except FileNotFoundError:
        click.echo(
            f"Could not find a pyproject.toml in the current directory.\n"
            f"Please create it by running `poetry init`, or run `codeflash init` again from a different project directory."
        )

    # Ensure the 'tool.poetry.dependencies' table exists
    poetry_dependencies = (
        pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", tomlkit.table())
    )

    # Update the 'pyproject_data' with the modified dependencies
    if "tool" not in pyproject_data:
        pyproject_data["tool"] = tomlkit.table()
    if "poetry" not in pyproject_data["tool"]:
        pyproject_data["tool"]["poetry"] = tomlkit.table()
    pyproject_data["tool"]["poetry"]["dependencies"] = poetry_dependencies
    codeflash_section = tomlkit.table()
    codeflash_section[
        "root"
    ] = "."  # Note we aren't using the project_root here, but the relative path to the root
    codeflash_section["test-root"] = setup_info["tests_root"]
    codeflash_section["test-framework"] = setup_info["test_framework"]
    codeflash_section["ignore-paths"] = setup_info["ignore_paths"]

    # Add the 'codeflash' section
    pyproject_data["tool"]["codeflash"] = codeflash_section
    click.echo(f"Writing CodeFlash configuration ...")
    with open(toml_path, "w") as pyproject_file:
        pyproject_file.write(tomlkit.dumps(pyproject_data))
    click.echo(f"✅ Added CodeFlash configuration to {toml_path}")


# Returns True if the user entered a new API key, False if they used an existing one
def prompt_api_key() -> bool:
    try:
        existing_api_key = get_codeflash_api_key()
    except EnvironmentError:
        existing_api_key = None
    if existing_api_key:
        display_key = f"{existing_api_key[:3]}****{existing_api_key[-4:]}"
        use_existing_key = click.prompt(
            f"I found a CODEFLASH_API_KEY in your environment [{display_key}]!\n"
            f"Press enter to use this key, or any other key to change it",
            default="",
            show_default=False,
        ).strip()
        if use_existing_key == "":
            ph("cli-existing-api-key-used")
            return False
        else:
            enter_api_key_and_save_to_rc(existing_api_key=use_existing_key)
            ph("cli-new-api-key-entered")
            return True
    else:
        enter_api_key_and_save_to_rc()
        ph("cli-new-api-key-entered")
        return True


def enter_api_key_and_save_to_rc(existing_api_key: str = ""):
    browser_launched = False
    api_key = existing_api_key
    while api_key == "":
        api_key = click.prompt(
            f"Enter your CodeFlash API key{' [or press Enter to open your API key page]' if not browser_launched else ''}",
            hide_input=False,
            default="",
            show_default=False,
        ).strip()
        if api_key:
            break
        else:
            if not browser_launched:
                click.echo(
                    "Opening your CodeFlash API key page. Grab a key from there!\n"
                    "You can also open this link manually: https://app.codeflash.ai/app/apikeys"
                )
                click.launch("https://app.codeflash.ai/app/apikeys")
                browser_launched = True  # This does not work on remote consoles
    shell_rc_path = os.path.expanduser(
        f"~/.{os.environ.get('SHELL', '/bin/bash').split('/')[-1]}rc"
    )
    api_key_line = f'export CODEFLASH_API_KEY="{api_key}"'
    api_key_pattern = re.compile(r'^export CODEFLASH_API_KEY=".*"$', re.M)
    with open(shell_rc_path, "r+") as shell_rc:
        shell_contents = shell_rc.read()
        if api_key_pattern.search(shell_contents):
            # Replace the existing API key line
            updated_shell_contents = api_key_pattern.sub(api_key_line, shell_contents)
        else:
            # Append the new API key line
            updated_shell_contents = shell_contents.rstrip() + f"\n{api_key_line}\n"
        shell_rc.seek(0)
        shell_rc.write(updated_shell_contents)
        shell_rc.truncate()
    click.echo(f"✅ Updated CODEFLASH_API_KEY in {shell_rc_path}")
    os.environ["CODEFLASH_API_KEY"] = api_key
