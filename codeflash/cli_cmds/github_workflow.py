from __future__ import annotations

import sys
from enum import Enum, auto
from pathlib import Path
from typing import Any

import click
import git
import inquirer
import tomlkit
from git import Repo
from rich.panel import Panel
from rich.text import Text

from codeflash.api.aiservice import AiServiceClient
from codeflash.api.cfapi import setup_github_actions
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, logger
from codeflash.cli_cmds.init_config import CodeflashTheme
from codeflash.code_utils.compat import LF
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.code_utils.git_utils import get_current_branch, get_repo_owner_and_name
from codeflash.code_utils.github_utils import get_github_secrets_page_url
from codeflash.telemetry.posthog_cf import ph

_POETRY_COMMANDS: str = """|
          python -m pip install --upgrade pip
          pip install poetry
          poetry install --all-extras"""

_UV_COMMANDS: str = """|
          uv sync --all-extras
          uv pip install --upgrade codeflash"""

_DEFAULT_COMMANDS: str = """|
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install codeflash"""


class DependencyManager(Enum):
    """Python dependency managers."""

    PIP = auto()
    POETRY = auto()
    UV = auto()
    UNKNOWN = auto()


def install_github_actions(override_formatter_check: bool = False) -> None:
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

        # Check if workflow file already exists locally BEFORE showing prompt
        if optimize_yaml_path.exists():
            # Workflow file already exists locally - skip prompt and setup
            already_exists_message = "✅ GitHub Actions workflow file already exists.\n\n"
            already_exists_message += "No changes needed - your repository is already configured!"

            already_exists_panel = Panel(
                Text(already_exists_message, style="green", justify="center"),
                title="✅ Already Configured",
                border_style="bright_green",
            )
            console.print(already_exists_panel)
            console.print()

            logger.info(
                "[github_workflow.py:install_github_actions] Workflow file already exists locally, skipping setup"
            )
            return

        # Get repository information for API call
        git_remote = config.get("git_remote", "origin")
        # get_current_branch handles detached HEAD and other edge cases internally
        try:
            base_branch = get_current_branch(repo)
        except Exception as e:
            logger.warning(
                f"[github_workflow.py:install_github_actions] Could not determine current branch: {e}. Falling back to 'main'."
            )
            base_branch = "main"

        # Generate workflow content
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

        # Show prompt only if workflow doesn't exist locally
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

        creation_questions = [
            inquirer.Confirm(
                "confirm_creation",
                message="Set up GitHub Actions for continuous optimization? We'll open a pull request with the workflow file.",
                default=True,
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

        # Generate workflow content AFTER user confirmation
        logger.info("[github_workflow.py:install_github_actions] User confirmed, generating workflow content...")

        # Select the appropriate workflow template based on project language
        project_language = detect_project_language_for_workflow(Path.cwd())
        if project_language in ("javascript", "typescript"):
            workflow_template = "codeflash-optimize-js.yaml"
        else:
            workflow_template = "codeflash-optimize.yaml"

        optimize_yml_content = (
            files("codeflash").joinpath("cli_cmds", "workflows", workflow_template).read_text(encoding="utf-8")
        )
        materialized_optimize_yml_content = generate_dynamic_workflow_content(
            optimize_yml_content, config, git_root, benchmark_mode
        )

        workflows_path.mkdir(parents=True, exist_ok=True)

        pr_created_via_api = False
        pr_url = None

        try:
            owner, repo_name = get_repo_owner_and_name(repo, git_remote)
        except Exception as e:
            logger.error(f"[github_workflow.py:install_github_actions] Failed to get repository owner and name: {e}")
            # Fall back to local file creation
            workflows_path.mkdir(parents=True, exist_ok=True)
            with optimize_yaml_path.open("w", encoding="utf8") as optimize_yml_file:
                optimize_yml_file.write(materialized_optimize_yml_content)
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
        else:
            # Try to create PR via API
            try:
                # Workflow file doesn't exist on remote or content differs - proceed with PR creation
                console.print("Creating PR with GitHub Actions workflow...")
                logger.info(
                    f"[github_workflow.py:install_github_actions] Calling setup_github_actions API for {owner}/{repo_name} on branch {base_branch}"
                )

                response = setup_github_actions(
                    owner=owner,
                    repo=repo_name,
                    base_branch=base_branch,
                    workflow_content=materialized_optimize_yml_content,
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("success"):
                        pr_url = response_data.get("pr_url")

                        if pr_url:
                            pr_created_via_api = True
                            success_message = f"✅ PR created: {pr_url}\n\n"
                            success_message += "Your repository is now configured for continuous optimization!"

                            workflow_success_panel = Panel(
                                Text(success_message, style="green", justify="center"),
                                title="🎉 Workflow PR Created!",
                                border_style="bright_green",
                            )
                            console.print(workflow_success_panel)
                            console.print()

                            logger.info(
                                f"[github_workflow.py:install_github_actions] Successfully created PR #{response_data.get('pr_number')} for {owner}/{repo_name}"
                            )
                        else:
                            # File already exists with same content
                            pr_created_via_api = True  # Mark as handled (no PR needed)
                            already_exists_message = "✅ Workflow file already exists with the same content.\n\n"
                            already_exists_message += "No changes needed - your repository is already configured!"

                            already_exists_panel = Panel(
                                Text(already_exists_message, style="green", justify="center"),
                                title="✅ Already Configured",
                                border_style="bright_green",
                            )
                            console.print(already_exists_panel)
                            console.print()
                    else:
                        # API returned success=false, extract error details
                        error_data = response_data
                        error_msg = error_data.get("error", "Unknown error")
                        error_message = error_data.get("message", error_msg)
                        error_help = error_data.get("help", "")
                        installation_url = error_data.get("installation_url")

                        # For permission errors, don't fall back - show a focused message and abort early
                        if response.status_code == 403:
                            logger.error(
                                f"[github_workflow.py:install_github_actions] Permission denied for {owner}/{repo_name}"
                            )
                            # Extract installation_url if available, otherwise use default
                            installation_url_403 = error_data.get(
                                "installation_url", "https://github.com/apps/codeflash-ai/installations/select_target"
                            )

                            permission_error_panel = Panel(
                                Text(
                                    "❌ Access Denied\n\n"
                                    f"The GitHub App may not be installed on {owner}/{repo_name}, or it doesn't have the required permissions.\n\n"
                                    "💡 To fix this:\n"
                                    "1. Install the CodeFlash GitHub App on your repository\n"
                                    "2. Ensure the app has 'Contents: write', 'Workflows: write', and 'Pull requests: write' permissions\n"
                                    "3. Make sure you have write access to the repository\n\n"
                                    f"🔗 Install GitHub App: {installation_url_403}",
                                    style="red",
                                ),
                                title="❌ Setup Failed",
                                border_style="red",
                            )
                            console.print(permission_error_panel)
                            console.print()
                            click.echo(
                                f"Please install the CodeFlash GitHub App and ensure it has the required permissions.{LF}"
                                f"Visit: {installation_url_403}{LF}"
                            )
                            apologize_and_exit()

                        # Show detailed error panel for all other errors
                        error_panel_text = f"❌ {error_msg}\n\n{error_message}\n"
                        if error_help:
                            error_panel_text += f"\n💡 {error_help}\n"
                        if installation_url:
                            error_panel_text += f"\n🔗 Install GitHub App: {installation_url}"

                        error_panel = Panel(
                            Text(error_panel_text, style="red"), title="❌ Setup Failed", border_style="red"
                        )
                        console.print(error_panel)
                        console.print()

                        # For GitHub App not installed, don't fall back - show clear instructions
                        if response.status_code == 404 and installation_url:
                            logger.error(
                                f"[github_workflow.py:install_github_actions] GitHub App not installed on {owner}/{repo_name}"
                            )
                            click.echo(
                                f"Please install the CodeFlash GitHub App on your repository to continue.{LF}"
                                f"Visit: {installation_url}{LF}"
                            )
                            return

                        # For other errors, fall back to local file creation
                        raise Exception(error_message)  # noqa: TRY002, TRY301
                else:
                    # API call returned non-200 status, try to parse error response
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "API request failed")
                        error_message = error_data.get("message", f"API returned status {response.status_code}")
                        error_help = error_data.get("help", "")
                        installation_url = error_data.get("installation_url")

                        # For permission errors, don't fall back - show a focused message and abort early
                        if response.status_code == 403:
                            logger.error(
                                f"[github_workflow.py:install_github_actions] Permission denied for {owner}/{repo_name}"
                            )
                            # Extract installation_url if available, otherwise use default
                            installation_url_403 = error_data.get(
                                "installation_url", "https://github.com/apps/codeflash-ai/installations/select_target"
                            )

                            permission_error_panel = Panel(
                                Text(
                                    "❌ Access Denied\n\n"
                                    f"The GitHub App may not be installed on {owner}/{repo_name}, or it doesn't have the required permissions.\n\n"
                                    "💡 To fix this:\n"
                                    "1. Install the CodeFlash GitHub App on your repository\n"
                                    "2. Ensure the app has 'Contents: write', 'Workflows: write', and 'Pull requests: write' permissions\n"
                                    "3. Make sure you have write access to the repository\n\n"
                                    f"🔗 Install GitHub App: {installation_url_403}",
                                    style="red",
                                ),
                                title="❌ Setup Failed",
                                border_style="red",
                            )
                            console.print(permission_error_panel)
                            console.print()
                            click.echo(
                                f"Please install the CodeFlash GitHub App and ensure it has the required permissions.{LF}"
                                f"Visit: {installation_url_403}{LF}"
                            )
                            apologize_and_exit()

                        # Show detailed error panel for all other errors
                        error_panel_text = f"❌ {error_msg}\n\n{error_message}\n"
                        if error_help:
                            error_panel_text += f"\n💡 {error_help}\n"
                        if installation_url:
                            error_panel_text += f"\n🔗 Install GitHub App: {installation_url}"

                        error_panel = Panel(
                            Text(error_panel_text, style="red"), title="❌ Setup Failed", border_style="red"
                        )
                        console.print(error_panel)
                        console.print()

                        # For GitHub App not installed, don't fall back - show clear instructions
                        if response.status_code == 404 and installation_url:
                            logger.error(
                                f"[github_workflow.py:install_github_actions] GitHub App not installed on {owner}/{repo_name}"
                            )
                            click.echo(
                                f"Please install the CodeFlash GitHub App on your repository to continue.{LF}"
                                f"Visit: {installation_url}{LF}"
                            )
                            return

                        # For authentication errors, don't fall back
                        if response.status_code == 401:
                            logger.error(
                                f"[github_workflow.py:install_github_actions] Authentication failed for {owner}/{repo_name}"
                            )
                            click.echo(f"Authentication failed. Please check your API key and try again.{LF}")
                            return

                        # For other errors, fall back to local file creation
                        raise Exception(error_message)  # noqa: TRY002
                    except (ValueError, KeyError) as parse_error:
                        # Couldn't parse error response, use generic message
                        status_msg = f"API returned status {response.status_code}"
                        raise Exception(status_msg) from parse_error  # noqa: TRY002

            except Exception as api_error:
                # Fall back to local file creation if API call fails (for non-critical errors)
                logger.warning(
                    f"[github_workflow.py:install_github_actions] API call failed, falling back to local file creation: {api_error}"
                )
                workflows_path.mkdir(parents=True, exist_ok=True)
                with optimize_yaml_path.open("w", encoding="utf8") as optimize_yml_file:
                    optimize_yml_file.write(materialized_optimize_yml_content)
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

        # Show appropriate message based on whether PR was created via API
        if pr_created_via_api:
            if pr_url:
                click.echo(
                    f"🚀 Codeflash is now configured to automatically optimize new Github PRs!{LF}"
                    f"Once you merge the PR, the workflow will be active.{LF}"
                )
            else:
                # File already exists
                click.echo(
                    f"🚀 Codeflash is now configured to automatically optimize new Github PRs!{LF}"
                    f"The workflow is ready to use.{LF}"
                )
        else:
            # Fell back to local file creation
            click.echo(
                f"Please edit, commit and push this GitHub actions file to your repo, and you're all set!{LF}"
                f"🚀 Codeflash is now configured to automatically optimize new Github PRs!{LF}"
            )

        # Show GitHub secrets setup panel (needed in both cases - PR created via API or local file)
        try:
            existing_api_key = get_codeflash_api_key()
        except OSError:
            existing_api_key = None

        # GitHub secrets setup panel - always shown since secrets are required for the workflow to work
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
        console.print()
        ph("cli-github-workflow-created")
    except KeyboardInterrupt:
        apologize_and_exit()


def determine_dependency_manager(pyproject_data: dict[str, Any]) -> DependencyManager:
    """Determine which dependency manager is being used based on pyproject.toml contents."""
    cwd = Path.cwd()
    if (cwd / "poetry.lock").exists():
        return DependencyManager.POETRY
    if (cwd / "uv.lock").exists():
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
    return _DEP_MANAGER_TO_COMMANDS.get(dep_manager, _DEFAULT_COMMANDS)


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


def detect_project_language_for_workflow(project_root: Path) -> str:
    """Detect the primary language of the project for workflow generation.

    Returns: 'python', 'javascript', or 'typescript'
    """
    # Check for TypeScript config
    if (project_root / "tsconfig.json").exists():
        return "typescript"

    # Check for JavaScript/TypeScript indicators
    has_package_json = (project_root / "package.json").exists()
    has_pyproject = (project_root / "pyproject.toml").exists()

    if has_package_json and not has_pyproject:
        # Pure JS/TS project
        return "javascript"
    if has_pyproject and not has_package_json:
        # Pure Python project
        return "python"

    # Both exist - count files to determine primary language
    js_count = 0
    py_count = 0
    for file in project_root.rglob("*"):
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                js_count += 1
            elif suffix == ".py":
                py_count += 1

    if js_count > py_count:
        return "javascript"
    return "python"


def collect_repo_files_for_workflow(git_root: Path) -> dict[str, Any]:
    """Collect important repository files and directory structure for workflow generation.

    :param git_root: Root directory of the git repository
    :return: Dictionary with 'files' (path -> content) and 'directory_structure' (nested dict)
    """
    # Important files to collect with contents
    important_files = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements/requirements.txt",
        "requirements/dev.txt",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "uv.lock",
        "setup.py",
        "setup.cfg",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "Makefile",
        "README.md",
        "README.rst",
    ]

    # Also collect GitHub workflows
    workflows_path = git_root / ".github" / "workflows"
    if workflows_path.exists():
        important_files.extend(
            str(workflow_file.relative_to(git_root)) for workflow_file in workflows_path.glob("*.yml")
        )
        important_files.extend(
            str(workflow_file.relative_to(git_root)) for workflow_file in workflows_path.glob("*.yaml")
        )

    files_dict: dict[str, str] = {}
    max_file_size = 8 * 1024  # 8KB limit per file

    for file_path_str in important_files:
        file_path = git_root / file_path_str
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                # Limit file size
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... (truncated)"
                files_dict[file_path_str] = content
            except Exception as e:
                logger.warning(
                    f"[github_workflow.py:collect_repo_files_for_workflow] Failed to read {file_path_str}: {e}"
                )

    # Collect 2-level directory structure
    directory_structure: dict[str, Any] = {}
    try:
        for item in sorted(git_root.iterdir()):
            if item.name.startswith(".") and item.name not in [".github", ".git"]:
                continue  # Skip hidden files/folders except .github

            if item.is_dir():
                # Level 1: directory
                dir_dict: dict[str, Any] = {"type": "directory", "contents": {}}
                try:
                    # Level 2: contents of directory
                    for subitem in sorted(item.iterdir()):
                        if subitem.name.startswith("."):
                            continue
                        if subitem.is_dir():
                            dir_dict["contents"][subitem.name] = {"type": "directory"}
                        else:
                            dir_dict["contents"][subitem.name] = {"type": "file"}
                except PermissionError:
                    pass  # Skip directories we can't read
                directory_structure[item.name] = dir_dict
            elif item.is_file():
                directory_structure[item.name] = {"type": "file"}
    except Exception as e:
        logger.warning(
            f"[github_workflow.py:collect_repo_files_for_workflow] Error collecting directory structure: {e}"
        )

    return {"files": files_dict, "directory_structure": directory_structure}


def generate_dynamic_workflow_content(
    optimize_yml_content: str, config: dict[str, Any], git_root: Path, benchmark_mode: bool = False
) -> str:
    """Generate workflow content with dynamic steps from AI service, falling back to static template."""
    # First, do the basic replacements that are always needed
    module_path = str(Path(config["module_root"]).relative_to(git_root) / "**")
    optimize_yml_content = optimize_yml_content.replace("{{ codeflash_module_path }}", module_path)

    # Detect project language
    project_language = detect_project_language_for_workflow(Path.cwd())

    # For JavaScript/TypeScript projects, use static template customization
    # (AI-generated steps are currently Python-only)
    if project_language in ("javascript", "typescript"):
        return customize_codeflash_yaml_content(optimize_yml_content, config, git_root, benchmark_mode)

    # Python project - try AI-generated steps
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

    # Try to generate dynamic steps using AI service
    try:
        repo_data = collect_repo_files_for_workflow(git_root)

        # Prepare codeflash config for AI
        codeflash_config = {
            "module_root": config["module_root"],
            "tests_root": config.get("tests_root", ""),
            "benchmark_mode": benchmark_mode,
        }

        aiservice_client = AiServiceClient()
        dynamic_steps = aiservice_client.generate_workflow_steps(
            repo_files=repo_data["files"],
            directory_structure=repo_data["directory_structure"],
            codeflash_config=codeflash_config,
        )

        if dynamic_steps:
            # Replace the entire steps section with AI-generated steps
            # Find the steps section in the template
            steps_start = optimize_yml_content.find("    steps:")
            if steps_start != -1:
                # Find the end of the steps section (next line at same or less indentation)
                lines = optimize_yml_content.split("\n")
                steps_start_line = optimize_yml_content[:steps_start].count("\n")
                steps_end_line = len(lines)

                # Find where steps section ends (next job or end of file)
                for i in range(steps_start_line + 1, len(lines)):
                    line = lines[i]
                    # Stop if we hit a line that's not indented (new job or end of jobs)
                    if line and not line.startswith(" ") and not line.startswith("\t"):
                        steps_end_line = i
                        break

                # Extract steps content from AI response (remove "steps:" prefix if present)
                steps_content = dynamic_steps
                if steps_content.startswith("steps:"):
                    # Remove "steps:" and leading newline
                    steps_content = steps_content[6:].lstrip("\n")

                # Ensure proper indentation (8 spaces for steps section in YAML)
                indented_steps = []
                for line in steps_content.split("\n"):
                    if line.strip():
                        # If line doesn't start with enough spaces, add them
                        if not line.startswith(" "):
                            indented_steps.append("        " + line)
                        else:
                            # Preserve existing indentation but ensure minimum 8 spaces
                            current_indent = len(line) - len(line.lstrip())
                            if current_indent < 8:
                                indented_steps.append(" " * 8 + line.lstrip())
                            else:
                                indented_steps.append(line)
                    else:
                        indented_steps.append("")

                # Add codeflash command step at the end
                dep_manager = determine_dependency_manager(pyproject_data)
                codeflash_cmd = get_codeflash_github_action_command(dep_manager)
                if benchmark_mode:
                    codeflash_cmd += " --benchmark"

                # Format codeflash command properly
                if "|" in codeflash_cmd:
                    # Multi-line command
                    cmd_lines = codeflash_cmd.split("\n")
                    codeflash_step = f"      - name: ⚡️Codeflash Optimization\n        run: {cmd_lines[0].strip()}"
                    for cmd_line in cmd_lines[1:]:
                        codeflash_step += f"\n          {cmd_line.strip()}"
                else:
                    codeflash_step = f"      - name: ⚡️Codeflash Optimization\n        run: {codeflash_cmd}"

                indented_steps.append(codeflash_step)

                # Reconstruct the workflow
                return "\n".join([*lines[:steps_start_line], "    steps:", *indented_steps, *lines[steps_end_line:]])
            logger.warning(
                "[github_workflow.py:generate_dynamic_workflow_content] Could not find steps section in template"
            )
        else:
            logger.debug(
                "[github_workflow.py:generate_dynamic_workflow_content] AI service returned no steps, falling back to static"
            )

    except Exception as e:
        logger.warning(
            f"[github_workflow.py:generate_dynamic_workflow_content] Error generating dynamic workflow, falling back to static: {e}"
        )

    # Fallback to static template
    return customize_codeflash_yaml_content(optimize_yml_content, config, git_root, benchmark_mode)


def customize_codeflash_yaml_content(
    optimize_yml_content: str, config: dict[str, Any], git_root: Path, benchmark_mode: bool = False
) -> str:
    module_path = str(Path(config["module_root"]).relative_to(git_root) / "**")
    optimize_yml_content = optimize_yml_content.replace("{{ codeflash_module_path }}", module_path)

    # Detect project language
    project_language = detect_project_language_for_workflow(Path.cwd())

    if project_language in ("javascript", "typescript"):
        # JavaScript/TypeScript project
        return _customize_js_workflow_content(optimize_yml_content, git_root, benchmark_mode)

    # Python project (default)
    return _customize_python_workflow_content(optimize_yml_content, git_root, benchmark_mode)


def _customize_python_workflow_content(optimize_yml_content: str, git_root: Path, benchmark_mode: bool = False) -> str:
    """Customize workflow content for Python projects."""
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
        "{{ setup_runtime_environment }}", python_depmanager_installation
    )
    install_deps_cmd = get_dependency_installation_commands(dep_manager)

    optimize_yml_content = optimize_yml_content.replace("{{ install_dependencies_command }}", install_deps_cmd)

    # Add codeflash command
    codeflash_cmd = get_codeflash_github_action_command(dep_manager)

    if benchmark_mode:
        codeflash_cmd += " --benchmark"
    return optimize_yml_content.replace("{{ codeflash_command }}", codeflash_cmd)


def _customize_js_workflow_content(optimize_yml_content: str, git_root: Path, benchmark_mode: bool = False) -> str:
    """Customize workflow content for JavaScript/TypeScript projects."""
    from codeflash.cli_cmds.init_javascript import (
        determine_js_package_manager,
        get_js_codeflash_install_step,
        get_js_codeflash_run_command,
        get_js_dependency_installation_commands,
        get_js_runtime_setup_steps,
        is_codeflash_dependency,
    )

    project_root = Path.cwd()
    package_json_path = project_root / "package.json"

    if not package_json_path.exists():
        click.echo(
            f"I couldn't find a package.json in the current directory.{LF}"
            f"Please run `npm init` or create a package.json file first."
        )
        apologize_and_exit()

    # Determine working directory relative to git root
    if project_root == git_root:
        working_dir = ""
    else:
        rel_path = str(project_root.relative_to(git_root))
        working_dir = f"""defaults:
      run:
        working-directory: ./{rel_path}"""

    optimize_yml_content = optimize_yml_content.replace("{{ working_directory }}", working_dir)

    # Determine package manager and codeflash dependency status
    pkg_manager = determine_js_package_manager(project_root)
    codeflash_is_dep = is_codeflash_dependency(project_root)

    # Setup runtime environment (Node.js/Bun)
    runtime_setup = get_js_runtime_setup_steps(pkg_manager)
    optimize_yml_content = optimize_yml_content.replace("{{ setup_runtime_steps }}", runtime_setup)

    # Install dependencies
    install_deps_cmd = get_js_dependency_installation_commands(pkg_manager)
    optimize_yml_content = optimize_yml_content.replace("{{ install_dependencies_command }}", install_deps_cmd)

    # Install codeflash step (only if not a dependency)
    install_codeflash = get_js_codeflash_install_step(pkg_manager, is_dependency=codeflash_is_dep)
    optimize_yml_content = optimize_yml_content.replace("{{ install_codeflash_step }}", install_codeflash)

    # Codeflash run command
    codeflash_cmd = get_js_codeflash_run_command(pkg_manager, is_dependency=codeflash_is_dep)
    if benchmark_mode:
        codeflash_cmd += " --benchmark"
    return optimize_yml_content.replace("{{ codeflash_command }}", codeflash_cmd)

_DEP_MANAGER_TO_COMMANDS = {
    DependencyManager.POETRY: _POETRY_COMMANDS,
    DependencyManager.UV: _UV_COMMANDS,
}
