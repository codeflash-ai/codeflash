"""Java project initialization for Codeflash."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Union

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


class JavaBuildTool(Enum):
    """Java build tools."""

    MAVEN = auto()
    GRADLE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class JavaSetupInfo:
    """Setup info for Java projects.

    Only stores values that override auto-detection or user preferences.
    Most config is auto-detected from pom.xml/build.gradle and project structure.
    """

    # Override values (None means use auto-detected value)
    module_root_override: Union[str, None] = None
    test_root_override: Union[str, None] = None
    formatter_override: Union[list[str], None] = None

    # User preferences (stored in config only if non-default)
    git_remote: str = "origin"
    disable_telemetry: bool = False
    ignore_paths: list[str] | None = None
    benchmarks_root: Union[str, None] = None


def _get_theme():
    """Get the CodeflashTheme - imported lazily to avoid circular imports."""
    from codeflash.cli_cmds.cmd_init import CodeflashTheme

    return CodeflashTheme()


def detect_java_build_tool(project_root: Path) -> JavaBuildTool:
    """Detect which Java build tool is being used."""
    if (project_root / "pom.xml").exists():
        return JavaBuildTool.MAVEN
    if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
        return JavaBuildTool.GRADLE
    return JavaBuildTool.UNKNOWN


def detect_java_source_root(project_root: Path) -> str:
    """Detect the Java source root directory."""
    # Standard Maven/Gradle layout
    standard_src = project_root / "src" / "main" / "java"
    if standard_src.is_dir():
        return "src/main/java"

    # Try to detect from pom.xml
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            # Handle Maven namespace
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}
            source_dir = root.find(".//m:sourceDirectory", ns)
            if source_dir is not None and source_dir.text:
                return source_dir.text
        except ET.ParseError:
            pass

    # Fallback to src directory
    if (project_root / "src").is_dir():
        return "src"

    return "."


def detect_java_test_root(project_root: Path) -> str:
    """Detect the Java test root directory."""
    # Standard Maven/Gradle layout
    standard_test = project_root / "src" / "test" / "java"
    if standard_test.is_dir():
        return "src/test/java"

    # Try to detect from pom.xml
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}
            test_source_dir = root.find(".//m:testSourceDirectory", ns)
            if test_source_dir is not None and test_source_dir.text:
                return test_source_dir.text
        except ET.ParseError:
            pass

    # Fallback patterns
    if (project_root / "test").is_dir():
        return "test"
    if (project_root / "tests").is_dir():
        return "tests"

    return "src/test/java"


def detect_java_test_framework(project_root: Path) -> str:
    """Detect the Java test framework in use."""
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            content = pom_path.read_text(encoding="utf-8")
            if "junit-jupiter" in content or "junit.jupiter" in content:
                return "junit5"
            if "junit" in content.lower():
                return "junit4"
            if "testng" in content.lower():
                return "testng"
        except Exception:
            pass

    gradle_file = project_root / "build.gradle"
    if gradle_file.exists():
        try:
            content = gradle_file.read_text(encoding="utf-8")
            if "junit-jupiter" in content or "useJUnitPlatform" in content:
                return "junit5"
            if "junit" in content.lower():
                return "junit4"
            if "testng" in content.lower():
                return "testng"
        except Exception:
            pass

    return "junit5"  # Default to JUnit 5


def init_java_project() -> None:
    """Initialize Codeflash for a Java project."""
    from codeflash.cli_cmds.cmd_init import install_github_actions, install_github_app, prompt_api_key

    lang_panel = Panel(
        Text(
            "Java project detected!\n\nI'll help you set up Codeflash for your project.", style="cyan", justify="center"
        ),
        title="Java Setup",
        border_style="bright_red",
    )
    console.print(lang_panel)
    console.print()

    did_add_new_key = prompt_api_key()

    should_modify, _config = should_modify_java_config()

    # Default git remote
    git_remote = "origin"

    if should_modify:
        setup_info = collect_java_setup_info()
        git_remote = setup_info.git_remote or "origin"
        configured = configure_java_project(setup_info)
        if not configured:
            apologize_and_exit()

    install_github_app(git_remote)

    install_github_actions(override_formatter_check=True)

    # Show completion message
    usage_table = Table(show_header=False, show_lines=False, border_style="dim")
    usage_table.add_column("Command", style="cyan")
    usage_table.add_column("Description", style="white")

    usage_table.add_row("codeflash --file <path-to-file> --function <function-name>", "Optimize a specific function")
    usage_table.add_row("codeflash --all", "Optimize all functions in all files")
    usage_table.add_row("codeflash --help", "See all available options")

    completion_message = "Codeflash is now set up for your Java project!\n\nYou can now run any of these commands:"

    if did_add_new_key:
        completion_message += (
            "\n\nDon't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!"
        )
        if os.name == "nt":
            reload_cmd = f". {get_shell_rc_path()}" if is_powershell() else f"call {get_shell_rc_path()}"
        else:
            reload_cmd = f"source {get_shell_rc_path()}"
        completion_message += f"\nOr run: {reload_cmd}"

    completion_panel = Panel(
        Group(Text(completion_message, style="bold green"), Text(""), usage_table),
        title="Setup Complete!",
        border_style="bright_green",
        padding=(1, 2),
    )
    console.print(completion_panel)

    ph("cli-java-installation-successful", {"did_add_new_key": did_add_new_key})
    sys.exit(0)


def should_modify_java_config() -> tuple[bool, dict[str, Any] | None]:
    """Check if the project already has Codeflash config."""
    from rich.prompt import Confirm

    project_root = Path.cwd()

    # Check for existing codeflash config in pom.xml or a separate config file
    codeflash_config_path = project_root / "codeflash.toml"
    if codeflash_config_path.exists():
        return Confirm.ask(
            "A Codeflash config already exists. Do you want to re-configure it?", default=False, show_default=True
        ), None

    return True, None


def collect_java_setup_info() -> JavaSetupInfo:
    """Collect setup information for Java projects."""
    from rich.prompt import Confirm

    from codeflash.cli_cmds.cmd_init import ask_for_telemetry

    curdir = Path.cwd()

    if not os.access(curdir, os.W_OK):
        click.echo(f"The current directory isn't writable, please check your folder permissions and try again.{LF}")
        sys.exit(1)

    # Auto-detect values
    build_tool = detect_java_build_tool(curdir)
    detected_source_root = detect_java_source_root(curdir)
    detected_test_root = detect_java_test_root(curdir)
    detected_test_framework = detect_java_test_framework(curdir)

    # Build detection summary
    build_tool_name = build_tool.name.lower() if build_tool != JavaBuildTool.UNKNOWN else "unknown"
    detection_table = Table(show_header=False, box=None, padding=(0, 2))
    detection_table.add_column("Setting", style="cyan")
    detection_table.add_column("Value", style="green")
    detection_table.add_row("Build tool", build_tool_name)
    detection_table.add_row("Source root", detected_source_root)
    detection_table.add_row("Test root", detected_test_root)
    detection_table.add_row("Test framework", detected_test_framework)

    detection_panel = Panel(
        Group(Text("Auto-detected settings for your Java project:\n", style="cyan"), detection_table),
        title="Auto-Detection Results",
        border_style="bright_blue",
    )
    console.print(detection_panel)
    console.print()

    # Ask if user wants to change any settings
    module_root_override = None
    test_root_override = None
    formatter_override = None

    if Confirm.ask("Would you like to change any of these settings?", default=False):
        # Source root override
        module_root_override = _prompt_directory_override("source", detected_source_root, curdir)

        # Test root override
        test_root_override = _prompt_directory_override("test", detected_test_root, curdir)

        # Formatter override
        formatter_questions = [
            inquirer.List(
                "formatter",
                message="Which code formatter do you use?",
                choices=[
                    ("keep detected (google-java-format)", "keep"),
                    ("google-java-format", "google-java-format"),
                    ("spotless", "spotless"),
                    ("other", "other"),
                    ("don't use a formatter", "disabled"),
                ],
                default="keep",
                carousel=True,
            )
        ]

        formatter_answers = inquirer.prompt(formatter_questions, theme=_get_theme())
        if not formatter_answers:
            apologize_and_exit()

        formatter_choice = formatter_answers["formatter"]
        if formatter_choice != "keep":
            formatter_override = get_java_formatter_cmd(formatter_choice, build_tool)

        ph("cli-java-formatter-provided", {"overridden": formatter_override is not None})

    # Git remote
    git_remote = _get_git_remote_for_setup()

    # Telemetry
    disable_telemetry = not ask_for_telemetry()

    return JavaSetupInfo(
        module_root_override=module_root_override,
        test_root_override=test_root_override,
        formatter_override=formatter_override,
        git_remote=git_remote,
        disable_telemetry=disable_telemetry,
    )


def _prompt_directory_override(dir_type: str, detected: str, curdir: Path) -> str | None:
    """Prompt for a directory override."""
    keep_detected_option = f"keep detected ({detected})"
    custom_dir_option = "enter a custom directory..."

    # Get subdirectories that might be relevant
    subdirs = [d.name for d in curdir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    subdirs = [d for d in subdirs if d not in ("target", "build", ".git", ".idea", detected)]

    options = [keep_detected_option, *subdirs[:5], custom_dir_option]

    questions = [
        inquirer.List(
            f"{dir_type}_root",
            message=f"Which directory contains your {dir_type} code?",
            choices=options,
            default=keep_detected_option,
            carousel=True,
        )
    ]

    answers = inquirer.prompt(questions, theme=_get_theme())
    if not answers:
        apologize_and_exit()

    answer = answers[f"{dir_type}_root"]
    if answer == keep_detected_option:
        return None
    if answer == custom_dir_option:
        return _prompt_custom_directory(dir_type)
    return answer


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

        click.echo(f"Invalid path: {error_msg}")
        click.echo("Please enter a valid relative directory path.")
        console.print()


def _get_git_remote_for_setup() -> str:
    """Get git remote for project setup."""
    try:
        repo = Repo(Path.cwd(), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if not git_remotes:
            return ""

        if len(git_remotes) == 1:
            return git_remotes[0]

        git_panel = Panel(
            Text(
                "Configure Git Remote for Pull Requests.\n\nCodeflash will use this remote to create pull requests.",
                style="blue",
            ),
            title="Git Remote Setup",
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


def get_java_formatter_cmd(formatter: str, build_tool: JavaBuildTool) -> list[str]:
    """Get formatter commands for Java."""
    if formatter == "google-java-format":
        return ["google-java-format --replace $file"]
    if formatter == "spotless":
        return _SPOTLESS_COMMANDS.get(build_tool, ["spotless $file"])
    if formatter == "other":
        if not hasattr(get_java_formatter_cmd, '_warning_shown'):
            click.echo("In codeflash.toml, please replace 'your-formatter' with your formatter command.")
            get_java_formatter_cmd._warning_shown = True
        return ["your-formatter $file"]
    return ["disabled"]


def configure_java_project(setup_info: JavaSetupInfo) -> bool:
    """Configure codeflash.toml for Java projects."""
    import tomlkit

    codeflash_config_path = Path.cwd() / "codeflash.toml"

    # Build config
    config: dict[str, Any] = {}

    # Detect values
    curdir = Path.cwd()
    source_root = setup_info.module_root_override or detect_java_source_root(curdir)
    test_root = setup_info.test_root_override or detect_java_test_root(curdir)

    config["module-root"] = source_root
    config["tests-root"] = test_root

    # Formatter
    if setup_info.formatter_override is not None:
        if setup_info.formatter_override != ["disabled"]:
            config["formatter-cmds"] = setup_info.formatter_override
        else:
            config["formatter-cmds"] = []

    # Git remote
    if setup_info.git_remote and setup_info.git_remote not in ("", "origin"):
        config["git-remote"] = setup_info.git_remote

    # User preferences
    if setup_info.disable_telemetry:
        config["disable-telemetry"] = True

    if setup_info.ignore_paths:
        config["ignore-paths"] = setup_info.ignore_paths

    if setup_info.benchmarks_root:
        config["benchmarks-root"] = setup_info.benchmarks_root

    try:
        # Create TOML document
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Codeflash configuration for Java project"))
        doc.add(tomlkit.nl())

        codeflash_table = tomlkit.table()
        for key, value in config.items():
            codeflash_table.add(key, value)

        doc.add("tool", tomlkit.table())
        doc["tool"]["codeflash"] = codeflash_table

        with codeflash_config_path.open("w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        click.echo(f"Created Codeflash configuration in {codeflash_config_path}")
        click.echo()
        return True
    except OSError as e:
        click.echo(f"Failed to create codeflash.toml: {e}")
        return False


# ============================================================================
# GitHub Actions Workflow Helpers for Java
# ============================================================================


def get_java_runtime_setup_steps(build_tool: JavaBuildTool) -> str:
    """Generate the appropriate Java setup steps for GitHub Actions."""
    java_setup = """- name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'"""

    if build_tool == JavaBuildTool.MAVEN:
        java_setup += """
          cache: 'maven'"""
    elif build_tool == JavaBuildTool.GRADLE:
        java_setup += """
          cache: 'gradle'"""

    return java_setup


def get_java_dependency_installation_commands(build_tool: JavaBuildTool) -> str:
    """Generate commands to install Java dependencies."""
    if build_tool == JavaBuildTool.MAVEN:
        return "mvn dependency:resolve"
    if build_tool == JavaBuildTool.GRADLE:
        return "./gradlew dependencies"
    return "mvn dependency:resolve"


def get_java_test_command(build_tool: JavaBuildTool) -> str:
    """Get the test command for Java projects."""
    if build_tool == JavaBuildTool.MAVEN:
        return "mvn test"
    if build_tool == JavaBuildTool.GRADLE:
        return "./gradlew test"
    return "mvn test"

_SPOTLESS_COMMANDS = {
    JavaBuildTool.MAVEN: ["mvn spotless:apply -DspotlessFiles=$file"],
    JavaBuildTool.GRADLE: ["./gradlew spotlessApply"],
}
