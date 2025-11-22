from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import git
import tomlkit
from git import InvalidGitRepositoryError, Repo
from textual import on, work
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from codeflash.cli_cmds.cmd_init import detect_test_framework_from_config_files, detect_test_framework_from_test_files
from codeflash.cli_cmds.console import logger
from codeflash.cli_cmds.validators import APIKeyValidator
from codeflash.code_utils.shell_utils import save_api_key_to_rc
from codeflash.either import is_successful
from codeflash.telemetry.posthog_cf import ph
from codeflash.version import __version__ as version

if TYPE_CHECKING:
    from textual.app import ComposeResult

CODEFLASH_LOGO: str = (
    r"                   _          ___  _               _     "
    "\n"
    r"                  | |        / __)| |             | |    "
    "\n"
    r"  ____   ___    _ | |  ____ | |__ | |  ____   ___ | | _  "
    "\n"
    r" / ___) / _ \  / || | / _  )|  __)| | / _  | /___)| || \ "
    "\n"
    r"( (___ | |_| |( (_| |( (/ / | |   | |( ( | ||___ || | | |"
    "\n"
    r" \____) \___/  \____| \____)|_|   |_| \_||_|(___/ |_| |_|"
    "\n"
    f"{('v' + version).rjust(66)}"
)


class CodeflashInit(App):
    CSS_PATH = "assets/style.tcss"
    TITLE = "Codeflash Configuration Setup"

    def __init__(self) -> None:
        super().__init__()
        # Store user configuration
        self.api_key: str = ""
        self.module_path: str = ""
        self.test_path: str = ""
        self.benchmarks_path: str = ""  # Optional: benchmarks directory
        self.test_framework: str = "pytest"
        self.formatter: str = "ruff"
        self.enable_telemetry: bool = True  # Default to enabled
        self.git_remote: str = ""
        self.github_app_installed: bool = False
        self.github_actions: bool = False
        self.vscode_extension: bool = False
        self.config_saved: bool = False
        # GitHub Actions PR creation tracking
        self.github_actions_pr_created: bool = False
        self.github_actions_pr_url: str = ""
        self.github_actions_secret_configured: bool = False
        self.github_actions_secret_error: str = ""
        self.github_actions_benchmark_mode: bool = False

    def on_mount(self) -> None:
        """Start with the welcome screen."""
        self.push_screen(WelcomeScreen())

    def save_configuration(self) -> bool:  # noqa: PLR0911
        """Save all configuration to pyproject.toml and API key to shell rc."""
        if self.config_saved:
            return True

        # Create pyproject.toml if it doesn't exist
        pyproject_path = Path.cwd() / "pyproject.toml"
        if not pyproject_path.exists():
            ph("tui-create-pyproject-toml")
            new_pyproject_toml = tomlkit.document()
            new_pyproject_toml["tool"] = {"codeflash": {}}
            try:
                pyproject_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")
                ph("tui-created-pyproject-toml")
            except OSError:
                return False

        # Read existing pyproject.toml
        try:
            with pyproject_path.open(encoding="utf8") as f:
                pyproject_data = tomlkit.parse(f.read())
        except Exception:
            return False

        # Build codeflash configuration section
        codeflash_section = tomlkit.table()
        codeflash_section.add(tomlkit.comment("All paths are relative to this pyproject.toml's directory."))
        codeflash_section["module-root"] = self.module_path
        codeflash_section["tests-root"] = self.test_path

        # Add benchmarks-root if specified
        if self.benchmarks_path:
            codeflash_section["benchmarks-root"] = self.benchmarks_path

        codeflash_section["test-framework"] = self.test_framework
        codeflash_section["ignore-paths"] = []

        # Add disable-telemetry if user opted out
        if not self.enable_telemetry:
            codeflash_section["disable-telemetry"] = True

        # Add formatter commands
        from codeflash.cli_cmds.cmd_init import get_formatter_cmds

        formatter_cmds = get_formatter_cmds(self.formatter)
        codeflash_section["formatter-cmds"] = formatter_cmds

        # Add git remote if not default
        if self.git_remote and self.git_remote != "origin":
            codeflash_section["git-remote"] = self.git_remote

        # Ensure tool section exists
        tool_section = pyproject_data.get("tool", tomlkit.table())
        tool_section["codeflash"] = codeflash_section
        pyproject_data["tool"] = tool_section

        try:
            with pyproject_path.open("w", encoding="utf8") as f:
                f.write(tomlkit.dumps(pyproject_data))
        except Exception:
            return False

        if self.api_key:
            from codeflash.code_utils.env_utils import get_codeflash_api_key

            try:
                existing_key = get_codeflash_api_key()
                # Only save if this is a new key
                if existing_key != self.api_key:
                    result = save_api_key_to_rc(self.api_key)
                    if not is_successful(result):
                        return False
            except OSError:
                # No existing key, save the new one
                result = save_api_key_to_rc(self.api_key)
                if not is_successful(result):
                    return False

            # Set environment variable for current session
            os.environ["CODEFLASH_API_KEY"] = self.api_key

        self.config_saved = True
        ph("tui-installation-successful", {"did_add_new_key": bool(self.api_key)})
        return True


class BaseConfigScreen(Screen):
    def get_next_screen(self) -> Screen | None:
        return None

    def get_previous_screen(self) -> bool:
        return True

    @on(Button.Pressed, "#continue_btn")
    def continue_pressed(self) -> None:
        next_screen = self.get_next_screen()
        if next_screen:
            self.app.push_screen(next_screen)
        else:
            self.app.exit()

    @on(Button.Pressed, "#back_btn")
    def back_pressed(self) -> None:
        if self.get_previous_screen():
            self.app.pop_screen()


class CustomDirectoryTree(DirectoryTree):
    """Directory tree that filters out hidden files and common cache directories."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        exclude_patterns = (".", "__pycache__", "node_modules", ".git")
        return [path for path in paths if not path.name.startswith(exclude_patterns)]


class DirectorySelectorWidget(Container):
    """Directory selector using DirectoryTree for visual navigation."""

    def __init__(self, tree_id: str, path_input_id: str, start_path: Path | None = None, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.tree_id = tree_id
        self.path_input_id = path_input_id
        self.start_path = start_path or Path.cwd()
        self.selected_dir: Path | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="path-controls"):
            yield Input(value=str(self.start_path), placeholder="Enter directory path", id=self.path_input_id)
            yield Button("⬆️", variant="default", id="btn_up", tooltip="Up One Level")
            yield Button("🏠", variant="default", id="btn_home", tooltip="Home")
            yield Button("📁", variant="default", id="btn_cwd", tooltip="Current Dir")

        yield CustomDirectoryTree(path=self.start_path, id=self.tree_id)

    def get_selected_path(self) -> str | None:
        """Get the currently selected directory path as a string relative to cwd."""
        if self.selected_dir:
            try:
                return str(self.selected_dir.relative_to(Path.cwd()))
            except ValueError:
                return str(self.selected_dir)
        return None

    @on(Button.Pressed, "#btn_up")
    def handle_up_button(self) -> None:
        """Navigate up one directory level."""
        tree = self.query_one(f"#{self.tree_id}", CustomDirectoryTree)
        current_path = tree.path
        if current_path and current_path.parent != current_path:
            self._set_directory(current_path.parent)

    @on(Button.Pressed, "#btn_home")
    def handle_home_button(self) -> None:
        """Navigate to home directory."""
        self._set_directory(Path.home())

    @on(Button.Pressed, "#btn_cwd")
    def handle_cwd_button(self) -> None:
        """Navigate to current working directory."""
        self._set_directory(Path.cwd())

    @on(Input.Changed)
    def handle_path_input_change(self, event: Input.Changed) -> None:
        """Update tree when path is manually entered."""
        if event.input.id != self.path_input_id:
            return

        try:
            path = Path(event.value)
            if path.exists() and path.is_dir():
                self._set_directory(path)
        except (OSError, ValueError):
            pass

    @on(DirectoryTree.DirectorySelected)
    def handle_directory_selection(self, event: DirectoryTree.DirectorySelected) -> None:
        """Update selected directory when user clicks on a directory."""
        if event.control.id != self.tree_id:
            return

        self.selected_dir = event.path
        path_input = self.query_one(f"#{self.path_input_id}", Input)
        try:
            relative_path = event.path.relative_to(Path.cwd())
            path_input.value = str(relative_path)
        except ValueError:
            path_input.value = str(event.path)

    def _set_directory(self, path: Path) -> None:
        """Set the directory tree to a new path."""
        if path.exists() and path.is_dir():
            tree = self.query_one(f"#{self.tree_id}", CustomDirectoryTree)
            tree.path = path
            path_input = self.query_one(f"#{self.path_input_id}", Input)
            path_input.value = str(path)
            self.selected_dir = path


class WelcomeScreen(Screen):
    def __init__(self) -> None:
        super().__init__()
        self.existing_api_key: str | None = None
        self.validating_api_key: bool = False
        self.current_api_key: str | None = None  # Track the key being validated
        self.last_validation_failed: bool = False  # Track if last validation failed

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(CODEFLASH_LOGO, classes="logo"),
            Static("Welcome to CodeFlash! 🚀\n", classes="title"),
            Static("", id="description", classes="description"),
            Static("", id="api_key_label", classes="label"),
            Input(placeholder="cf_xxxxxxxxxxxxxxxxxxxxxxxx", id="api_key_input", validators=[APIKeyValidator()]),
            Horizontal(
                Button("Verify API Key", variant="success", id="verify_btn", classes="hidden"),
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Quit", variant="default", id="quit_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Check for existing API key and update UI accordingly."""
        from codeflash.code_utils.env_utils import get_codeflash_api_key

        description = self.query_one("#description", Static)
        label = self.query_one("#api_key_label", Static)
        api_key_input = self.query_one("#api_key_input", Input)

        try:
            self.existing_api_key = get_codeflash_api_key()
            if self.existing_api_key:
                # API key exists - show confirmation message and validate it
                display_key = f"{self.existing_api_key[:3]}****{self.existing_api_key[-4:]}"
                description.update(
                    "CodeFlash automatically optimizes your Python code for better performance.\n\n"
                    f"🔍 Found existing API key: {display_key}\n"
                    "Validating..."
                )
                # Hide API key input
                label.update("")
                api_key_input.display = False
                # Validate the existing API key asynchronously
                self.current_api_key = self.existing_api_key
                self.validate_api_key_async(self.existing_api_key)
                return
        except OSError:
            # No existing API key found
            pass

        # No API key - show input form
        description.update(
            "CodeFlash automatically optimizes your Python code for better performance.\n\n"
            "Before we begin, you'll need a CodeFlash API key.\n"
            "Visit app.codeflash.ai, sign up with GitHub, and generate your API key.\n"
        )
        label.update("Enter your CodeFlash API Key:")

    @on(Button.Pressed, "#continue_btn")
    def continue_pressed(self) -> None:
        if self.validating_api_key:
            return

        if self.app.api_key:
            self.app.push_screen(ConfigCheckScreen())
            return

        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        validation_result = api_key_input.validate(api_key)
        if not validation_result.is_valid:
            error_msgs = validation_result.failure_descriptions
            self.notify("; ".join(error_msgs) if error_msgs else "Invalid API key", severity="error", timeout=5)
            return

        self.validating_api_key = True
        self.current_api_key = api_key
        continue_btn = self.query_one("#continue_btn", Button)
        continue_btn.label = "Validating..."
        continue_btn.disabled = True

        self.validate_api_key_async(api_key)

    @work(exclusive=True, thread=True)
    def validate_api_key_async(self, api_key: str) -> bool:
        from codeflash.api.cfapi import get_user_id_minimal

        return get_user_id_minimal(api_key)

    def on_worker_state_changed(self, event) -> None:  # noqa: ANN001
        """Handle worker state changes for API key validation."""
        if event.worker.name == "validate_api_key_async" and event.worker.is_finished:
            self.validating_api_key = False
            api_key = self.current_api_key

            if event.worker.result and api_key:
                self.app.api_key = api_key
                self.last_validation_failed = False

                # Hide verify button on success
                verify_btn = self.query_one("#verify_btn", Button)
                verify_btn.add_class("hidden")

                continue_btn = self.query_one("#continue_btn", Button)
                if self.existing_api_key:
                    description = self.query_one("#description", Static)
                    display_key = f"{api_key[:3]}****{api_key[-4:]}"
                    description.update(
                        "CodeFlash automatically optimizes your Python code for better performance.\n\n"
                        f"✅ Validated API key: {display_key}\n\n"
                        "You're all set! Click Continue to proceed with configuration."
                    )
                    self.set_timer(0.5, lambda: self.app.push_screen(ConfigCheckScreen()))
                else:
                    result = save_api_key_to_rc(api_key)
                    if is_successful(result):
                        logger.debug(f"Saved new API key to shell config: {result.unwrap()}")
                        os.environ["CODEFLASH_API_KEY"] = api_key
                    else:
                        logger.warning(f"Failed to save API key to shell config: {result.unwrap()}")

                    continue_btn.label = "Continue"
                    continue_btn.disabled = False
                    continue_btn.display = True
                    self.notify("✅ API key verified successfully!", severity="information", timeout=3)
            # Validation failed
            elif self.existing_api_key:
                # Show input field so user can enter a new key
                self.last_validation_failed = True
                description = self.query_one("#description", Static)
                label = self.query_one("#api_key_label", Static)
                api_key_input = self.query_one("#api_key_input", Input)
                verify_btn = self.query_one("#verify_btn", Button)

                display_key = f"{api_key[:3]}****{api_key[-4:]}" if api_key else "[hidden]"
                description.update(
                    "CodeFlash automatically optimizes your Python code for better performance.\n\n"
                    f"❌ Invalid API key found: {display_key}\n\n"
                    "Please enter a valid API key below:"
                )
                label.update("Enter your CodeFlash API Key:")
                continue_btn = self.query_one("#continue_btn", Button)
                continue_btn.display = False
                api_key_input.display = True
                api_key_input.focus()

                # Show verify button, reset its state
                verify_btn.label = "Verify API Key"
                verify_btn.disabled = False
                verify_btn.remove_class("hidden")

                self.existing_api_key = None  # Clear invalid existing key
                self.notify("Invalid API key - authentication failed", severity="error", timeout=5)
            else:
                self.last_validation_failed = True
                verify_btn = self.query_one("#verify_btn", Button)
                continue_btn = self.query_one("#continue_btn", Button)

                # Hide continue button, show verify button
                continue_btn.display = False
                verify_btn.label = "Verify API Key"
                verify_btn.disabled = False
                verify_btn.remove_class("hidden")

                self.notify("Invalid API key - authentication failed", severity="error", timeout=5)

    @on(Button.Pressed, "#verify_btn")
    def verify_api_key_pressed(self) -> None:
        """Manually verify API key when user clicks the Verify button."""
        if self.validating_api_key:
            return

        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        if not api_key:
            self.notify("Please enter an API key", severity="error", timeout=3)
            return

        # Check basic format validation first
        validation_result = api_key_input.validate(api_key)
        if not validation_result.is_valid:
            error_msgs = validation_result.failure_descriptions
            self.notify("; ".join(error_msgs) if error_msgs else "Invalid API key format", severity="error", timeout=5)
            return

        # Start validation
        self.validating_api_key = True
        self.last_validation_failed = False
        self.current_api_key = api_key

        # Update verify button state
        verify_btn = self.query_one("#verify_btn", Button)
        verify_btn.label = "Verifying..."
        verify_btn.disabled = True

        self.validate_api_key_async(api_key)

    @on(Button.Pressed, "#quit_btn")
    def quit_pressed(self) -> None:
        self.app.exit()


class VSCodeExtensionScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("VS Code Extension", classes="screen_title"),
            Static(
                "Installing the CodeFlash VS Code extension for enhanced development experience.", classes="description"
            ),
            Static(
                "[b]Extension Features:[/b]\n"
                "  • Inline optimization suggestions as you code\n"
                "  • Real-time performance metrics\n"
                "  • One-click apply optimizations\n"
                "  • Seamless integration with your workflow\n\n"
                "[b dim]Status:[/b dim] Attempting automatic installation...",
                classes="info_section",
                id="extension_info",
            ),
            Vertical(Static("", id="install_status", classes="status_text"), classes="action_section"),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.install_vscode_extension()

    def install_vscode_extension(self) -> None:
        status_widget = self.query_one("#install_status", Static)

        try:
            result = subprocess.run(["code", "--version"], check=False, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                status_widget.update("[yellow]⚠️ VS Code CLI not found[/yellow]")
                self.notify(
                    "VS Code CLI not available. Please install the extension manually from the marketplace.",
                    severity="warning",
                    timeout=5,
                )
                self.app.vscode_extension = False
                return

            install_result = subprocess.run(
                ["code", "--install-extension", "codeflash.codeflash"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if install_result.returncode == 0:
                status_widget.update("[green]✓ Extension installed successfully![/green]")
                self.notify("✓ VS Code extension installed successfully!", severity="information", timeout=3)
                self.app.vscode_extension = True
            else:
                error_msg = install_result.stderr.strip() if install_result.stderr else "Unknown error"
                if "not found" in error_msg.lower() or "marketplace" in error_msg.lower():
                    status_widget.update("[yellow]⚠️ Extension not yet available in marketplace[/yellow]")
                    self.notify(
                        "Extension not yet published. You'll be notified when it's available.",
                        severity="warning",
                        timeout=5,
                    )
                else:
                    status_widget.update(f"[yellow]⚠️ Installation failed: {error_msg[:50]}[/yellow]")
                    self.notify(
                        "Installation failed. Please install manually from VS Code.", severity="warning", timeout=5
                    )
                self.app.vscode_extension = False

        except subprocess.TimeoutExpired:
            status_widget.update("[yellow]⚠️ Installation timed out[/yellow]")
            self.notify("Installation timed out. Please try installing manually.", severity="warning", timeout=5)
            self.app.vscode_extension = False
        except FileNotFoundError:
            status_widget.update("[yellow]⚠️ VS Code CLI not found[/yellow]")
            self.notify(
                "VS Code CLI not available. Install the extension manually from the marketplace.",
                severity="warning",
                timeout=5,
            )
            self.app.vscode_extension = False
        except Exception as e:
            status_widget.update(f"[red]✗ Error: {str(e)[:50]}[/red]")
            self.notify("Error installing extension. Please install manually.", severity="error", timeout=5)
            self.app.vscode_extension = False

    def get_next_screen(self) -> Screen | None:
        return CompletionScreen()


class TestFrameworkScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Test Framework", classes="screen_title"),
            Static("", id="framework_desc", classes="description"),
            RadioSet(
                RadioButton("pytest", id="pytest", value=True),
                RadioButton("unittest", id="unittest"),
                id="framework_radio",
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Auto-detect test framework and update UI."""
        desc_widget = self.query_one("#framework_desc", Static)
        framework_radio = self.query_one("#framework_radio", RadioSet)

        # Try to auto-detect from config files
        detected_framework = detect_test_framework_from_config_files(Path.cwd())

        # If not found in config, try to detect from test files
        if not detected_framework:
            test_path = getattr(self.app, "test_path", "tests")
            tests_dir = Path.cwd() / test_path
            if tests_dir.exists():
                detected_framework = detect_test_framework_from_test_files(tests_dir)

        if detected_framework:
            desc_widget.update(f"Which test framework do you use?\n\n✅ Auto-detected: {detected_framework}\n")
            # Set the detected framework as default
            if detected_framework == "unittest":
                framework_radio.action_toggle_button("unittest")
        else:
            desc_widget.update("Which test framework do you use?\n\n(No test framework auto-detected)\n")

    def get_next_screen(self) -> Screen | None:
        framework_radio = self.query_one("#framework_radio", RadioSet)
        framework = "pytest" if framework_radio.pressed_button.id == "pytest" else "unittest"
        self.app.test_framework = framework
        ph("tui-test-framework-provided", {"test_framework": framework})
        return FormatterScreen()


class TestDiscoveryScreen(BaseConfigScreen):
    def __init__(self) -> None:
        super().__init__()
        self.tests_dir_exists = (Path.cwd() / "tests").exists()

    def compose(self) -> ComposeResult:
        # Auto-detect tests directory
        suggested_path = Path.cwd()
        tests_dir = Path.cwd() / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            suggested_path = tests_dir

        yield Header()
        yield Container(
            Static("Test Directory Discovery", classes="screen_title"),
            Static(
                "Where are your test files located?\n\nNavigate to your test directory using the tree below:",
                classes="description",
            ),
            DirectorySelectorWidget("test_tree", "test_path_input", start_path=suggested_path),
            Horizontal(
                Button("📁 Create tests/ directory", variant="success", id="create_tests_btn"),
                classes="centered_single_button" + (" hidden" if self.tests_dir_exists else ""),
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    @on(Button.Pressed, "#create_tests_btn")
    def create_tests_directory(self) -> None:
        """Create the tests directory and update the tree."""
        tests_dir = Path.cwd() / "tests"
        try:
            tests_dir.mkdir(exist_ok=True)
            self.notify("✅ Created tests/ directory", severity="information", timeout=3)

            # Hide the create button
            create_btn = self.query_one("#create_tests_btn", Button)
            create_btn.add_class("hidden")

            # Update the directory tree to show the new directory
            selector = self.query_one(DirectorySelectorWidget)
            selector._set_directory(tests_dir)  # noqa: SLF001
            selector.selected_dir = tests_dir

            self.tests_dir_exists = True
        except Exception as e:
            self.notify(f"Failed to create tests/ directory: {e}", severity="error", timeout=5)

    def get_next_screen(self) -> Screen | None:
        selector = self.query_one(DirectorySelectorWidget)
        test_path = selector.get_selected_path()

        if not test_path:
            self.notify("Please select a test directory", severity="error")
            return None

        path = Path.cwd() / test_path
        if not path.exists():
            # Offer to create the directory
            self.notify(
                f"Directory does not exist: {test_path}. Please select an existing directory.",
                severity="error",
                timeout=5,
            )
            return None

        if not path.is_dir():
            self.notify(f"Path is not a directory: {test_path}", severity="error", timeout=5)
            return None

        module_path = getattr(self.app, "module_path", "")
        if module_path and Path(test_path).resolve() == Path(module_path).resolve():
            self.notify(
                "Tests root cannot be the same as module root. This can lead to unexpected behavior.",
                severity="warning",
                timeout=6,
            )

        self.app.test_path = test_path
        ph("tui-tests-root-provided")
        return TestFrameworkScreen()


class ModuleDiscoveryScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Module Discovery", classes="screen_title"),
            Static(
                "Which Python module would you like to optimize?\n\nNavigate to your module directory using the tree below:",
                classes="description",
            ),
            DirectorySelectorWidget("module_tree", "module_path_input", start_path=Path.cwd()),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def get_next_screen(self) -> Screen | None:
        selector = self.query_one(DirectorySelectorWidget)
        module_path = selector.get_selected_path()

        if not module_path:
            self.notify("Please select a module directory", severity="error")
            return None

        path = Path.cwd() / module_path
        if not path.exists():
            self.notify(f"Directory does not exist: {module_path}", severity="error", timeout=5)
            return None

        if not path.is_dir():
            self.notify(f"Path is not a directory: {module_path}", severity="error", timeout=5)
            return None

        self.app.module_path = module_path
        ph("tui-project-root-provided")
        return TestDiscoveryScreen()


class GitHubAppScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("GitHub App Installation", classes="screen_title"),
            Static(
                "Enable automated optimization pull requests by installing the CodeFlash GitHub App.",
                classes="description",
            ),
            Static(
                "[b]What the GitHub App does:[/b]\n"
                "  • Opens pull requests with optimized code\n"
                "  • Accesses your repository (read/write permissions)\n"
                "  • Runs automated checks on your codebase\n\n"
                "[b dim]Steps:[/b dim] Open GitHub → Select repo → Approve permissions → Return here",
                classes="info_section",
            ),
            Vertical(
                Button("🌐 Open GitHub App Installation", id="open_browser_btn", variant="success"),
                Checkbox("✓ I have completed the installation", id="installed_check"),
                classes="action_section",
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Skip for Now", variant="default", id="skip_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    @on(Button.Pressed, "#open_browser_btn")
    def open_browser_pressed(self) -> None:
        """Open the GitHub App installation page in browser."""
        try:
            self.app.open_url("https://github.com/apps/codeflash-ai/installations/select_target")
            self.notify(
                "Opening GitHub in your browser... Complete the installation and return here.",
                severity="information",
                timeout=5,
            )
        except Exception as e:
            self.notify(
                f"Unable to open browser automatically: {e}\n"
                "Please visit: https://github.com/apps/codeflash-ai/installations/select_target",
                severity="error",
                timeout=8,
            )

    def get_next_screen(self) -> Screen | None:
        installed_check = self.query_one("#installed_check", Checkbox)
        if not installed_check.value:
            self.notify(
                "Please complete the GitHub App installation and check the box above, or click 'Skip for Now'.",
                severity="warning",
                timeout=5,
            )
            return None

        self.app.github_app_installed = True
        self.notify("✓ GitHub App configured successfully!", severity="information", timeout=2)
        if not self.app.save_configuration():
            self.notify("Failed to save configuration. GitHub Actions setup may fail.", severity="warning", timeout=5)

        return GitHubActionsScreen()

    @on(Button.Pressed, "#skip_btn")
    def skip_pressed(self) -> None:
        self.app.github_app_installed = False
        if not self.app.save_configuration():
            self.notify("Failed to save configuration. GitHub Actions setup may fail.", severity="warning", timeout=5)

        self.app.push_screen(GitHubActionsScreen())


class GitConfigScreen(BaseConfigScreen):
    def __init__(self) -> None:
        super().__init__()
        self.git_remotes: list[str] = []
        self.selected_remote: str = ""

    def on_mount(self) -> None:
        """Detect git configuration when screen is mounted."""
        status_widget = self.query_one("#git_status", Static)

        try:
            module_root = getattr(self.app, "module_path", ".")
            repo = Repo(Path.cwd() / module_root, search_parent_directories=True)

            # Get git remotes
            self.git_remotes = [remote.name for remote in repo.remotes]

            if not self.git_remotes:
                status_widget.update(
                    "⚠️  No git remotes found.\n\n"
                    "You can still use CodeFlash locally, but you'll need to set up a remote\n"
                    "repository to use GitHub features like pull requests."
                )
                self.selected_remote = ""
                return

            # Get current branch
            current_branch = repo.active_branch.name

            # Get remote URL for display
            if self.git_remotes:
                try:
                    remote_url = repo.remote(name=self.git_remotes[0]).url
                    # Clean up URL for display
                    display_url = remote_url.removesuffix(".git")
                    if "@" in display_url:
                        # SSH URL like git@github.com:user/repo
                        display_url = display_url.split("@")[1].replace(":", "/")
                    elif "://" in display_url:
                        # HTTPS URL
                        display_url = display_url.split("://")[1]
                except Exception:
                    display_url = "<remote>"
            else:
                display_url = "<no remote>"

            if len(self.git_remotes) > 1:
                # Multiple remotes - show select widget
                status_widget.update(
                    f"✅ Git repository detected\n\n"
                    f"Repository: {display_url}\n"
                    f"Branch: {current_branch}\n\n"
                    f"Multiple remotes found. Please select which remote\n"
                    f"CodeFlash should use for creating pull requests:"
                )

                # Show the select widget
                remote_select = self.query_one("#remote_select", Select)
                remote_select.remove_class("hidden")
                options = [(name, name) for name in self.git_remotes]
                remote_select.set_options(options)
                self.selected_remote = "origin" if "origin" in self.git_remotes else self.git_remotes[0]
            else:
                # Single remote - auto-select
                self.selected_remote = self.git_remotes[0]
                status_widget.update(
                    f"✅ Git repository detected\n\n"
                    f"Repository: {display_url}\n"
                    f"Branch: {current_branch}\n"
                    f"Remote: {self.selected_remote}\n\n"
                    f"This will be used for GitHub integration."
                )

        except InvalidGitRepositoryError:
            status_widget.update(
                "⚠️  No git repository found.\n\n"
                "You can still use CodeFlash locally, but you'll need to initialize\n"
                "a git repository to use GitHub features like pull requests."
            )
            self.selected_remote = ""
        except Exception as e:
            status_widget.update(
                f"❌ Error detecting git configuration:\n\n{e}\n\nYou can continue with local optimization."
            )
            self.selected_remote = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Git Configuration", classes="screen_title"),
            Static("Detecting git remote information...", id="git_status", classes="description"),
            Select(
                [("Loading...", "")],
                prompt="Select git remote",
                id="remote_select",
                classes="hidden",
                allow_blank=False,
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    @on(Select.Changed, "#remote_select")
    def remote_changed(self, event: Select.Changed) -> None:
        """Update selected remote when user changes selection."""
        if event.value != Select.BLANK:
            self.selected_remote = event.value

    def get_next_screen(self) -> Screen | None:
        self.app.git_remote = self.selected_remote
        return GitHubAppScreen()


class ConfigCheckScreen(BaseConfigScreen):
    def __init__(self) -> None:
        super().__init__()
        self.should_continue = False
        self.has_valid_config = False
        self.existing_config: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Configuration Check", classes="screen_title"),
            Static("", id="config_status", classes="description"),
            Checkbox("✓ I want to reconfigure", id="reconfigure_check", classes="hidden"),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        from codeflash.cli_cmds.cmd_init import config_found, is_valid_pyproject_toml

        status_widget = self.query_one("#config_status", Static)
        reconfigure_check = self.query_one("#reconfigure_check", Checkbox)

        # Check if current directory is writable
        if not os.access(Path.cwd(), os.W_OK):
            status_widget.update(
                "❌  Current directory is not writable\n\n"
                "Please check your folder permissions and try again.\n"
                "You need write permissions to create/modify pyproject.toml."
            )
            self.notify("Current directory is not writable. Please check permissions.", severity="error", timeout=10)
            return

        pyproject_path = Path.cwd() / "pyproject.toml"

        # Check if pyproject.toml exists
        found, _ = config_found(pyproject_path)
        if not found:
            status_widget.update(
                "⚠️  No pyproject.toml found in current directory\n\n"
                "This file is essential for CodeFlash configuration.\n"
                "A basic pyproject.toml will be created for you."
            )
            self.should_continue = True
            return

        # Check if it has valid codeflash config
        valid, config, message = is_valid_pyproject_toml(pyproject_path)

        if valid:
            # Valid config exists - ask if they want to reconfigure
            assert config is not None, "config should not be None when valid is True"
            self.has_valid_config = True
            self.existing_config = config
            status_widget.update(
                "✅ Found existing CodeFlash configuration!\n\n"
                f"Module: {config.get('module_root', 'N/A')}\n"
                f"Tests: {config.get('tests_root', 'N/A')}\n"
                f"Framework: {config.get('test_framework', 'N/A')}\n\n"
                "Do you want to reconfigure?"
            )
            # Show reconfigure checkbox
            reconfigure_check.remove_class("hidden")
        else:
            # Invalid or incomplete config
            status_widget.update(
                f"⚠️  Found pyproject.toml but configuration is invalid:\n\n"
                f"{message}\n\n"
                "Let's set up your project properly!"
            )
            self.should_continue = True

    @on(Checkbox.Changed, "#reconfigure_check")
    def reconfigure_changed(self, event: Checkbox.Changed) -> None:
        """Update should_continue when checkbox changes."""
        self.should_continue = event.value

    def get_next_screen(self) -> Screen | None:
        if self.has_valid_config and not self.should_continue:
            self.notify("Using existing configuration.", severity="information", timeout=2)
            return None

        return ModuleDiscoveryScreen()


class CompletionScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("✨ Setup Complete! ✨", classes="screen_title"),
            Static("Configuration saved to pyproject.toml", classes="subtitle"),
            Container(
                Static("📋 Configuration", classes="section_header"),
                Static("", id="config_content", classes="section_content"),
                classes="config_section",
            ),
            Container(
                Static("🚀 Available Commands", classes="section_header"),
                Static(
                    "codeflash optimize  - Run optimization\n"
                    "codeflash status    - Check status\n"
                    "codeflash report    - View report",
                    classes="section_content",
                ),
                classes="commands_section",
            ),
            Container(
                Static(
                    "⚠️  Restart your shell to load the\nCODEFLASH_API_KEY environment variable", classes="warning_text"
                ),
                classes="warning_section",
            ),
            Horizontal(Button("Finish", variant="success", id="finish_btn"), classes="button_row"),
            classes="center_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Update summary with actual configuration and save it."""
        config_content = self.query_one("#config_content", Static)

        module_path = getattr(self.app, "module_path", "src")
        test_path = getattr(self.app, "test_path", "tests")
        benchmarks_path = getattr(self.app, "benchmarks_path", "")
        test_framework = getattr(self.app, "test_framework", "pytest")
        formatter = getattr(self.app, "formatter", "ruff")
        github_app = getattr(self.app, "github_app_installed", False)
        github_actions = getattr(self.app, "github_actions", False)
        github_actions_pr_created = getattr(self.app, "github_actions_pr_created", False)
        github_actions_pr_url = getattr(self.app, "github_actions_pr_url", "")

        github_status = "✓ Installed" if github_app else "✗ Not installed"
        if github_actions_pr_created and github_actions_pr_url:
            actions_status = f"✓ Enabled\n  PR: {github_actions_pr_url}"
        else:
            actions_status = "✓ Enabled" if github_actions else "✗ Disabled"

        config_lines = [f"Module Path      : {module_path}", f"Test Path        : {test_path}"]

        if benchmarks_path:
            config_lines.append(f"Benchmarks Path  : {benchmarks_path}")

        config_lines.extend(
            [
                f"Test Framework   : {test_framework}",
                f"Code Formatter   : {formatter}",
                f"GitHub App       : {github_status}",
                f"GitHub Actions   : {actions_status}",
            ]
        )

        config_content.update("\n".join(config_lines))

        # Save configuration to pyproject.toml and API key to shell rc
        if not self.app.save_configuration():
            self.notify("Failed to save configuration. Please check permissions.", severity="error", timeout=5)

    def get_next_screen(self) -> Screen | None:
        return None  # Exit the app

    @on(Button.Pressed, "#finish_btn")
    def finish_pressed(self) -> None:
        self.app.exit()


class FormatterScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Code Formatter Setup", classes="screen_title"),
            Static(
                "Which code formatter would you like to use?\n\nThis will be used to format optimized code.",
                classes="description",
            ),
            RadioSet(
                RadioButton("ruff (recommended)", id="ruff", value=True),
                RadioButton("black", id="black"),
                RadioButton("disabled", id="disabled"),
                id="formatter_radio",
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def get_next_screen(self) -> Screen | None:
        formatter_radio = self.query_one("#formatter_radio", RadioSet)
        formatter = formatter_radio.pressed_button.id
        self.app.formatter = formatter
        return TelemetryScreen()


class TelemetryScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Telemetry Configuration", classes="screen_title"),
            Static(
                "Help us improve CodeFlash by sharing anonymous usage data.\n\n"
                "This includes:\n"
                "  • Errors encountered during optimization\n"
                "  • Performance metrics and success rates\n"
                "  • Feature usage statistics\n\n"
                "We never collect:\n"
                "  • Your source code\n"
                "  • File names or paths\n"
                "  • Personal information\n\n"
                "You can change this setting anytime in pyproject.toml",
                classes="description",
            ),
            RadioSet(
                RadioButton("✓ Enable telemetry (recommended)", id="enable", value=True),
                RadioButton("✗ Disable telemetry", id="disable"),
                id="telemetry_radio",
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def get_next_screen(self) -> Screen | None:
        telemetry_radio = self.query_one("#telemetry_radio", RadioSet)
        self.app.enable_telemetry = telemetry_radio.pressed_button.id == "enable"
        return GitConfigScreen()


class GitHubActionsScreen(BaseConfigScreen):
    def __init__(self) -> None:
        super().__init__()
        self.workflow_exists = False
        self.benchmark_mode_available = False

    def on_mount(self) -> None:
        """Check for existing workflow and benchmark mode."""
        try:
            repo = Repo(Path.cwd(), search_parent_directories=True)
            git_root = Path(repo.git.rev_parse("--show-toplevel"))
            workflow_path = git_root / ".github" / "workflows" / "codeflash.yaml"
            self.workflow_exists = workflow_path.exists()

            # Check if benchmarks directory exists
            benchmarks_path = getattr(self.app, "benchmarks_path", "")
            self.benchmark_mode_available = bool(benchmarks_path) and Path(benchmarks_path).exists()
        except Exception:
            self.workflow_exists = False
            self.benchmark_mode_available = False

    def compose(self) -> ComposeResult:
        yield Header()
        if self.workflow_exists:
            # Show existing workflow message
            yield Container(
                Static("GitHub Actions Setup", classes="screen_title"),
                Static("GitHub Actions workflow already configured.", classes="description"),
                Static(
                    "[b]Current Status:[/b]\\n"
                    "  ✅ Workflow exists at .github/workflows/codeflash.yaml\\n\\n"
                    "[b dim]Next Step:[/b dim] Make sure CODEFLASH_API_KEY is configured\\n"
                    "in your GitHub repository settings.",
                    classes="info_section",
                ),
                Horizontal(
                    Button("Continue", variant="primary", id="continue_btn"),
                    Button("Back", variant="default", id="back_btn"),
                    classes="button_row",
                ),
                classes="center_container",
            )
        else:
            # Show workflow setup options
            yield Container(
                Static("GitHub Actions Setup", classes="screen_title"),
                Static("Enable continuous optimization with automated CI/CD workflows.", classes="description"),
                Static(
                    "[b]Workflow Features:[/b]\\n"
                    "  • Triggers on every push to your repository\\n"
                    "  • Analyzes code for performance improvements\\n"
                    "  • Automatically creates PRs with optimizations\\n\\n"
                    "[b dim]Note:[/b dim] The workflow file will be added to .github/workflows/",
                    classes="info_section",
                ),
                Vertical(
                    Checkbox("✓ Install GitHub Actions workflow", id="actions_check", value=True),
                    classes="action_section",
                ),
                classes="center_container",
            )
            if self.benchmark_mode_available:
                yield Container(
                    Checkbox("✓ Enable benchmark mode (performance reports)", id="benchmark_check", value=False),
                    classes="action_section",
                )
            yield Container(
                Horizontal(
                    Button("Continue", variant="primary", id="continue_btn"),
                    Button("Back", variant="default", id="back_btn"),
                    classes="button_row",
                ),
                classes="center_container",
            )
        yield Footer()

    def get_next_screen(self) -> Screen | None:
        from importlib.resources import files

        from codeflash.api.cfapi import setup_github_actions
        from codeflash.cli_cmds.cmd_init import customize_codeflash_yaml_content
        from codeflash.code_utils.env_utils import get_codeflash_api_key
        from codeflash.code_utils.git_utils import get_current_branch, get_repo_owner_and_name

        # If workflow already exists, just continue
        if self.workflow_exists:
            self.app.github_actions = True
            return VSCodeExtensionScreen()

        # Check if user wants to install workflow
        actions_check = self.query_one("#actions_check", Checkbox)
        self.app.github_actions = actions_check.value

        if not actions_check.value:
            self.notify("Skipping GitHub Actions workflow", severity="information", timeout=5)
            return VSCodeExtensionScreen()

        # Get benchmark mode preference if available
        benchmark_mode = False
        if self.benchmark_mode_available:
            try:
                benchmark_check = self.query_one("#benchmark_check", Checkbox)
                benchmark_mode = benchmark_check.value
                self.app.github_actions_benchmark_mode = benchmark_mode
            except Exception:  # noqa: S110
                pass

        # Prepare workflow content
        try:
            # Get git information
            try:
                repo = Repo(Path.cwd(), search_parent_directories=True)
                git_root = Path(repo.git.rev_parse("--show-toplevel"))
            except git.InvalidGitRepositoryError:
                self.notify("Not in a git repository. GitHub Actions requires git.", severity="error", timeout=5)
                self.app.github_actions = False
                return VSCodeExtensionScreen()

            # Load and customize workflow template
            workflow_template = Path(files("codeflash") / "cli_cmds" / "workflows" / "codeflash-optimize.yaml")
            workflow_content = workflow_template.read_text(encoding="utf-8")

            # customize_codeflash_yaml_content expects config as a tuple: (dict, Path)
            try:
                config = (
                    {
                        "module_root": getattr(self.app, "module_path", "src"),
                        "tests_root": getattr(self.app, "test_path", "tests"),
                    },
                    Path.cwd() / "pyproject.toml",
                )

                workflow_content = customize_codeflash_yaml_content(
                    workflow_content, config, git_root, benchmark_mode=benchmark_mode
                )
            except FileNotFoundError as e:
                self.notify(
                    f"Configuration file not found. {e!s}\n\nPlease ensure pyproject.toml exists in your project.",
                    severity="error",
                    timeout=5,
                )
                self.app.github_actions = False
                return VSCodeExtensionScreen()

            # Get repository information
            try:
                owner, repo_name = get_repo_owner_and_name()
                base_branch = get_current_branch()
            except Exception as e:
                logger.warning(f"Could not get git remote info: {e}. Falling back to local workflow creation.")
                # Fallback: write workflow locally
                workflows_dir = git_root / ".github" / "workflows"
                workflows_dir.mkdir(parents=True, exist_ok=True)
                workflow_file = workflows_dir / "codeflash.yaml"
                workflow_file.write_text(workflow_content, encoding="utf-8")
                self.app.github_actions_pr_created = False
                self.notify(
                    "✓ GitHub Actions workflow created locally at .github/workflows/codeflash.yaml",
                    severity="information",
                    timeout=5,
                )
                return VSCodeExtensionScreen()

            # Get API key for secret setup (optional)
            api_key = None
            try:
                api_key = get_codeflash_api_key()
            except Exception:
                logger.debug("No API key available for secret setup")

            # Call CF API to create PR with workflow
            self.notify("Setting up GitHub Actions workflow...", severity="information", timeout=5)
            response = setup_github_actions(owner, repo_name, base_branch, workflow_content, api_key)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if response_data.get("success"):
                        pr_url = response_data.get("pr_url", "")
                        secret_success = response_data.get("secret_setup_success", False)
                        secret_error = response_data.get("secret_setup_error", "")

                        self.app.github_actions_pr_created = bool(pr_url)
                        self.app.github_actions_pr_url = pr_url
                        self.app.github_actions_secret_configured = secret_success
                        if secret_error:
                            self.app.github_actions_secret_error = secret_error

                        if pr_url:
                            self.notify(f"✓ PR created: {pr_url}", severity="information", timeout=5)
                            if not secret_success and secret_error:
                                self.notify(f"⚠️ Secret setup warning: {secret_error}", severity="warning", timeout=5)
                        else:
                            self.notify("✓ GitHub Actions workflow configured", severity="information", timeout=5)
                    else:
                        error_msg = response_data.get("error", "Unknown error")
                        logger.warning(f"API response indicated failure: {error_msg}")
                        self.notify(f"Could not create PR: {error_msg}", severity="warning", timeout=5)
                except Exception as e:
                    logger.error(f"Failed to parse API response: {e}")
                    self.notify("API response was invalid", severity="warning", timeout=5)
            else:
                logger.warning(f"API call failed with status {response.status_code}")
                self.notify(f"API error: Status {response.status_code}", severity="warning", timeout=5)

        except Exception as e:
            logger.error(f"Error setting up GitHub Actions: {e}")
            self.notify(f"Error: {e!s}", severity="error", timeout=5)
            self.app.github_actions = False

        return VSCodeExtensionScreen()


class GitHubActionsOnlyApp(App):
    """Lightweight TUI for `codeflash init-actions` command.

    This allows users to add/update GitHub Actions workflow after initial setup.
    """

    CSS_PATH = "assets/style.tcss"
    TITLE = "GitHub Actions Setup"

    def __init__(self) -> None:
        super().__init__()
        self.module_path: str = ""
        self.test_path: str = ""
        self.workflow_installed: bool = False

    def on_mount(self) -> None:
        """Load config and start with GitHub Actions screen."""
        from codeflash.code_utils.config_parser import parse_config_file

        ph("tui-github-actions-install-started")
        try:
            config, _ = parse_config_file()
            self.module_path = config.get("module_root", "src")
            self.test_path = config.get("tests_root", "tests")
            self.push_screen(GitHubActionsStandaloneScreen())
        except Exception:
            self.notify(
                "Could not load CodeFlash configuration. Please run 'codeflash init' first.",
                severity="error",
                timeout=5,
            )
            ph("tui-github-actions-config-not-found")
            self.exit()


class GitHubActionsStandaloneScreen(BaseConfigScreen):
    """Standalone screen for GitHub Actions setup."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("GitHub Actions Workflow Setup", classes="screen_title"),
            Static(
                "Add or update the CodeFlash GitHub Actions workflow for continuous optimization.\n\n"
                "This workflow will automatically optimize new code in every pull request.",
                classes="description",
            ),
            Static(
                "[b]Workflow Features:[/b]\n"
                "  • Triggers on every push to your repository\n"
                "  • Analyzes code for performance improvements\n"
                "  • Automatically creates PRs with optimizations\n\n"
                "[b dim]Note:[/b dim] The workflow file will be added/updated at .github/workflows/codeflash.yaml",
                classes="info_section",
            ),
            Horizontal(
                Button("Install Workflow", variant="primary", id="install_btn"),
                Button("Cancel", variant="default", id="cancel_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def install_workflow(self) -> bool:
        from importlib.resources import files

        from codeflash.cli_cmds.cmd_init import customize_codeflash_yaml_content
        from codeflash.code_utils.env_utils import get_codeflash_api_key
        from codeflash.code_utils.github_utils import get_github_secrets_page_url

        try:
            repo = Repo(Path.cwd(), search_parent_directories=True)
            git_root = Path(repo.git.rev_parse("--show-toplevel"))
        except (InvalidGitRepositoryError, AttributeError):
            self.notify("Not in a git repository. GitHub Actions requires git.", severity="error", timeout=5)
            return False

        workflows_dir = git_root / ".github" / "workflows"
        workflow_file = workflows_dir / "codeflash.yaml"

        # Check if workflow already exists
        if workflow_file.exists():
            self.notify("Workflow file already exists. It will be overwritten.", severity="warning", timeout=3)

        try:
            workflows_dir.mkdir(parents=True, exist_ok=True)

            # Read workflow template
            workflow_template = Path(files("codeflash") / "cli_cmds" / "workflows" / "codeflash-optimize.yaml")
            workflow_content = workflow_template.read_text(encoding="utf-8")

            config_dict = {
                "module_root": getattr(self.app, "module_path", "src"),
                "tests_root": getattr(self.app, "test_path", "tests"),
            }
            config = (config_dict, Path.cwd() / "pyproject.toml")

            # Customize workflow content
            workflow_content = customize_codeflash_yaml_content(
                workflow_content, config, git_root, benchmark_mode=False
            )

            # Write workflow file
            workflow_file.write_text(workflow_content, encoding="utf-8")

            # Show success message with API key reminder
            try:
                api_key = get_codeflash_api_key()
                secrets_url = get_github_secrets_page_url(repo)
                self.notify(
                    f"✅ Workflow installed at {workflow_file}\n\n"
                    f"Next: Add CODEFLASH_API_KEY to GitHub secrets: {secrets_url}\n"
                    f"Your API key: {api_key}",
                    severity="information",
                    timeout=10,
                )
            except OSError:
                self.notify(
                    f"✅ Workflow installed at {workflow_file}\n\n"
                    "Next: Add CODEFLASH_API_KEY to your GitHub repository secrets.",
                    severity="information",
                    timeout=8,
                )

            ph("tui-github-workflow-created")
            return True  # noqa: TRY300

        except PermissionError:
            self.notify(
                "Permission denied: Unable to create workflow file. Check directory permissions.",
                severity="error",
                timeout=5,
            )
            return False
        except Exception as e:
            self.notify(f"Failed to install workflow: {e!s}", severity="error", timeout=5)
            return False

    @on(Button.Pressed, "#install_btn")
    def install_pressed(self) -> None:
        if self.install_workflow():
            self.app.workflow_installed = True
            # Exit after successful installation
            self.app.exit()
        else:
            ph("tui-github-workflow-installation-failed")

    @on(Button.Pressed, "#cancel_btn")
    def cancel_pressed(self) -> None:
        ph("tui-github-workflow-skipped")
        self.app.exit()

    def get_previous_screen(self) -> bool:
        return False  # No back button in standalone mode
