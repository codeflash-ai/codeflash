from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import git
import tomlkit
from git import InvalidGitRepositoryError, Repo
from textual import on
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, RadioButton, RadioSet, Select, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
from codeflash.cli_cmds.cmd_init import (
    detect_test_framework_from_config_files,
    detect_test_framework_from_test_files,
    get_valid_subdirs,
)
from codeflash.cli_cmds.validators import APIKeyValidator
from codeflash.code_utils.shell_utils import save_api_key_to_rc
from codeflash.either import is_successful
from codeflash.telemetry.posthog_cf import ph
from codeflash.version import __version__ as version

CODEFLASH_LOGO: str = (
    r"                   _          ___  _               _     "
    r"                  | |        / __)| |             | |    "
    r"  ____   ___    _ | |  ____ | |__ | |  ____   ___ | | _  "
    r" / ___) / _ \  / || | / _  )|  __)| | / _  | /___)| || \ "
    r"( (___ | |_| |( (_| |( (/ / | |   | |( ( | ||___ || | | |"
    r" \____) \___/  \____| \____)|_|   |_| \_||_|(___/ |_| |_|"
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
        self.git_remote: str = ""
        self.github_app_installed: bool = False
        self.github_actions: bool = False
        self.vscode_extension: bool = False
        self.config_saved: bool = False

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
            new_pyproject_toml = tomlkit.document()
            new_pyproject_toml["tool"] = {"codeflash": {}}
            try:
                pyproject_path.write_text(tomlkit.dumps(new_pyproject_toml), encoding="utf8")
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
        ph("cli-installation-successful", {"did_add_new_key": bool(self.api_key)})
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


class DirectorySelectorWidget(Container):
    def __init__(self, select_id: str, input_id: str, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.select_id = select_id
        self.input_id = input_id

    def compose(self) -> ComposeResult:
        yield Select([("Loading...", "")], prompt="Select directory", id=self.select_id, allow_blank=False)
        yield Input(placeholder="Enter custom path", id=self.input_id, classes="hidden")

    def set_options(self, options: list[tuple[str, str]]) -> None:
        select = self.query_one(f"#{self.select_id}", Select)
        select.set_options(options)

    def get_selected_path(self) -> str | None:
        select = self.query_one(f"#{self.select_id}", Select)
        custom_input = self.query_one(f"#{self.input_id}", Input)

        if select.value == Select.BLANK:
            return None

        if select.value == "custom":
            return custom_input.value.strip() or None

        return select.value

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if event.select.id != self.select_id:
            return

        custom_input = self.query_one(f"#{self.input_id}", Input)
        if event.value == "custom":
            custom_input.remove_class("hidden")
        else:
            custom_input.add_class("hidden")


class WelcomeScreen(Screen):
    def __init__(self) -> None:
        super().__init__()
        self.existing_api_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(CODEFLASH_LOGO, classes="logo"),
            Static("Welcome to CodeFlash! 🚀\n", classes="title"),
            Static("", id="description", classes="description"),
            Static("", id="api_key_label", classes="label"),
            Input(placeholder="cf_xxxxxxxxxxxxxxxxxxxxxxxx", id="api_key_input", validators=[APIKeyValidator()]),
            Horizontal(
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
                # API key exists - show confirmation message
                display_key = f"{self.existing_api_key[:3]}****{self.existing_api_key[-4:]}"
                description.update(
                    "CodeFlash automatically optimizes your Python code for better performance.\n\n"
                    f"✅ Found existing API key: {display_key}\n\n"
                    "You're all set! Click Continue to proceed with configuration."
                )
                # Hide API key input
                label.update("")
                api_key_input.display = False
                # Store in app
                self.app.api_key = self.existing_api_key
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
        # If we already have an API key from environment, just continue
        if self.existing_api_key:
            self.app.push_screen(ConfigCheckScreen())
            return

        # Otherwise validate the input
        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        validation_result = api_key_input.validate(api_key)
        if not validation_result.is_valid:
            error_msgs = validation_result.failure_descriptions
            self.notify("; ".join(error_msgs) if error_msgs else "Invalid API key", severity="error", timeout=5)
            return

        self.app.api_key = api_key
        self.app.push_screen(ConfigCheckScreen())

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
            Static("Test Framework Configuration", classes="screen_title"),
            Static("", id="framework_desc", classes="description"),
            RadioSet(
                RadioButton("pytest (recommended)", id="pytest", value=True),
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
        return FormatterScreen()


class TestDiscoveryScreen(BaseConfigScreen):
    def on_mount(self) -> None:
        valid_subdirs = get_valid_subdirs()
        module_root = getattr(self.app, "module_path", None)

        test_subdirs = [d for d in valid_subdirs if d != module_root]
        options = [(d, d) for d in test_subdirs]

        if "tests" not in valid_subdirs:
            options.append(("Create new tests/ directory", "create"))

        options.append(("Custom path...", "custom"))

        selector = self.query_one(DirectorySelectorWidget)
        selector.set_options(options)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Test Directory Discovery", classes="screen_title"),
            Static("Where are your test files located?\n\nAvailable test directories:", classes="description"),
            DirectorySelectorWidget("test_select", "custom_path"),
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
        test_select = self.query_one("#test_select", Select)
        test_path = selector.get_selected_path()

        if not test_path:
            self.notify("Please select or enter a test directory", severity="error")
            return None

        if test_select.value == "create":
            test_path = "tests"
            tests_dir = Path.cwd() / test_path
            try:
                tests_dir.mkdir(exist_ok=True)
                self.notify(f"✅ Created directory: {test_path}", timeout=3)
            except Exception as e:
                self.notify(f"Failed to create directory: {e}", severity="error", timeout=5)
                return None
        else:
            path = Path.cwd() / test_path
            if not path.exists():
                self.notify(f"Test directory does not exist: {test_path}", severity="error", timeout=5)
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
        return TestFrameworkScreen()


class ModuleDiscoveryScreen(BaseConfigScreen):
    def on_mount(self) -> None:
        valid_subdirs = [d for d in get_valid_subdirs() if d != "tests"]

        curdir = Path.cwd()
        options = [
            *[(d, d) for d in valid_subdirs],
            (f"current directory ({curdir.name})", "."),
            ("Custom path...", "custom"),
        ]

        selector = self.query_one(DirectorySelectorWidget)
        selector.set_options(options)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Module Discovery", classes="screen_title"),
            Static("Which Python module would you like to optimize?\n\nAvailable directories:", classes="description"),
            DirectorySelectorWidget("module_select", "custom_path"),
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
            self.notify("Please select or enter a module path", severity="error")
            return None

        self.app.module_path = module_path
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
        return GitHubActionsScreen()

    @on(Button.Pressed, "#skip_btn")
    def skip_pressed(self) -> None:
        self.app.github_app_installed = False
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
            Container(
                Checkbox("✓ I want to reconfigure", id="reconfigure_check", classes="hidden"),
                id="reconfigure_container",
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
        from codeflash.cli_cmds.cmd_init import config_found, is_valid_pyproject_toml

        status_widget = self.query_one("#config_status", Static)
        reconfigure_check = self.query_one("#reconfigure_check", Checkbox)

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

        github_status = "✓ Installed" if github_app else "✗ Not installed"
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
        return GitConfigScreen()


class GitHubActionsScreen(BaseConfigScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("GitHub Actions Setup", classes="screen_title"),
            Static("Enable continuous optimization with automated CI/CD workflows.", classes="description"),
            Static(
                "[b]Workflow Features:[/b]\n"
                "  • Triggers on every push to your repository\n"
                "  • Analyzes code for performance improvements\n"
                "  • Automatically creates PRs with optimizations\n\n"
                "[b dim]Note:[/b dim] The workflow file will be added to .github/workflows/",
                classes="info_section",
            ),
            Vertical(
                Checkbox("✓ Install GitHub Actions workflow", id="actions_check", value=True), classes="action_section"
            ),
            Horizontal(
                Button("Continue", variant="primary", id="continue_btn"),
                Button("Back", variant="default", id="back_btn"),
                classes="button_row",
            ),
            classes="center_container",
        )
        yield Footer()

    def install_workflow(self) -> bool:
        from importlib.resources import files

        from codeflash.cli_cmds.cmd_init import customize_codeflash_yaml_content

        try:
            # Check if git is available
            if Repo is None or git is None:
                self.notify("GitPython not available. Install with: pip install gitpython", severity="error", timeout=5)
                return False

            try:
                repo = Repo(Path.cwd(), search_parent_directories=True)
                git_root = Path(repo.git.rev_parse("--show-toplevel"))
            except git.InvalidGitRepositoryError:
                self.notify("Not in a git repository. GitHub Actions requires git.", severity="error", timeout=5)
                return False

            workflows_dir = git_root / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)

            # Read workflow template
            workflow_template = files("codeflash").joinpath("cli_cmds", "workflows", "codeflash-optimize.yaml")
            workflow_content = workflow_template.read_text(encoding="utf-8")

            # Build config dict to match the format expected by customize_codeflash_yaml_content
            config = {
                "module_root": getattr(self.app, "module_path", "src"),
                "tests_root": getattr(self.app, "test_path", "tests"),
            }

            # Use the existing customization function from cmd_init
            workflow_content = customize_codeflash_yaml_content(
                workflow_content,
                config,
                git_root,
                benchmark_mode=False,  # Could be made configurable later
            )

            # Write workflow file
            workflow_file = workflows_dir / "codeflash.yaml"
            workflow_file.write_text(workflow_content, encoding="utf-8")
            return True
        except PermissionError:
            self.notify(
                "Permission denied: Unable to create workflow file. Please check directory permissions.",
                severity="error",
                timeout=5,
            )
            return False
        except Exception as e:
            self.notify(f"Failed to install workflow: {e!s}", severity="error", timeout=5)
            return False

    def get_next_screen(self) -> Screen | None:
        actions_check = self.query_one("#actions_check", Checkbox)
        self.app.github_actions = actions_check.value

        if actions_check.value:
            if self.install_workflow():
                self.notify(
                    "✓ GitHub Actions workflow installed to .github/workflows/codeflash.yml\n"
                    "Remember to add CODEFLASH_API_KEY to your GitHub repository secrets!",
                    severity="information",
                    timeout=6,
                )
            else:
                self.app.github_actions = False
        else:
            self.notify("Skipping GitHub Actions workflow", severity="information", timeout=2)

        return VSCodeExtensionScreen()
