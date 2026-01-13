from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from inquirer_textual.InquirerApp import InquirerApp
from inquirer_textual.widgets.InquirerCheckbox import InquirerCheckbox
from inquirer_textual.widgets.InquirerConfirm import InquirerConfirm
from inquirer_textual.widgets.InquirerMulti import InquirerMulti
from inquirer_textual.widgets.InquirerSelect import InquirerSelect
from inquirer_textual.widgets.InquirerText import InquirerText
from textual.binding import Binding

if TYPE_CHECKING:
    from collections.abc import Iterable

    from inquirer_textual.common.Choice import Choice
    from inquirer_textual.widgets.InquirerWidget import InquirerWidget
    from textual.validation import Validator

# Keyboard hints for each widget type (styled dim)
HINT_SELECT = "[dim]↑/↓ navigate • Enter confirm • Esc cancel[/dim]"
HINT_CONFIRM = "[dim]y/n answer • Enter confirm • Esc cancel[/dim]"
HINT_TEXT = "[dim]Enter confirm • Esc cancel[/dim]"
HINT_CHECKBOX = "[dim]↑/↓ navigate • Space select • Enter confirm • Esc cancel[/dim]"


def with_hint(message: str, hint: str) -> str:
    """Append a keyboard hint to the message."""
    return f"{message}  {hint}"


def is_cancelled(result) -> bool:  # noqa: ANN001
    """Check if the user cancelled the prompt (Esc, Ctrl+C, Ctrl+D)."""
    return result.command is None or result.command == "quit"


class CodeflashThemedApp(InquirerApp):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "quit", "Cancel", show=False, priority=True),
        Binding("ctrl+c", "quit", "Cancel", show=False, priority=True),
        Binding("ctrl+d", "quit", "Cancel", show=False, priority=True),
    ]

    CSS = """
        App {
            background: #1e293b;
        }
        Screen {
            border-top: none;
            border-bottom: none;
            background: transparent;
            height: auto;
        }
        ListView {
            background: transparent;
            border: none;
        }
        ListItem {
            background: transparent;
            padding: 0 1;
            color: #f1f5f9;
        }
        ListItem.-highlight {
            background: #334155;
            color: $select-list-item-highlight-foreground;
        }
        Label {
            background: transparent;
            color: #f8fafc;
        }
        Static {
            background: transparent;
            color: #f8fafc;
        }
        Input {
            background: #334155;
            border: solid $primary;
            color: #f8fafc;
        }
        Input:focus {
            border: solid $accent;
        }
        InquirerHeader {
            margin-bottom: 1;
        }
        InquirerHeader Static {
            text-align: center;
            color: #38bdf8;
        }
    """

    def _update_bindings(self) -> None:
        """Override to preserve our cancel bindings."""
        super()._update_bindings()
        for binding in self.BINDINGS:
            self._bindings.bind(
                binding.key,
                binding.action,
                description=binding.description,
                show=binding.show,
                priority=binding.priority,
            )

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {
            "select-question-mark": "#FFC143",
            "select-list-item-highlight-foreground": "#2563EB",
            "input-color": "#3B82F6",
            "input-selection-background": "#1e293b",
            "accent": "#FFC143",
            "primary": "#2563EB",
            "secondary": "#414372",
        }


def create_app(
    widget: InquirerWidget, header: str | list[str] | None = None, *, show_footer: bool = False
) -> CodeflashThemedApp:
    app: CodeflashThemedApp = CodeflashThemedApp()
    app.widget = widget
    app.header = header
    app.show_footer = show_footer
    return app


def select(  # noqa: ANN201
    message: str,
    choices: list[str | Choice],
    default: str | Choice | None = None,
    header: str | list[str] | None = None,
):  # type: ignore[no-untyped-def]
    widget = InquirerSelect(with_hint(message, HINT_SELECT), choices, default, mandatory=True)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def confirm(message: str, *, default: bool = False, header: str | list[str] | None = None):  # noqa: ANN201  # type: ignore[no-untyped-def]
    widget = InquirerConfirm(with_hint(message, HINT_CONFIRM), default=default, mandatory=True)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def text(  # noqa: ANN201  # type: ignore[no-untyped-def]
    message: str, validators: Validator | Iterable[Validator] | None = None, header: str | list[str] | None = None
):
    widget = InquirerText(with_hint(message, HINT_TEXT), validators=validators)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def checkbox(  # noqa: ANN201
    message: str,
    choices: list[str | Choice],
    enabled: list[str | Choice] | None = None,
    header: str | list[str] | None = None,
):  # type: ignore[no-untyped-def]
    widget = InquirerCheckbox(with_hint(message, HINT_CHECKBOX), choices, enabled)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def multi(widgets: list[InquirerWidget], header: str | list[str] | None = None):  # noqa: ANN201  # type: ignore[no-untyped-def]
    multi_widget = InquirerMulti(widgets)
    app = create_app(multi_widget, header=header)
    return app.run(inline=True)


def select_or_exit(  # noqa: ANN201
    message: str,
    choices: list[str | Choice],
    default: str | Choice | None = None,
    exit_callback=None,  # noqa: ANN001
    header: str | list[str] | None = None,
):  # type: ignore[no-untyped-def]
    """Select with automatic exit on cancellation."""
    result = select(message, choices, default, header=header)
    if is_cancelled(result):
        if exit_callback:
            exit_callback()
        else:
            from codeflash.cli_cmds.cli_common import apologize_and_exit

            apologize_and_exit()
    return result.value


def text_or_exit(  # noqa: ANN201  # type: ignore[no-untyped-def]
    message: str,
    validators=None,  # noqa: ANN001
    exit_callback=None,  # noqa: ANN001
    header: str | list[str] | None = None,
):
    """Text input with automatic exit on cancellation."""
    result = text(message, validators, header=header)
    if is_cancelled(result):
        if exit_callback:
            exit_callback()
        else:
            from codeflash.cli_cmds.cli_common import apologize_and_exit

            apologize_and_exit()
    return result.value


def checkbox_or_default(  # noqa: ANN201
    message: str,
    choices: list[str | Choice],
    enabled: list[str | Choice] | None = None,
    default_on_cancel=None,  # noqa: ANN001
    header: str | list[str] | None = None,
):  # type: ignore[no-untyped-def]
    """Checkbox with default value on cancellation."""
    result = checkbox(message, choices, enabled, header=header)
    if is_cancelled(result):
        return default_on_cancel if default_on_cancel is not None else []
    return result.value
