from __future__ import annotations

from typing import TYPE_CHECKING

from inquirer_textual.InquirerApp import InquirerApp
from inquirer_textual.widgets.InquirerCheckbox import InquirerCheckbox
from inquirer_textual.widgets.InquirerConfirm import InquirerConfirm
from inquirer_textual.widgets.InquirerMulti import InquirerMulti
from inquirer_textual.widgets.InquirerSelect import InquirerSelect
from inquirer_textual.widgets.InquirerText import InquirerText

if TYPE_CHECKING:
    from collections.abc import Iterable

    from inquirer_textual.common.Choice import Choice
    from inquirer_textual.widgets.InquirerWidget import InquirerWidget
    from textual.validation import Validator


class CodeflashThemedApp(InquirerApp):
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
    widget = InquirerSelect(message, choices, default, mandatory=True)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def confirm(message: str, *, default: bool = False, header: str | list[str] | None = None):  # noqa: ANN201  # type: ignore[no-untyped-def]
    widget = InquirerConfirm(message, default=default, mandatory=True)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def text(  # noqa: ANN201  # type: ignore[no-untyped-def]
    message: str, validators: Validator | Iterable[Validator] | None = None, header: str | list[str] | None = None
):
    widget = InquirerText(message, validators=validators)
    app = create_app(widget, header=header)
    return app.run(inline=True)


def checkbox(  # noqa: ANN201
    message: str,
    choices: list[str | Choice],
    enabled: list[str | Choice] | None = None,
    header: str | list[str] | None = None,
):  # type: ignore[no-untyped-def]
    widget = InquirerCheckbox(message, choices, enabled)
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
    if result.command is None:
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
    if result.command is None:
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
    if result.command is None:
        return default_on_cancel if default_on_cancel is not None else []
    return result.value
