from __future__ import annotations

from typing import TYPE_CHECKING

from inquirer_textual.InquirerApp import InquirerApp
from inquirer_textual.widgets.InquirerCheckbox import InquirerCheckbox
from inquirer_textual.widgets.InquirerConfirm import InquirerConfirm
from inquirer_textual.widgets.InquirerSelect import InquirerSelect
from inquirer_textual.widgets.InquirerText import InquirerText

if TYPE_CHECKING:
    from inquirer_textual.common.Choice import Choice


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


def select(  # noqa: ANN201
    message: str, choices: list[str | Choice], default: str | Choice | None = None
):  # type: ignore[no-untyped-def]
    widget = InquirerSelect(message, choices, default, mandatory=True)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)
    return app.run(inline=True)


def confirm(message: str, *, default: bool = False):  # noqa: ANN201  # type: ignore[no-untyped-def]
    widget = InquirerConfirm(message, default=default, mandatory=True)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)
    return app.run(inline=True)


def text(message: str):  # noqa: ANN201  # type: ignore[no-untyped-def]
    widget = InquirerText(message)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)
    return app.run(inline=True)


def checkbox(  # noqa: ANN201
    message: str, choices: list[str | Choice], enabled: list[str | Choice] | None = None
):  # type: ignore[no-untyped-def]
    widget = InquirerCheckbox(message, choices, enabled)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)
    return app.run(inline=True)
