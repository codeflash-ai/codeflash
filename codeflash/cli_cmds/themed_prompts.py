"""Themed prompts wrapper for inquirer-textual with CodeFlash styling.

This module provides themed prompt functions that match the original CodeFlash
inquirer theme (yellow question marks, bright blue selections, cyan defaults).
"""

from __future__ import annotations

from inquirer_textual.common.Choice import Choice  # type: ignore[import-untyped]
from inquirer_textual.common.Result import Result  # type: ignore[import-untyped]
from inquirer_textual.InquirerApp import InquirerApp  # type: ignore[import-untyped]
from inquirer_textual.widgets.InquirerCheckbox import InquirerCheckbox  # type: ignore[import-untyped]
from inquirer_textual.widgets.InquirerConfirm import InquirerConfirm  # type: ignore[import-untyped]
from inquirer_textual.widgets.InquirerSelect import InquirerSelect  # type: ignore[import-untyped]
from inquirer_textual.widgets.InquirerText import InquirerText  # type: ignore[import-untyped]


class CodeflashThemedApp(InquirerApp):  # type: ignore[misc]
    """Custom themed InquirerApp matching the original CodeFlash theme colors."""

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Return CodeFlash theme colors.

        Original CodeFlash theme from inquirer:
        - Question mark: yellow
        - Brackets: bright blue
        - Default: bright cyan
        - Selection: bright blue
        - Checkbox selected: ✅
        - Checkbox unselected: ⬜
        """
        return {
            # Question mark color - yellow like the original
            "select-question-mark": "#e5c07b",  # Gold/yellow
            # List item highlight - bright blue like the original selection
            "select-list-item-highlight-foreground": "#61afef",  # Bright blue
            # Input/text color - cyan like the original
            "input-color": "#61afef",  # Bright blue (used for inputs and selections)
            # Additional contrast colors
            "input-selection-background": "#3e4451",  # Subtle background for selected items
        }


def select(
    message: str, choices: list[str | Choice], default: str | Choice | None = None, mandatory: bool = True
) -> Result[str | Choice]:  # type: ignore[type-arg]
    """Display a select prompt with CodeFlash theming.

    Args:
        message: The prompt message to display
        choices: List of choices (strings or Choice objects)
        default: Default choice to pre-select
        mandatory: Whether a response is mandatory

    Returns:
        Result object containing the selected value and command

    """
    widget = InquirerSelect(message, choices, default, mandatory)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)  # type: ignore[assignment]
    return app.run(inline=True)  # type: ignore[return-value]


def confirm(message: str, default: bool = False, mandatory: bool = True) -> Result[bool]:  # type: ignore[type-arg]
    """Display a confirm prompt with CodeFlash theming.

    Args:
        message: The prompt message to display
        default: Default value (True for yes, False for no)
        mandatory: Whether a response is mandatory

    Returns:
        Result object containing the boolean value and command

    """
    widget = InquirerConfirm(message, default=default, mandatory=mandatory)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)  # type: ignore[assignment]
    return app.run(inline=True)  # type: ignore[return-value]


def text(message: str) -> Result[str]:  # type: ignore[type-arg]
    """Display a text input prompt with CodeFlash theming.

    Args:
        message: The prompt message to display

    Returns:
        Result object containing the text value and command

    """
    widget = InquirerText(message)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)  # type: ignore[assignment]
    return app.run(inline=True)  # type: ignore[return-value]


def checkbox(
    message: str, choices: list[str | Choice], enabled: list[str | Choice] | None = None
) -> Result[list[str | Choice]]:  # type: ignore[type-arg]
    """Display a checkbox prompt with CodeFlash theming.

    Args:
        message: The prompt message to display
        choices: List of choices (strings or Choice objects)
        enabled: List of pre-selected choices

    Returns:
        Result object containing the list of selected values and command

    """
    widget = InquirerCheckbox(message, choices, enabled)
    app: CodeflashThemedApp = CodeflashThemedApp(widget, shortcuts=None, show_footer=False)  # type: ignore[assignment]
    return app.run(inline=True)  # type: ignore[return-value]
