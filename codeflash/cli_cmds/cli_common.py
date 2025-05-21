from __future__ import annotations

import shutil
import sys
from typing import Callable, cast

import click
import inquirer

from codeflash.cli_cmds.console import console, logger


def apologize_and_exit() -> None:
    console.rule()
    logger.info(
        "💡 If you're having trouble, see https://docs.codeflash.ai/getting-started/local-installation for further help getting started with Codeflash!"
    )
    console.rule()
    logger.info("👋 Exiting...")
    sys.exit(1)


def inquirer_wrapper(func: Callable[..., str | bool], *args: str | bool, **kwargs: str | bool) -> str | bool:
    new_args = []
    new_kwargs = {}

    if len(args) == 1:
        message = str(args[0])
    else:
        message = str(kwargs["message"])
        new_kwargs = kwargs.copy()
    split_messages = split_string_to_cli_width(message, is_confirm=func == inquirer.confirm)
    for split_message in split_messages[:-1]:
        click.echo(split_message)

    last_message = split_messages[-1]

    if len(args) == 1:
        new_args.append(last_message)
    else:
        new_kwargs["message"] = last_message

    return func(*new_args, **new_kwargs)


def split_string_to_cli_width(string: str, is_confirm: bool = False) -> list[str]:  # noqa: FBT001, FBT002
    cli_width, _ = shutil.get_terminal_size()
    # split string to lines that accommodate "[?] " prefix
    cli_width -= len("[?] ")
    lines = split_string_to_fit_width(string, cli_width)

    # split last line to additionally accommodate ": " or " (y/N): " suffix
    cli_width -= len(" (y/N):") if is_confirm else len(": ")
    last_lines = split_string_to_fit_width(lines[-1], cli_width)

    lines = lines[:-1] + last_lines

    if len(lines) > 1:
        for i in range(len(lines[:-1])):
            # Add yellow color to question mark in "[?] " prefix
            lines[i] = "[\033[33m?\033[0m] " + lines[i]
    return lines


def inquirer_wrapper_path(*args: str, **kwargs: str) -> dict[str, str] | None:
    new_args = []
    message = kwargs["message"]
    new_kwargs = kwargs.copy()
    split_messages = split_string_to_cli_width(message)
    for split_message in split_messages[:-1]:
        click.echo(split_message)

    last_message = split_messages[-1]
    new_kwargs["message"] = last_message
    new_args.append(args[0])

    return cast("dict[str, str]", inquirer.prompt([inquirer.Path(*new_args, **new_kwargs)]))


def split_string_to_fit_width(string: str, width: int) -> list[str]:
    words = string.split()
    lines = []
    current_line = [words[0]]
    current_length = len(words[0])

    for word in words[1:]:
        word_length = len(word)
        if current_length + word_length + 1 <= width:
            current_line.append(word)
            current_length += word_length + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    lines.append(" ".join(current_line))
    return lines
