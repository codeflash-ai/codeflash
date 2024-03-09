import os
import re
from pathlib import Path
from typing import Optional

from returns.result import Failure, Result, Success

from codeflash.code_utils.compat import LF

if os.name == "nt":  # Windows
    SHELL_RC_EXPORT_PATTERN = re.compile(r"^set CODEFLASH_API_KEY=(.*)$", re.M)
    SHELL_RC_EXPORT_PREFIX = f"set CODEFLASH_API_KEY="
else:
    SHELL_RC_EXPORT_PATTERN = re.compile(r'^export CODEFLASH_API_KEY="?([^"]*)"?$', re.M)
    SHELL_RC_EXPORT_PREFIX = f"export CODEFLASH_API_KEY="


def read_api_key_from_shell_config() -> Optional[str]:
    try:
        shell_rc_path = get_shell_rc_path()
        with open(shell_rc_path, "r", encoding="utf8") as shell_rc:
            shell_contents = shell_rc.read()
            match = SHELL_RC_EXPORT_PATTERN.search(shell_contents)
            return match.group(1) if match else None
    except FileNotFoundError:
        return None


def get_shell_rc_path() -> str:
    """Get the path to the user's shell configuration file."""
    if os.name == "nt":  # on Windows, we use a batch file in the user's home directory
        return str(Path.home() / "codeflash_env.bat")
    else:
        shell = os.environ.get("SHELL", "/bin/bash").split("/")[-1]
        shell_rc_filename = {
            "zsh": ".zshrc",
            "ksh": ".kshrc",
            "csh": ".cshrc",
            "tcsh": ".cshrc",
            "dash": ".profile",
        }.get(
            shell, ".bashrc"
        )  # map each shell to its config file and default to .bashrc
        return str(Path.home() / shell_rc_filename)


def save_api_key_to_rc(api_key) -> Result[str, str]:
    shell_rc_path = get_shell_rc_path()
    api_key_line = f"{SHELL_RC_EXPORT_PREFIX}{api_key}"
    try:
        with open(shell_rc_path, "r+", encoding="utf8") as shell_file:
            shell_contents = shell_file.read()
            if os.name == "nt":  # on Windows, we're writing a batch file
                if not shell_contents:
                    shell_contents = "@echo off"
            existing_api_key = read_api_key_from_shell_config()

            if existing_api_key:
                # Replace the existing API key line
                updated_shell_contents = re.sub(
                    SHELL_RC_EXPORT_PATTERN, api_key_line, shell_contents
                )
                action = "Updated CODEFLASH_API_KEY in"
            else:
                # Append the new API key line
                updated_shell_contents = shell_contents.rstrip() + f"{LF}{api_key_line}{LF}"
                action = "Added CODEFLASH_API_KEY to"

            shell_file.seek(0)
            shell_file.write(updated_shell_contents)
            shell_file.truncate()
        return Success(f"✅ {action} {shell_rc_path}.")
    except PermissionError:
        return Failure(
            f"💡 I tried adding your CodeFlash API key to {shell_rc_path} - but seems like I don't have permissions to do so.{LF}"
            f"You'll need to open it yourself and add the following line:{LF}{LF}{api_key_line}{LF}"
        )
    except FileNotFoundError:
        return Failure(
            f"💡 I couldn't find your shell configuration file at {shell_rc_path}.{LF}"
            f"Please create it and add the following line:{LF}{LF}{api_key_line}{LF}"
        )
