from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import LF
from codeflash.either import Failure, Success

if TYPE_CHECKING:
    from codeflash.either import Result

# PowerShell patterns and prefixes
POWERSHELL_RC_EXPORT_PATTERN = re.compile(
    r'^\$env:CODEFLASH_API_KEY\s*=\s*(?:"|\')?(cf-[^\s"\']+)(?:"|\')?\s*$', re.MULTILINE
)
POWERSHELL_RC_EXPORT_PREFIX = '$env:CODEFLASH_API_KEY = '

# CMD/Batch patterns and prefixes
CMD_RC_EXPORT_PATTERN = re.compile(r"^set CODEFLASH_API_KEY=(cf-.*)$", re.MULTILINE)
CMD_RC_EXPORT_PREFIX = "set CODEFLASH_API_KEY="

# Unix shell patterns and prefixes
UNIX_RC_EXPORT_PATTERN = re.compile(
    r'^(?!#)export CODEFLASH_API_KEY=(?:"|\')?(cf-[^\s"\']+)(?:"|\')?$', re.MULTILINE
)
UNIX_RC_EXPORT_PREFIX = "export CODEFLASH_API_KEY="


def is_powershell() -> bool:
    """
    Detect if we're running in PowerShell on Windows.
    
    Uses multiple heuristics:
    1. PSModulePath environment variable (PowerShell always sets this)
    2. COMSPEC pointing to powershell.exe
    3. TERM_PROGRAM indicating Windows Terminal (often uses PowerShell)
    """
    if os.name != "nt":
        return False
    
    # Primary check: PSModulePath is set by PowerShell
    # This is the most reliable indicator as PowerShell always sets this
    ps_module_path = os.environ.get("PSModulePath")
    if ps_module_path:
        logger.debug(f"shell_utils.py:is_powershell - Detected PowerShell via PSModulePath")
        return True
    
    # Secondary check: COMSPEC points to PowerShell
    comspec = os.environ.get("COMSPEC", "").lower()
    if "powershell" in comspec:
        logger.debug(f"shell_utils.py:is_powershell - Detected PowerShell via COMSPEC: {comspec}")
        return True
    
    # Tertiary check: Windows Terminal often uses PowerShell by default
    # But we only use this if other indicators are ambiguous
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if "windows" in term_program and "terminal" in term_program:
        # Check if we can find evidence of CMD (cmd.exe in COMSPEC)
        # If not, assume PowerShell for Windows Terminal
        if "cmd.exe" not in comspec:
            logger.debug(f"shell_utils.py:is_powershell - Detected PowerShell via Windows Terminal (COMSPEC: {comspec})")
            return True
    
    logger.debug(f"shell_utils.py:is_powershell - Not PowerShell (COMSPEC: {comspec}, TERM_PROGRAM: {term_program})")
    return False


def read_api_key_from_shell_config() -> Optional[str]:
    """Read API key from shell configuration file, checking both PowerShell and CMD files on Windows."""
    if os.name == "nt":  # Windows
        # Check PowerShell profile first if we're in PowerShell
        if is_powershell():
            ps_path = Path.home() / "codeflash_env.ps1"
            try:
                with open(ps_path, encoding="utf8") as shell_rc:  # noqa: PTH123
                    shell_contents = shell_rc.read()
                    matches = POWERSHELL_RC_EXPORT_PATTERN.findall(shell_contents)
                    if matches:
                        logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Found API key in PowerShell file: {ps_path}")
                        return matches[-1]
            except FileNotFoundError:
                logger.debug(f"shell_utils.py:read_api_key_from_shell_config - PowerShell file not found: {ps_path}")
            except Exception as e:
                logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Error reading PowerShell file: {e}")
        
        # Also check CMD batch file (for compatibility)
        bat_path = Path.home() / "codeflash_env.bat"
        try:
            with open(bat_path, encoding="utf8") as shell_rc:  # noqa: PTH123
                shell_contents = shell_rc.read()
                matches = CMD_RC_EXPORT_PATTERN.findall(shell_contents)
                if matches:
                    logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Found API key in CMD file: {bat_path}")
                    return matches[-1]
        except FileNotFoundError:
            logger.debug(f"shell_utils.py:read_api_key_from_shell_config - CMD file not found: {bat_path}")
        except Exception as e:
            logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Error reading CMD file: {e}")
        
        logger.debug(f"shell_utils.py:read_api_key_from_shell_config - No API key found in Windows config files")
        return None
    else:  # Unix-like
        shell_rc_path = get_shell_rc_path()
        try:
            with open(shell_rc_path, encoding="utf8") as shell_rc:  # noqa: PTH123
                shell_contents = shell_rc.read()
                matches = UNIX_RC_EXPORT_PATTERN.findall(shell_contents)
                if matches:
                    logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Found API key in Unix file: {shell_rc_path}")
                    return matches[-1]
                logger.debug(f"shell_utils.py:read_api_key_from_shell_config - No API key found in Unix file: {shell_rc_path}")
                return None
        except FileNotFoundError:
            logger.debug(f"shell_utils.py:read_api_key_from_shell_config - Unix file not found: {shell_rc_path}")
            return None


def get_shell_rc_path() -> Path:
    """Get the path to the user's shell configuration file."""
    if os.name == "nt":  # Windows
        if is_powershell():
            return Path.home() / "codeflash_env.ps1"
        else:
            return Path.home() / "codeflash_env.bat"
    shell = os.environ.get("SHELL", "/bin/bash").split("/")[-1]
    shell_rc_filename = {"zsh": ".zshrc", "ksh": ".kshrc", "csh": ".cshrc", "tcsh": ".cshrc", "dash": ".profile"}.get(
        shell, ".bashrc"
    )  # map each shell to its config file and default to .bashrc
    return Path.home() / shell_rc_filename


def get_api_key_export_line(api_key: str) -> str:
    """Get the appropriate export line based on the shell type."""
    if os.name == "nt":  # Windows
        if is_powershell():
            return f'{POWERSHELL_RC_EXPORT_PREFIX}"{api_key}"'
        else:
            return f'{CMD_RC_EXPORT_PREFIX}"{api_key}"'
    else:  # Unix-like
        return f'{UNIX_RC_EXPORT_PREFIX}"{api_key}"'


def save_api_key_to_rc(api_key: str) -> Result[str, str]:
    """Save API key to the appropriate shell configuration file."""
    shell_rc_path = get_shell_rc_path()
    api_key_line = get_api_key_export_line(api_key)
    
    logger.debug(f"shell_utils.py:save_api_key_to_rc - Saving API key to: {shell_rc_path}")
    logger.debug(f"shell_utils.py:save_api_key_to_rc - API key line format: {api_key_line[:30]}...")
    
    # Determine the correct pattern to use for replacement
    if os.name == "nt":  # Windows
        if is_powershell():
            pattern = POWERSHELL_RC_EXPORT_PATTERN
            logger.debug(f"shell_utils.py:save_api_key_to_rc - Using PowerShell pattern")
        else:
            pattern = CMD_RC_EXPORT_PATTERN
            logger.debug(f"shell_utils.py:save_api_key_to_rc - Using CMD pattern")
    else:  # Unix-like
        pattern = UNIX_RC_EXPORT_PATTERN
        logger.debug(f"shell_utils.py:save_api_key_to_rc - Using Unix pattern")
    
    try:
        # Create file if it doesn't exist
        shell_rc_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing contents or initialize
        try:
            with open(shell_rc_path, "r", encoding="utf8") as shell_file:  # noqa: PTH123
                shell_contents = shell_file.read()
                logger.debug(f"shell_utils.py:save_api_key_to_rc - Read existing file, length: {len(shell_contents)}")
        except FileNotFoundError:
            shell_contents = ""
            logger.debug(f"shell_utils.py:save_api_key_to_rc - File does not exist, creating new")
            # Add header for batch files
            if os.name == "nt" and not is_powershell() and not shell_contents:
                shell_contents = "@echo off"
                logger.debug(f"shell_utils.py:save_api_key_to_rc - Added @echo off header for batch file")
        
        # Check if API key already exists in the current file
        matches = pattern.findall(shell_contents)
        existing_in_file = bool(matches)
        logger.debug(f"shell_utils.py:save_api_key_to_rc - Existing key in file: {existing_in_file}")

        if existing_in_file:
            # Replace the existing API key line in this file
            updated_shell_contents = re.sub(pattern, api_key_line, shell_contents)
            action = "Updated CODEFLASH_API_KEY in"
            logger.debug(f"shell_utils.py:save_api_key_to_rc - Replaced existing API key")
        else:
            # Append the new API key line
            if shell_contents and not shell_contents.endswith(LF):
                updated_shell_contents = shell_contents + LF + api_key_line + LF
            else:
                updated_shell_contents = shell_contents.rstrip() + f"{LF}{api_key_line}{LF}"
            action = "Added CODEFLASH_API_KEY to"
            logger.debug(f"shell_utils.py:save_api_key_to_rc - Appended new API key")

        # Write the updated contents
        with open(shell_rc_path, "w", encoding="utf8") as shell_file:  # noqa: PTH123
            shell_file.write(updated_shell_contents)
        logger.debug(f"shell_utils.py:save_api_key_to_rc - Successfully wrote to {shell_rc_path}")
        
        return Success(f"✅ {action} {shell_rc_path}")
    except PermissionError as e:
        logger.debug(f"shell_utils.py:save_api_key_to_rc - Permission error: {e}")
        return Failure(
            f"💡 I tried adding your Codeflash API key to {shell_rc_path} - but seems like I don't have permissions to do so.{LF}"
            f"You'll need to open it yourself and add the following line:{LF}{LF}{api_key_line}{LF}"
        )
    except Exception as e:
        logger.debug(f"shell_utils.py:save_api_key_to_rc - Error: {e}")
        return Failure(
            f"💡 I went to save your Codeflash API key to {shell_rc_path}, but encountered an error: {e}{LF}"
            f"To ensure your Codeflash API key is automatically loaded into your environment at startup, you can create {shell_rc_path} and add the following line:{LF}"
            f"{LF}{api_key_line}{LF}"
        )
