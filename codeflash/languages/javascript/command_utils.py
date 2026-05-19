from __future__ import annotations

import shutil
import sys
from pathlib import Path


def resolve_node_command(command: str) -> str:
    """Resolve a Node ecosystem executable to a subprocess-safe path.

    On Windows, tools such as npm, npx, pnpm, yarn, and bun are often exposed
    as .cmd wrappers. Passing the bare command name to subprocess can fail even
    when the command is on PATH, so resolve it to the concrete executable path
    first.
    """
    candidates = [command]
    if sys.platform == "win32" and not Path(command).suffix:
        candidates.extend([f"{command}.cmd", f"{command}.bat", f"{command}.exe"])

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return command


def resolve_node_command_list(command: list[str]) -> list[str]:
    if not command:
        return command
    return [resolve_node_command(command[0]), *command[1:]]
