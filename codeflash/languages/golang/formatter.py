from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def format_go_code(source: str, file_path: Path | None = None) -> str:
    goimports = _find_go_tool("goimports")
    if goimports is not None:
        formatted = _run_formatter(goimports, source)
        if formatted is not None:
            return formatted

    gofmt = _find_go_tool("gofmt")
    if gofmt is not None:
        formatted = _run_formatter(gofmt, source)
        if formatted is not None:
            return formatted

    logger.debug("No Go formatter found (goimports/gofmt), returning source unchanged")
    return source


def _find_go_tool(name: str) -> str | None:
    import os
    from pathlib import Path

    found = shutil.which(name)
    if found:
        return found
    gopath = os.environ.get("GOPATH") or str(Path.home() / "go")
    for bin_dir in ("bin", str(Path("packages") / "bin")):
        candidate = Path(gopath) / bin_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run_formatter(tool: str, source: str) -> str | None:
    try:
        result = subprocess.run([tool], input=source, capture_output=True, text=True, timeout=15, check=False)
        if result.returncode == 0:
            return result.stdout
        logger.debug("%s failed: %s", tool, result.stderr)
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out", tool)
    except Exception:
        logger.debug("%s error", tool, exc_info=True)
    return None


def normalize_go_code(source: str) -> str:
    lines = source.splitlines()
    normalized: list[str] = []
    in_block_comment = False

    for line in lines:
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
                line = line[line.index("*/") + 2 :]
            else:
                continue

        if "//" in line:
            comment_pos = _find_line_comment_pos(line)
            if comment_pos >= 0:
                line = line[:comment_pos]

        if "/*" in line:
            start_idx = line.index("/*")
            if "*/" in line[start_idx:]:
                end_idx = line.index("*/", start_idx)
                line = line[:start_idx] + line[end_idx + 2 :]
            else:
                in_block_comment = True
                line = line[:start_idx]

        stripped = line.strip()
        if stripped:
            normalized.append(stripped)

    return "\n".join(normalized)


def _find_line_comment_pos(line: str) -> int:
    in_string = False
    in_rune = False
    escape_next = False
    in_raw_string = False

    i = 0
    while i < len(line):
        char = line[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if in_raw_string:
            if char == "`":
                in_raw_string = False
            i += 1
            continue

        if char == "`":
            in_raw_string = True
            i += 1
            continue

        if char == "\\":
            escape_next = True
            i += 1
            continue

        if char == '"' and not in_rune:
            in_string = not in_string
        elif char == "'" and not in_string:
            in_rune = not in_rune
        elif not in_string and not in_rune and i < len(line) - 1 and line[i : i + 2] == "//":
            return i

        i += 1

    return -1
