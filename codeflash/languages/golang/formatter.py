from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def format_go_code(source: str, file_path: Path | None = None) -> str:
    gofmt = shutil.which("gofmt")
    if gofmt is None:
        goimports = shutil.which("goimports")
        if goimports is not None:
            gofmt = goimports
        else:
            logger.debug("No Go formatter found (gofmt/goimports), returning source unchanged")
            return source

    try:
        result = subprocess.run([gofmt], input=source, capture_output=True, text=True, timeout=15, check=False)
        if result.returncode == 0:
            return result.stdout
        logger.debug("gofmt failed: %s", result.stderr)
    except subprocess.TimeoutExpired:
        logger.warning("gofmt timed out")
    except Exception:
        logger.debug("gofmt error", exc_info=True)

    return source


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
