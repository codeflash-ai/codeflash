from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import git
from unidiff import PatchSet

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class DiffCategory(str, Enum):
    COSMETIC = "cosmetic"
    TRIVIAL = "trivial"
    MEANINGFUL = "meaningful"
    MAJOR = "major"


@dataclass(frozen=True)
class FunctionDiffInfo:
    category: DiffCategory
    added_logic_lines: int
    removed_logic_lines: int
    total_changed_lines: int
    is_comment_only: bool
    is_whitespace_only: bool


# Patterns for comments across languages
_COMMENT_PATTERNS = [
    re.compile(r"^\s*#"),  # Python
    re.compile(r"^\s*//"),  # JS/TS/Java
    re.compile(r"^\s*\*"),  # Multiline comment body
    re.compile(r"^\s*/\*"),  # Multiline comment start
    re.compile(r"^\s*\*/"),  # Multiline comment end
    re.compile(r'^\s*"""'),  # Python docstring
    re.compile(r"^\s*'''"),  # Python docstring
]

# Patterns for import/require statements
_IMPORT_PATTERNS = [
    re.compile(r"^\s*(import |from \S+ import )"),  # Python
    re.compile(r"^\s*(const|let|var)\s+.*=\s*require\("),  # JS require
    re.compile(r"^\s*import\s+"),  # JS/TS/Java import
]


def _is_comment_line(line: str) -> bool:
    return any(p.match(line) for p in _COMMENT_PATTERNS)


def _is_import_line(line: str) -> bool:
    return any(p.match(line) for p in _IMPORT_PATTERNS)


def _is_logic_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_comment_line(line):
        return False
    # String-only lines (just a string literal)
    if stripped.startswith(('"""', "'''", '"', "'")) and stripped.endswith(('"""', "'''", '"', "'")):
        return False
    return True


def classify_function_diff(func: FunctionToOptimize, repo_directory: Path | None = None) -> FunctionDiffInfo:
    """Classify the type of change made to a function based on git diff content."""
    if func.starting_line is None or func.ending_line is None:
        return FunctionDiffInfo(
            category=DiffCategory.MEANINGFUL,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=0,
            is_comment_only=False,
            is_whitespace_only=False,
        )

    diff_lines = _get_function_diff_lines(func, repo_directory)
    if not diff_lines:
        return FunctionDiffInfo(
            category=DiffCategory.COSMETIC,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=0,
            is_comment_only=False,
            is_whitespace_only=True,
        )

    added_lines = [line for line in diff_lines if line.startswith("+")]
    removed_lines = [line for line in diff_lines if line.startswith("-")]
    total_changed = len(added_lines) + len(removed_lines)

    # Strip the +/- prefix for content analysis
    added_content = [line[1:] for line in added_lines]
    removed_content = [line[1:] for line in removed_lines]

    # Check if all changes are whitespace-only
    added_stripped = [line.strip() for line in added_content]
    removed_stripped = [line.strip() for line in removed_content]
    if all(not s for s in added_stripped) and all(not s for s in removed_stripped):
        return FunctionDiffInfo(
            category=DiffCategory.COSMETIC,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=total_changed,
            is_comment_only=False,
            is_whitespace_only=True,
        )

    # Check if all changes are comment-only
    added_logic = [line for line in added_content if _is_logic_line(line)]
    removed_logic = [line for line in removed_content if _is_logic_line(line)]

    is_comment_only = len(added_logic) == 0 and len(removed_logic) == 0
    if is_comment_only:
        return FunctionDiffInfo(
            category=DiffCategory.COSMETIC,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=total_changed,
            is_comment_only=True,
            is_whitespace_only=False,
        )

    # Classify by logic change magnitude
    logic_change_count = len(added_logic) + len(removed_logic)

    if logic_change_count <= 2:
        category = DiffCategory.TRIVIAL
    elif logic_change_count <= 10:
        category = DiffCategory.MEANINGFUL
    else:
        category = DiffCategory.MAJOR

    return FunctionDiffInfo(
        category=category,
        added_logic_lines=len(added_logic),
        removed_logic_lines=len(removed_logic),
        total_changed_lines=total_changed,
        is_comment_only=False,
        is_whitespace_only=False,
    )


def _get_function_diff_lines(func: FunctionToOptimize, repo_directory: Path | None = None) -> list[str]:
    """Extract diff lines that fall within a function's line range."""
    if repo_directory is None:
        repo_directory = Path.cwd()

    try:
        repository = git.Repo(repo_directory, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return []

    commit = repository.head.commit
    try:
        uni_diff_text = repository.git.diff(
            commit.hexsha + "^1", commit.hexsha, ignore_blank_lines=True, ignore_space_at_eol=True
        )
    except git.GitCommandError:
        return []

    patch_set = PatchSet(StringIO(uni_diff_text))
    func_start = func.starting_line or 0
    func_end = func.ending_line or 0
    result: list[str] = []

    for patched_file in patch_set:
        file_path = Path(repository.working_dir) / patched_file.path
        if file_path != func.file_path:
            continue

        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no and func_start <= line.target_line_no <= func_end:
                    result.append(f"+{line.value}")
                elif line.is_removed and line.source_line_no and func_start <= line.source_line_no <= func_end:
                    result.append(f"-{line.value}")

    return result


def filter_cosmetic_diff_functions(
    functions: dict[Path, list[FunctionToOptimize]], repo_directory: Path | None = None
) -> tuple[dict[Path, list[FunctionToOptimize]], int]:
    """Remove functions where the diff is purely cosmetic (comments/whitespace only)."""
    filtered: dict[Path, list[FunctionToOptimize]] = {}
    skipped_count = 0

    for file_path, funcs in functions.items():
        kept: list[FunctionToOptimize] = []
        for func in funcs:
            try:
                diff_info = classify_function_diff(func, repo_directory)
            except Exception:
                kept.append(func)
                continue

            if diff_info.category == DiffCategory.COSMETIC:
                skipped_count += 1
                logger.debug(
                    f"Skipping {func.qualified_name} — diff is cosmetic "
                    f"({'comments only' if diff_info.is_comment_only else 'whitespace only'})"
                )
            else:
                kept.append(func)

        if kept:
            filtered[file_path] = kept

    if skipped_count > 0:
        logger.info(f"Diff analysis: skipped {skipped_count} function(s) with cosmetic-only changes")

    return filtered, skipped_count


def get_effort_for_diff(diff_info: FunctionDiffInfo) -> str | None:
    """Suggest an effort level based on diff category. Returns None to use default."""
    if diff_info.category == DiffCategory.TRIVIAL:
        return "low"
    if diff_info.category == DiffCategory.MAJOR:
        return "high"
    return None
