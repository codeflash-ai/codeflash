import os
import re
from functools import lru_cache

from rich.tree import Tree

from codeflash.models.test_type import TestType

_double_quote_pat = re.compile(r'"(.*?)"')
_single_quote_pat = re.compile(r"'(.*?)'")
worktree_path_regex = re.compile(r'\/[^"]*worktrees\/[^"]\S*')


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.getenv("CODEFLASH_LSP", default="false").lower() == "true"


def tree_to_markdown(tree: Tree, level: int = 0) -> str:
    """Convert a rich Tree into a Markdown bullet list."""
    indent = "  " * level
    if level == 0:
        lines: list[str] = [f"{indent}### {tree.label}"]
    else:
        lines: list[str] = [f"{indent}- {tree.label}"]
    for child in tree.children:
        lines.extend(tree_to_markdown(child, level + 1).splitlines())
    return "\n".join(lines)


def report_to_markdown_table(report: dict[TestType, dict[str, int]], title: str) -> str:
    lines = ["| Test Type | Passed ✅ |", "|-----------|--------|"]
    for test_type in TestType:
        if test_type is TestType.INIT_STATE_TEST:
            continue
        passed = report[test_type]["passed"]
        # failed = report[test_type]["failed"]
        if passed == 0:
            continue
        lines.append(f"| {test_type.to_name()} | {passed} |")
    table = "\n".join(lines)
    if title:
        return f"### {title}\n{table}"
    return table


def simplify_worktree_paths(msg: str, highlight: bool = True) -> str:  # noqa: FBT001, FBT002
    path_in_msg = worktree_path_regex.search(msg)
    if path_in_msg:
        last_part_of_path = path_in_msg.group(0).split("/")[-1]
        if highlight:
            last_part_of_path = f"`{last_part_of_path}`"
        return msg.replace(path_in_msg.group(0), last_part_of_path)
    return msg


def replace_quotes_with_backticks(text: str) -> str:
    # double-quoted strings
    text = _double_quote_pat.sub(r"`\1`", text)
    # single-quoted strings
    return _single_quote_pat.sub(r"`\1`", text)
