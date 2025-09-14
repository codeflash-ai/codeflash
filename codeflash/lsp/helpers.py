import os
import re
from functools import lru_cache

_double_quote_pat = re.compile(r'"(.*?)"')

_single_quote_pat = re.compile(r"'(.*?)'")


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.getenv("CODEFLASH_LSP", default="false").lower() == "true"


worktree_path_regex = re.compile(r'\/[^"]*worktrees\/[^"]\S*')


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
