from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field


@dataclass
class FunctionFilterCriteria:
    """Criteria for filtering which functions to discover.

    Attributes:
        include_patterns: Glob patterns for functions to include.
        exclude_patterns: Glob patterns for functions to exclude.
        require_return: Only include functions with return statements.
        include_async: Include async functions.
        include_methods: Include class methods.
        min_lines: Minimum number of lines in the function.
        max_lines: Maximum number of lines in the function.

    """

    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    require_return: bool = True
    require_export: bool = True
    include_async: bool = True
    include_methods: bool = True
    min_lines: int | None = None
    max_lines: int | None = None

    def __post_init__(self) -> None:
        """Pre-compile regex patterns from glob patterns for faster matching."""
        self.include_regexes = [re.compile(fnmatch.translate(p)) for p in self.include_patterns]
        self.exclude_regexes = [re.compile(fnmatch.translate(p)) for p in self.exclude_patterns]

    def matches_include_patterns(self, name: str) -> bool:
        """Check if name matches any include pattern."""
        if not self.include_regexes:
            return True
        return any(regex.match(name) for regex in self.include_regexes)

    def matches_exclude_patterns(self, name: str) -> bool:
        """Check if name matches any exclude pattern."""
        if not self.exclude_regexes:
            return False
        return any(regex.match(name) for regex in self.exclude_regexes)
