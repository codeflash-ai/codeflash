"""Static analysis for React optimization opportunities.

Detects common performance anti-patterns in React components:
- Inline object/array creation in JSX props
- Functions defined inside render body (missing useCallback)
- Expensive computations without useMemo
- Components receiving referentially unstable props
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.languages.javascript.frameworks.react.discovery import ReactComponentInfo


class OpportunitySeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OpportunityType(str, Enum):
    INLINE_OBJECT_PROP = "inline_object_prop"
    INLINE_ARRAY_PROP = "inline_array_prop"
    MISSING_USECALLBACK = "missing_usecallback"
    MISSING_USEMEMO = "missing_usememo"
    MISSING_REACT_MEMO = "missing_react_memo"
    UNSTABLE_REFERENCE = "unstable_reference"


@dataclass(frozen=True)
class OptimizationOpportunity:
    """A detected optimization opportunity in a React component."""

    type: OpportunityType
    line: int
    description: str
    severity: OpportunitySeverity


# Patterns for expensive operations inside render body
EXPENSIVE_OPS_RE = re.compile(r"\.(filter|map|sort|reduce|flatMap|find|findIndex|every|some)\s*\(")
INLINE_OBJECT_IN_JSX_RE = re.compile(r"=\{\s*\{")  # ={{ ... }} in JSX
INLINE_ARRAY_IN_JSX_RE = re.compile(r"=\{\s*\[")  # ={[ ... ]} in JSX
FUNCTION_DEF_RE = re.compile(
    r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>"
    r"|function\s+\w+\s*\("
)
USECALLBACK_RE = re.compile(r"\buseCallback\s*\(")
USEMEMO_RE = re.compile(r"\buseMemo\s*\(")


def detect_optimization_opportunities(source: str, component_info: ReactComponentInfo) -> list[OptimizationOpportunity]:
    """Detect optimization opportunities in a React component."""
    opportunities: list[OptimizationOpportunity] = []
    lines = source.splitlines()

    # Only analyze the component's own lines
    start = component_info.start_line - 1
    end = min(component_info.end_line, len(lines))
    component_lines = lines[start:end]
    # Avoid building a large joined string; detectors will inspect lines directly.
    component_source = ""  # kept for signature compatibility with helpers

    # Check for inline objects in JSX props

    # Check for inline objects in JSX props
    _detect_inline_props(component_lines, start, opportunities)

    # Check for functions defined in render body without useCallback
    _detect_missing_usecallback(component_source, component_lines, start, opportunities)

    # Check for expensive computations without useMemo
    _detect_missing_usememo(component_source, component_lines, start, opportunities)

    # Check if component should be wrapped in React.memo
    if not component_info.is_memoized:
        opportunities.append(
            OptimizationOpportunity(
                type=OpportunityType.MISSING_REACT_MEMO,
                line=component_info.start_line,
                description=f"Component '{component_info.function_name}' is not wrapped in React.memo(). "
                "If it receives stable props, wrapping can prevent unnecessary re-renders.",
                severity=OpportunitySeverity.MEDIUM,
            )
        )

    return opportunities


def _detect_inline_props(lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]) -> None:
    """Detect inline object/array literals in JSX prop positions."""
    for i, line in enumerate(lines):
        line_num = offset + i + 1
        # Quick check to avoid running regexes when no JSX prop assignment is present
        if "={" not in line:
            continue
        if INLINE_OBJECT_IN_JSX_RE.search(line):
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.INLINE_OBJECT_PROP,
                    line=line_num,
                    description="Inline object literal in JSX prop creates a new reference on every render. "
                    "Extract to useMemo or a module-level constant.",
                    severity=OpportunitySeverity.HIGH,
                )
            )
        if INLINE_ARRAY_IN_JSX_RE.search(line):
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.INLINE_ARRAY_PROP,
                    line=line_num,
                    description="Inline array literal in JSX prop creates a new reference on every render. "
                    "Extract to useMemo or a module-level constant.",
                    severity=OpportunitySeverity.HIGH,
                )
            )


def _detect_missing_usecallback(
    component_source: str, lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]
) -> None:
    """Detect arrow functions or function expressions that could use useCallback."""
    # Determine whether the component uses useCallback anywhere by scanning lines (avoid joining)
    has_usecallback = False
    for line in lines:
        if "useCallback" in line:
            # cheap substring check before regex to avoid unnecessary work
            if USECALLBACK_RE.search(line):
                has_usecallback = True
                break

    for i, line in enumerate(lines):
        line_num = offset + i + 1
        stripped = line.strip()
        # Look for arrow function or function expression definitions inside the component
        # Quick substring check: FUNCTION_DEF_RE targets lines with var/const/let/function
        if "const" not in stripped and "let" not in stripped and "var" not in stripped and "function" not in stripped:
            continue
        # Look for arrow function or function expression definitions inside the component
        if FUNCTION_DEF_RE.search(stripped) and "useCallback" not in stripped and "useMemo" not in stripped:
            # Skip if the component already uses useCallback extensively
            if not has_usecallback:
                opportunities.append(
                    OptimizationOpportunity(
                        type=OpportunityType.MISSING_USECALLBACK,
                        line=line_num,
                        description="Function defined inside render body creates a new reference on every render. "
                        "Wrap with useCallback() if passed as a prop to child components.",
                        severity=OpportunitySeverity.MEDIUM,
                    )
                )


def _detect_missing_usememo(
    component_source: str, lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]
) -> None:
    """Detect expensive computations that could benefit from useMemo."""
    for i, line in enumerate(lines):
        line_num = offset + i + 1
        stripped = line.strip()
        # Quick exclusions to avoid running the expensive regex when impossible to match
        # expensive ops are accessed via a dot call like arr.map(
        if "." not in stripped:
            continue
        if EXPENSIVE_OPS_RE.search(stripped) and "useMemo" not in stripped:
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.MISSING_USEMEMO,
                    line=line_num,
                    description="Expensive array operation in render body runs on every render. "
                    "Wrap with useMemo() and specify dependencies.",
                    severity=OpportunitySeverity.HIGH,
                )
            )
