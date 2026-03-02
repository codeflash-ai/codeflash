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
    EAGER_STATE_INIT = "eager_state_init"
    EXPENSIVE_RENDER_CALL = "expensive_render_call"
    COMBINABLE_LOOPS = "combinable_loops"
    SEQUENTIAL_AWAITS = "sequential_awaits"


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

# Lazy state initialization: useState(expensiveCall()) should be useState(() => expensiveCall())
# Matches useState(someFunc(...)) but not useState(() => ...) or useState(literal)
EAGER_STATE_INIT_RE = re.compile(
    r"\buseState\s*(?:<[^>]*>)?\s*\(\s*"
    r"(?![\s)'\"`\d\[\{tfnu])"  # not a literal, true/false/null/undefined, arrow, or empty
    r"(?!\(\s*\)\s*=>)"  # not () =>
    r"(?![a-zA-Z_]\w*\s*=>)"  # not param =>
    r"(\w+\s*\()"  # function call like computeX(
)

# Expensive object construction in render: new RegExp, new Date, new Map, new Set, JSON.parse
EXPENSIVE_RENDER_CALL_RE = re.compile(
    r"\bnew\s+(?:RegExp|Date|Map|Set|WeakMap|WeakSet|Int(?:8|16|32)Array|Float(?:32|64)Array)\s*\("
    r"|\bJSON\.(?:parse|stringify)\s*\("
)

# Sequential awaits: consecutive lines with await that could be parallelized
AWAIT_RE = re.compile(r"\bawait\s+(?!Promise\.all)")

# Array method chains on same variable for combinable loop detection
ARRAY_METHOD_RE = re.compile(r"(\w+)\.(?:filter|map|reduce|forEach|find|findIndex|every|some|flatMap)\s*\(")


def detect_optimization_opportunities(source: str, component_info: ReactComponentInfo) -> list[OptimizationOpportunity]:
    """Detect optimization opportunities in a React component."""
    opportunities: list[OptimizationOpportunity] = []
    lines = source.splitlines()

    # Only analyze the component's own lines
    start = component_info.start_line - 1
    end = min(component_info.end_line, len(lines))
    component_lines = lines[start:end]
    component_source = "\n".join(component_lines)

    # Check for inline objects in JSX props
    _detect_inline_props(component_lines, start, opportunities)

    # Check for functions defined in render body without useCallback
    _detect_missing_usecallback(component_source, component_lines, start, opportunities)

    # Check for expensive computations without useMemo
    _detect_missing_usememo(component_source, component_lines, start, opportunities)

    # Check for eager state initialization
    _detect_eager_state_init(component_lines, start, opportunities)

    # Check for expensive object construction in render
    _detect_expensive_render_calls(component_source, component_lines, start, opportunities)

    # Check for combinable array loops
    _detect_combinable_loops(component_lines, start, opportunities)

    # Check for sequential awaits
    _detect_sequential_awaits(component_lines, start, opportunities)

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
    has_usecallback = bool(USECALLBACK_RE.search(component_source))

    for i, line in enumerate(lines):
        line_num = offset + i + 1
        stripped = line.strip()
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


def _detect_eager_state_init(lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]) -> None:
    """Detect useState with eager (non-lazy) expensive initializers.

    useState(expensiveComputation()) runs the computation on every render,
    but only uses the result on mount. useState(() => expensiveComputation())
    only runs it once.
    """
    for i, line in enumerate(lines):
        line_num = offset + i + 1
        stripped = line.strip()
        if EAGER_STATE_INIT_RE.search(stripped):
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.EAGER_STATE_INIT,
                    line=line_num,
                    description="useState() initializer calls a function eagerly on every render. "
                    "Use lazy initialization: useState(() => expensiveCall()) to run it only on mount.",
                    severity=OpportunitySeverity.HIGH,
                )
            )


def _detect_expensive_render_calls(
    component_source: str, lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]
) -> None:
    """Detect expensive object constructions in render body (new RegExp, new Date, JSON.parse)."""
    for i, line in enumerate(lines):
        line_num = offset + i + 1
        stripped = line.strip()
        if EXPENSIVE_RENDER_CALL_RE.search(stripped) and "useMemo" not in stripped:
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.EXPENSIVE_RENDER_CALL,
                    line=line_num,
                    description="Expensive object construction (new RegExp/Date/Map/Set or JSON.parse/stringify) "
                    "in render body runs on every render. Move to useMemo or module scope.",
                    severity=OpportunitySeverity.HIGH,
                )
            )


def _detect_combinable_loops(lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]) -> None:
    """Detect multiple array method calls on the same variable that could be combined.

    E.g., items.filter(...) and items.map(...) and items.reduce(...) on the same source
    could potentially be combined into a single pass.
    """
    # Collect variable -> lines mapping for array operations
    var_operations: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "useMemo" in stripped:
            continue
        for match in ARRAY_METHOD_RE.finditer(stripped):
            var_name = match.group(1)
            # Skip common false positives (single-char vars, 'this', 'console')
            if len(var_name) <= 1 or var_name in ("this", "console", "Math", "Object", "Array", "Promise"):
                continue
            line_num = offset + i + 1
            if var_name not in var_operations:
                var_operations[var_name] = []
            var_operations[var_name].append(line_num)

    for var_name, line_nums in var_operations.items():
        if len(line_nums) >= 3:
            opportunities.append(
                OptimizationOpportunity(
                    type=OpportunityType.COMBINABLE_LOOPS,
                    line=line_nums[0],
                    description=f"Variable '{var_name}' is iterated {len(line_nums)} times with separate array methods. "
                    "Consider combining into a single .reduce() pass to avoid scanning the data multiple times.",
                    severity=OpportunitySeverity.MEDIUM,
                )
            )


def _detect_sequential_awaits(lines: list[str], offset: int, opportunities: list[OptimizationOpportunity]) -> None:
    """Detect consecutive await statements that could be parallelized with Promise.all."""
    consecutive_awaits: list[int] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if AWAIT_RE.search(stripped):
            consecutive_awaits.append(offset + i + 1)
        elif stripped and not stripped.startswith("//") and not stripped.startswith("*"):
            # Non-await, non-empty, non-comment line breaks the sequence
            if len(consecutive_awaits) >= 2:
                opportunities.append(
                    OptimizationOpportunity(
                        type=OpportunityType.SEQUENTIAL_AWAITS,
                        line=consecutive_awaits[0],
                        description=f"{len(consecutive_awaits)} sequential await calls could be parallelized. "
                        "Use Promise.all() or Promise.allSettled() if the operations are independent.",
                        severity=OpportunitySeverity.MEDIUM,
                    )
                )
            consecutive_awaits = []

    # Check remaining at end
    if len(consecutive_awaits) >= 2:
        opportunities.append(
            OptimizationOpportunity(
                type=OpportunityType.SEQUENTIAL_AWAITS,
                line=consecutive_awaits[0],
                description=f"{len(consecutive_awaits)} sequential await calls could be parallelized. "
                "Use Promise.all() or Promise.allSettled() if the operations are independent.",
                severity=OpportunitySeverity.MEDIUM,
            )
        )
