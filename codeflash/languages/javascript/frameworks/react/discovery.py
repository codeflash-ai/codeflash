"""React component discovery via tree-sitter analysis.

Identifies React components (function, arrow, class) and hooks by analyzing
PascalCase naming, JSX returns, and hook usage patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

    from codeflash.languages.javascript.treesitter import FunctionNode, TreeSitterAnalyzer

logger = logging.getLogger(__name__)

PASCAL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
HOOK_CALL_RE = re.compile(r"\buse[A-Z]\w*\s*(?:<[^>]*>)?\s*\(")
HOOK_NAME_RE = re.compile(r"^use[A-Z]\w*$")

# Built-in React hooks
BUILTIN_HOOKS = frozenset(
    {
        "useState",
        "useEffect",
        "useContext",
        "useReducer",
        "useCallback",
        "useMemo",
        "useRef",
        "useImperativeHandle",
        "useLayoutEffect",
        "useInsertionEffect",
        "useDebugValue",
        "useDeferredValue",
        "useTransition",
        "useId",
        "useSyncExternalStore",
        "useOptimistic",
        "useActionState",
        "useFormStatus",
    }
)


class ComponentType(str, Enum):
    FUNCTION = "function"
    ARROW = "arrow"
    CLASS = "class"
    HOOK = "hook"


@dataclass(frozen=True)
class ReactComponentInfo:
    """Information about a discovered React component or hook."""

    function_name: str
    component_type: ComponentType
    uses_hooks: tuple[str, ...] = ()
    returns_jsx: bool = False
    props_type: str | None = None
    is_memoized: bool = False
    start_line: int = 0
    end_line: int = 0


def is_react_component(func: FunctionNode, source: str, analyzer: TreeSitterAnalyzer) -> bool:
    """Check if a function is a React component.

    A React component:
    - Has a PascalCase name
    - Returns JSX (or could be a hook if named use*)
    - Is not a class method (standalone function)
    """
    if func.is_method:
        return False

    name = func.name

    # Hooks (useXxx) are not components
    if HOOK_NAME_RE.match(name):
        return False

    if not PASCAL_CASE_RE.match(name):
        return False

    return _function_returns_jsx(func, source, analyzer)


def is_react_hook(func: FunctionNode) -> bool:
    """Check if a function is a custom React hook (useXxx naming)."""
    return bool(HOOK_NAME_RE.match(func.name)) and not func.is_method


def classify_component(func: FunctionNode, source: str, analyzer: TreeSitterAnalyzer) -> ComponentType | None:
    """Classify a function as a React component type, hook, or None."""
    if is_react_hook(func):
        return ComponentType.HOOK

    if not is_react_component(func, source, analyzer):
        return None

    if func.is_arrow:
        return ComponentType.ARROW

    return ComponentType.FUNCTION


def find_react_components(source: str, file_path: Path, analyzer: TreeSitterAnalyzer) -> list[ReactComponentInfo]:
    """Find all React components and hooks in a source file.

    Skips files with "use server" directive (Next.js Server Components).
    """
    # Skip Server Components
    if _has_server_directive(source):
        logger.debug("Skipping server component file: %s", file_path)
        return []

    functions = analyzer.find_functions(source, include_methods=False, include_arrow_functions=True, require_name=True)

    components: list[ReactComponentInfo] = []
    for func in functions:
        comp_type = classify_component(func, source, analyzer)
        if comp_type is None:
            continue

        hooks_used = _extract_hooks_used(func.source_text)
        props_type = _extract_props_type(func, source, analyzer)
        is_memoized = _is_wrapped_in_memo(func, source)

        components.append(
            ReactComponentInfo(
                function_name=func.name,
                component_type=comp_type,
                uses_hooks=tuple(hooks_used),
                returns_jsx=comp_type != ComponentType.HOOK and _function_returns_jsx(func, source, analyzer),
                props_type=props_type,
                is_memoized=is_memoized,
                start_line=func.start_line,
                end_line=func.end_line,
            )
        )

    return components


def _has_server_directive(source: str) -> bool:
    """Check for 'use server' directive at the top of the file."""
    for line in source.splitlines()[:5]:
        stripped = line.strip()
        if stripped in ('"use server"', "'use server'", '"use server";', "'use server';"):
            return True
        if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            break
    return False


def _function_returns_jsx(func: FunctionNode, source: str, analyzer: TreeSitterAnalyzer) -> bool:
    """Check if a function returns JSX by looking for jsx_element/jsx_self_closing_element nodes."""
    source_bytes = source.encode("utf-8")
    node = func.node

    # For arrow functions with expression body (implicit return), check the body directly
    body = node.child_by_field_name("body")
    if body:
        return _node_contains_jsx(body)

    return False


def _node_contains_jsx(node: Node) -> bool:
    """Recursively check if a tree-sitter node contains JSX."""
    if node.type in (
        "jsx_element",
        "jsx_self_closing_element",
        "jsx_fragment",
        "jsx_expression",
        "jsx_opening_element",
    ):
        return True

    # Check return statements
    if node.type == "return_statement":
        for child in node.children:
            if _node_contains_jsx(child):
                return True

    for child in node.children:
        if _node_contains_jsx(child):
            return True

    return False


HOOK_EXTRACT_RE = re.compile(r"\b(use[A-Z]\w*)\s*(?:<[^>]*>)?\s*\(")


def _extract_hooks_used(function_source: str) -> list[str]:
    """Extract hook names called within a function body."""
    hooks = []
    seen = set()
    for match in HOOK_EXTRACT_RE.finditer(function_source):
        hook_name = match.group(1)
        if hook_name not in seen:
            seen.add(hook_name)
            hooks.append(hook_name)
    return hooks


def _extract_props_type(func: FunctionNode, source: str, analyzer: TreeSitterAnalyzer) -> str | None:
    """Extract the TypeScript props type annotation from a component's parameters."""
    source_bytes = source.encode("utf-8")
    node = func.node

    # Look for formal_parameters -> type_annotation
    params = node.child_by_field_name("parameters")
    if not params:
        return None

    for param in params.children:
        # Look for type annotation on first parameter
        if param.type in ("required_parameter", "optional_parameter"):
            type_node = param.child_by_field_name("type")
            if type_node:
                # Get the type annotation node (skip the colon)
                for child in type_node.children:
                    if child.type != ":":
                        return source_bytes[child.start_byte : child.end_byte].decode("utf-8")
        # Destructured params with type: { foo, bar }: Props
        if param.type == "object_pattern":
            # Look for next sibling that is a type_annotation
            next_sib = param.next_named_sibling
            if next_sib and next_sib.type == "type_annotation":
                for child in next_sib.children:
                    if child.type != ":":
                        return source_bytes[child.start_byte : child.end_byte].decode("utf-8")

    return None


def _is_wrapped_in_memo(func: FunctionNode, source: str) -> bool:
    """Check if the component is already wrapped in React.memo or memo()."""
    # Check if the variable declaration wrapping this function uses memo()
    # e.g., const MyComp = React.memo(function MyComp(...) {...})
    # or    const MyComp = memo((...) => {...})
    node = func.node
    parent = node.parent

    while parent:
        if parent.type == "call_expression":
            func_node = parent.child_by_field_name("function")
            if func_node:
                source_bytes = source.encode("utf-8")
                func_text = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf-8")
                if func_text in ("React.memo", "memo"):
                    return True
        parent = parent.parent

    # Also check for memo wrapping at the export level:
    # export default memo(MyComponent)
    name = func.name
    memo_patterns = [f"React.memo({name})", f"memo({name})", f"React.memo({name},", f"memo({name},"]
    return any(pattern in source for pattern in memo_patterns)
