"""React Profiler instrumentation for render counting and timing.

Wraps React components with React.Profiler to capture render count,
phase (mount/update), actualDuration, and baseDuration. Outputs structured
markers parseable by the existing marker-parsing infrastructure.

Marker format:
    !######REACT_RENDER:{component}:{phase}:{actualDuration}:{baseDuration}:{count}######!
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")

logger = logging.getLogger(__name__)

MARKER_PREFIX = "REACT_RENDER"


def generate_render_counter_code(component_name: str) -> str:
    """Generate the onRender callback and counter variable for Profiler instrumentation."""
    safe_name = _SAFE_NAME_RE.sub("_", component_name)
    return f"""\
let _codeflash_render_count_{safe_name} = 0;
function _codeflashOnRender_{safe_name}(id, phase, actualDuration, baseDuration) {{
  _codeflash_render_count_{safe_name}++;
  console.log(`!######{MARKER_PREFIX}:${{id}}:${{phase}}:${{actualDuration}}:${{baseDuration}}:${{_codeflash_render_count_{safe_name}}}######!`);
}}"""


def instrument_component_with_profiler(source: str, component_name: str, analyzer: TreeSitterAnalyzer) -> str:
    """Instrument a single component with React.Profiler.

    Wraps all JSX return statements with <React.Profiler> and adds the
    onRender callback + counter at module scope.

    Handles:
    - Single return statements
    - Conditional returns (if/else)
    - Fragment returns (<>...</>)
    - Early returns (leaves non-JSX returns alone)
    """
    source_bytes = _encode_cached(source)
    tree = analyzer.parse(source_bytes)

    safe_name = _SAFE_NAME_RE.sub("_", component_name)
    profiler_id = component_name

    # Find the component function node
    func_node = _find_component_function(tree.root_node, component_name, source_bytes)
    if func_node is None:
        logger.debug("Could not find component function: %s", component_name)
        return source

    # Find all return statements with JSX inside this function
    return_nodes = _find_jsx_returns(func_node, source_bytes)
    if not return_nodes:
        logger.debug("No JSX return statements found in: %s", component_name)
        return source

    # Compute all wrapped segments based on original source bytes, then build final string once.
    # This avoids repeated slicing/concatenation of the full source for each replacement.
    replacements: list[tuple[int, int, str]] = []
    for ret_node in sorted(return_nodes, key=lambda n: n.start_byte):
        computed = _compute_wrapped_segment(source_bytes, ret_node, profiler_id, safe_name)
        if computed is not None:
            replacements.append(computed)

    if replacements:
        # Reconstruct result in a single pass
        parts: list[str] = []
        prev = 0
        for start, end, wrapped in replacements:
            # Use original source slices (string indices expected by original logic)
            parts.append(source[prev:start])
            parts.append(wrapped)
            prev = end
        parts.append(source[prev:])
        result = "".join(parts)
    else:
        result = source

    # Add render counter code at the top (after imports) using the already-parsed tree

    # Add render counter code at the top (after imports)
    counter_code = generate_render_counter_code(component_name)

    # Inline logic similar to _insert_after_imports but reuse existing `tree` to avoid re-parsing
    last_import_end = 0
    for child in tree.root_node.children:
        if child.type == "import_statement":
            last_import_end = child.end_byte

    insert_pos = last_import_end
    while insert_pos < len(result) and result[insert_pos] != "\n":
        insert_pos += 1
    if insert_pos < len(result):
        insert_pos += 1  # skip the newline

    result = result[:insert_pos] + "\n" + counter_code + "\n\n" + result[insert_pos:]

    # Ensure React is imported

    # Ensure React is imported
    return _ensure_react_import(result)


def instrument_all_components_for_tracing(source: str, file_path: Path, analyzer: TreeSitterAnalyzer) -> str:
    """Instrument ALL components in a file for tracing/discovery mode."""
    from codeflash.languages.javascript.frameworks.react.discovery import find_react_components

    components = find_react_components(source, file_path, analyzer)
    if not components:
        return source

    result = source
    # Process in reverse order by start_line to preserve positions
    for comp in sorted(components, key=lambda c: c.start_line, reverse=True):
        if comp.returns_jsx:
            result = instrument_component_with_profiler(result, comp.function_name, analyzer)

    return result


def _find_component_function(root_node: Node, component_name: str, source_bytes: bytes) -> Node | None:
    """Find the tree-sitter node for a named component function."""
    # Check function declarations
    if root_node.type == "function_declaration":
        name_node = root_node.child_by_field_name("name")
        if name_node:
            name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name == component_name:
                return root_node

    # Check variable declarators with arrow functions (const MyComp = () => ...)
    if root_node.type == "variable_declarator":
        name_node = root_node.child_by_field_name("name")
        if name_node:
            name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name == component_name:
                return root_node

    # Check export statements
    if root_node.type in ("export_statement", "lexical_declaration", "variable_declaration"):
        for child in root_node.children:
            result = _find_component_function(child, component_name, source_bytes)
            if result:
                return result

    for child in root_node.children:
        result = _find_component_function(child, component_name, source_bytes)
        if result:
            return result

    return None


def _find_jsx_returns(func_node: Node, source_bytes: bytes) -> list[Node]:
    """Find all return statements that contain JSX within a function node."""
    returns: list[Node] = []

    def walk(node: Node) -> None:
        # Don't descend into nested functions
        if node != func_node and node.type in (
            "function_declaration",
            "arrow_function",
            "function",
            "method_definition",
        ):
            return

        if node.type == "return_statement":
            # Check if return value contains JSX
            for child in node.children:
                if _contains_jsx(child):
                    returns.append(node)
                    break
        else:
            for child in node.children:
                walk(child)

    walk(func_node)
    return returns


def _contains_jsx(node: Node) -> bool:
    """Check if a tree-sitter node contains JSX elements."""
    if node.type in ("jsx_element", "jsx_self_closing_element", "jsx_fragment"):
        return True
    return any(_contains_jsx(child) for child in node.children)


def _wrap_return_with_profiler(source: str, return_node: Node, profiler_id: str, safe_name: str) -> str:
    """Wrap a return statement's JSX with React.Profiler."""
    source_bytes = source.encode("utf-8")

    # Find the JSX part of the return (skip "return" keyword and whitespace)
    jsx_start = None
    jsx_end = return_node.end_byte

    for child in return_node.children:
        if child.type == "return":
            continue
        if child.type == ";":
            jsx_end = child.start_byte
            continue
        if _contains_jsx(child):
            jsx_start = child.start_byte
            jsx_end = child.end_byte
            break

    if jsx_start is None:
        return source

    jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()

    # Check if the return uses parentheses: return (...)
    # If so, we need to wrap inside the parens
    has_parens = False
    for child in return_node.children:
        if child.type == "parenthesized_expression":
            has_parens = True
            jsx_start = child.start_byte + 1  # skip (
            jsx_end = child.end_byte - 1  # skip )
            jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()
            break

    wrapped = (
        f'<React.Profiler id="{profiler_id}" onRender={{_codeflashOnRender_{safe_name}}}>'
        f"\n{jsx_content}\n"
        f"</React.Profiler>"
    )

    return source[:jsx_start] + wrapped + source[jsx_end:]


def _insert_after_imports(source: str, code: str, analyzer: TreeSitterAnalyzer) -> str:
    """Insert code after the last import statement."""
    source_bytes = source.encode("utf-8")
    tree = analyzer.parse(source_bytes)

    last_import_end = 0
    for child in tree.root_node.children:
        if child.type == "import_statement":
            last_import_end = child.end_byte

    # Find end of line after last import
    insert_pos = last_import_end
    while insert_pos < len(source) and source[insert_pos] != "\n":
        insert_pos += 1
    if insert_pos < len(source):
        insert_pos += 1  # skip the newline

    return source[:insert_pos] + "\n" + code + "\n\n" + source[insert_pos:]


def _ensure_react_import(source: str) -> str:
    """Ensure React is imported (needed for React.Profiler)."""
    if "import React" in source or "import * as React" in source:
        return source
    # Add React import at the top
    if "from 'react'" in source or 'from "react"' in source:
        # React is imported but maybe not as the default. That's fine for JSX.
        # We need React.Profiler so add it
        if "React" not in source.split("from", maxsplit=1)[0] if "from" in source else "":
            return 'import React from "react";\n' + source
        return source
    return 'import React from "react";\n' + source


# Cache small number of encoded strings to avoid repeated encode() overhead
@lru_cache(maxsize=64)
def _encode_cached(s: str) -> bytes:
    return s.encode("utf-8")





def _compute_wrapped_segment(source_bytes: bytes, return_node: Node, profiler_id: str, safe_name: str) -> tuple[int, int, str] | None:
    """Compute the replacement segment (start, end, wrapped) for a return node.

    Returns None if no JSX segment was found.
    """
    # Find the JSX part of the return (skip "return" keyword and whitespace)
    jsx_start = None
    jsx_end = return_node.end_byte

    for child in return_node.children:
        if child.type == "return":
            continue
        if child.type == ";":
            jsx_end = child.start_byte
            continue
        if _contains_jsx(child):
            jsx_start = child.start_byte
            jsx_end = child.end_byte
            break

    if jsx_start is None:
        return None

    # Default jsx_content from bytes slice
    jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()

    # Check if the return uses parentheses: return (...)
    # If so, we need to wrap inside the parens
    for child in return_node.children:
        if child.type == "parenthesized_expression":
            # skip outer parentheses
            jsx_start = child.start_byte + 1  # skip (
            jsx_end = child.end_byte - 1  # skip )
            jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()
            break

    wrapped = (
        f'<React.Profiler id="{profiler_id}" onRender={{_codeflashOnRender_{safe_name}}}>'
        f"\n{jsx_content}\n"
        f"</React.Profiler>"
    )

    return jsx_start, jsx_end, wrapped

# Cache small number of encoded strings to avoid repeated encode() overhead
@lru_cache(maxsize=64)
def _encode_cached(s: str) -> bytes:
    return s.encode("utf-8")




def _compute_wrapped_segment(source_bytes: bytes, return_node: Node, profiler_id: str, safe_name: str) -> tuple[int, int, str] | None:
    """Compute the replacement segment (start, end, wrapped) for a return node.

    Returns None if no JSX segment was found.
    """
    # Find the JSX part of the return (skip "return" keyword and whitespace)
    jsx_start = None
    jsx_end = return_node.end_byte

    for child in return_node.children:
        if child.type == "return":
            continue
        if child.type == ";":
            jsx_end = child.start_byte
            continue
        if _contains_jsx(child):
            jsx_start = child.start_byte
            jsx_end = child.end_byte
            break

    if jsx_start is None:
        return None

    # Default jsx_content from bytes slice
    jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()

    # Check if the return uses parentheses: return (...)
    # If so, we need to wrap inside the parens
    for child in return_node.children:
        if child.type == "parenthesized_expression":
            # skip outer parentheses
            jsx_start = child.start_byte + 1  # skip (
            jsx_end = child.end_byte - 1  # skip )
            jsx_content = source_bytes[jsx_start:jsx_end].decode("utf-8").strip()
            break

    wrapped = (
        f'<React.Profiler id="{profiler_id}" onRender={{_codeflashOnRender_{safe_name}}}>'
        f"\n{jsx_content}\n"
        f"</React.Profiler>"
    )

    return jsx_start, jsx_end, wrapped
