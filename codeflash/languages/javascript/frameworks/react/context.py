"""React-specific context extraction for component optimization.

Extracts props interfaces, hook usage, parent/child component relationships,
context subscriptions, and optimization opportunities from React components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

    from codeflash.languages.javascript.frameworks.react.analyzer import OptimizationOpportunity
    from codeflash.languages.javascript.frameworks.react.discovery import ReactComponentInfo
    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class HookUsage:
    """Represents a hook call within a component."""

    name: str
    has_dependency_array: bool = False
    dependency_count: int = 0


@dataclass
class ReactContext:
    """Context information for a React component, used in LLM prompts."""

    props_interface: str | None = None
    hooks_used: list[HookUsage] = field(default_factory=list)
    parent_usages: list[str] = field(default_factory=list)
    child_components: list[str] = field(default_factory=list)
    context_subscriptions: list[str] = field(default_factory=list)
    is_already_memoized: bool = False
    optimization_opportunities: list[OptimizationOpportunity] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """Format this context for inclusion in an LLM optimization prompt."""
        parts: list[str] = []

        if self.props_interface:
            parts.append(f"Props interface:\n```typescript\n{self.props_interface}\n```")

        if self.hooks_used:
            hook_lines = []
            for hook in self.hooks_used:
                dep_info = f" (deps: {hook.dependency_count})" if hook.has_dependency_array else " (no deps)"
                hook_lines.append(f"  - {hook.name}{dep_info}")
            parts.append("Hooks used:\n" + "\n".join(hook_lines))

        if self.child_components:
            parts.append("Child components rendered: " + ", ".join(self.child_components))

        if self.context_subscriptions:
            parts.append("Context subscriptions: " + ", ".join(self.context_subscriptions))

        if self.is_already_memoized:
            parts.append("Note: Component is already wrapped in React.memo()")

        if self.optimization_opportunities:
            opp_lines = []
            for opp in self.optimization_opportunities:
                opp_lines.append(f"  - [{opp.severity.value}] Line {opp.line}: {opp.description}")
            parts.append("Detected optimization opportunities:\n" + "\n".join(opp_lines))

        return "\n\n".join(parts)


def extract_react_context(
    component_info: ReactComponentInfo, source: str, analyzer: TreeSitterAnalyzer, module_root: Path
) -> ReactContext:
    """Extract React-specific context for a component.

    Analyzes the component source to find props types, hooks, child components,
    and optimization opportunities.
    """
    from codeflash.languages.javascript.frameworks.react.analyzer import detect_optimization_opportunities

    context = ReactContext(props_interface=component_info.props_type, is_already_memoized=component_info.is_memoized)

    # Extract hook usage details from the component source
    lines = source.splitlines()
    start = component_info.start_line - 1
    end = min(component_info.end_line, len(lines))
    component_source = "\n".join(lines[start:end])

    context.hooks_used = _extract_hook_usages(component_source)
    context.child_components = _extract_child_components(component_source, analyzer, source)
    context.context_subscriptions = _extract_context_subscriptions(component_source)
    context.optimization_opportunities = detect_optimization_opportunities(source, component_info)

    # Extract full props interface definition if we have a type name
    if component_info.props_type:
        full_interface = _find_type_definition(component_info.props_type, source, analyzer)
        if full_interface:
            context.props_interface = full_interface

    return context


def _extract_hook_usages(component_source: str) -> list[HookUsage]:
    """Parse hook calls and their dependency arrays from component source."""
    import re

    hooks: list[HookUsage] = []
    # Match useXxx( patterns
    hook_pattern = re.compile(r"\b(use[A-Z]\w*)\s*\(")

    for match in hook_pattern.finditer(component_source):
        hook_name = match.group(1)
        # Try to determine if there's a dependency array
        # Look for ], [ pattern after the hook call (simplified heuristic)
        rest_of_line = component_source[match.end() :]
        has_deps = False
        dep_count = 0

        # Simple heuristic: count brackets to find dependency array
        bracket_depth = 1
        for i, char in enumerate(rest_of_line):
            if char == "(":
                bracket_depth += 1
            elif char == ")":
                bracket_depth -= 1
                if bracket_depth == 0:
                    # Check if the last argument before closing paren is an array
                    preceding = rest_of_line[:i].rstrip()
                    if preceding.endswith("]"):
                        has_deps = True
                        # Count items in the array (rough: count commas + 1 for non-empty)
                        array_start = preceding.rfind("[")
                        if array_start >= 0:
                            array_content = preceding[array_start + 1 : -1].strip()
                            if array_content:
                                dep_count = array_content.count(",") + 1
                            else:
                                dep_count = 0  # empty deps []
                                has_deps = True
                    break

        hooks.append(HookUsage(name=hook_name, has_dependency_array=has_deps, dependency_count=dep_count))

    return hooks


def _extract_child_components(component_source: str, analyzer: TreeSitterAnalyzer, full_source: str) -> list[str]:
    """Find child component names rendered in JSX."""
    import re

    # Match JSX tags that start with uppercase (React components)
    jsx_component_re = re.compile(r"<([A-Z][a-zA-Z0-9.]*)")
    children = set()
    for match in jsx_component_re.finditer(component_source):
        name = match.group(1)
        # Skip React built-ins like React.Fragment
        if name not in ("React.Fragment", "Fragment", "Suspense", "React.Suspense"):
            children.add(name)
    return sorted(children)


def _extract_context_subscriptions(component_source: str) -> list[str]:
    """Find React context subscriptions via useContext calls."""
    import re

    context_re = re.compile(r"\buseContext\s*\(\s*(\w+)")
    return [match.group(1) for match in context_re.finditer(component_source)]


def _find_type_definition(type_name: str, source: str, analyzer: TreeSitterAnalyzer) -> str | None:
    """Find the full type/interface definition for a props type."""
    source_bytes = source.encode("utf-8")
    tree = analyzer.parse(source_bytes)

    def search_node(node: Node) -> str | None:
        if node.type in ("interface_declaration", "type_alias_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                if name == type_name:
                    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        for child in node.children:
            result = search_node(child)
            if result:
                return result
        return None

    return search_node(tree.root_node)
