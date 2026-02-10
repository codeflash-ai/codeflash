"""Java test discovery for JUnit 5.

This module provides functionality to discover tests that exercise
specific functions, mapping source functions to their tests.

The core matching strategy traces method invocations in test code back to their
declaring class by resolving variable types from declarations, field types, static
imports, and constructor expressions. This is analogous to how Python test discovery
uses jedi's "goto" functionality.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from codeflash.languages.base import TestInfo
from codeflash.languages.java.config import detect_java_project
from codeflash.languages.java.discovery import discover_test_methods
from codeflash.languages.java.parser import get_java_analyzer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tree_sitter import Node

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.parser import JavaAnalyzer

logger = logging.getLogger(__name__)


def discover_tests(
    test_root: Path, source_functions: Sequence[FunctionToOptimize], analyzer: JavaAnalyzer | None = None
) -> dict[str, list[TestInfo]]:
    """Map source functions to their tests via static analysis.

    Resolves method invocations in test code back to their declaring class by
    tracing variable types, field types, static imports, and constructor calls.

    Args:
        test_root: Root directory containing tests.
        source_functions: Functions to find tests for.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Dict mapping qualified function names to lists of TestInfo.

    """
    analyzer = analyzer or get_java_analyzer()

    function_map: dict[str, FunctionToOptimize] = {}
    for func in source_functions:
        function_map[func.qualified_name] = func

    test_files = (
        list(test_root.rglob("*Test.java")) + list(test_root.rglob("*Tests.java")) + list(test_root.rglob("Test*.java"))
    )
    # Deduplicate (a file like FooTest.java could match multiple patterns)
    test_files = list(dict.fromkeys(test_files))

    result: dict[str, list[TestInfo]] = defaultdict(list)

    for test_file in test_files:
        try:
            test_methods = discover_test_methods(test_file, analyzer)
            source = test_file.read_text(encoding="utf-8")

            # Pre-compute per-file context once, reuse for all test methods in this file
            source_bytes, tree, static_import_map = _compute_file_context(source, analyzer)
            field_type_cache: dict[str | None, dict[str, str]] = {}

            for test_method in test_methods:
                matched_functions = _match_test_method_with_context(
                    test_method, source_bytes, tree, static_import_map, field_type_cache, function_map, analyzer
                )

                for func_name in matched_functions:
                    result[func_name].append(
                        TestInfo(
                            test_name=test_method.function_name, test_file=test_file, test_class=test_method.class_name
                        )
                    )

        except Exception as e:
            logger.warning("Failed to analyze test file %s: %s", test_file, e)

    return dict(result)


def _compute_file_context(test_source: str, analyzer: JavaAnalyzer) -> tuple:
    """Pre-compute per-file analysis data: parse tree and static imports.

    Returns (source_bytes, tree, static_import_map).
    """
    source_bytes = test_source.encode("utf8")
    tree = analyzer.parse(source_bytes)
    static_import_map = _build_static_import_map(tree.root_node, source_bytes, analyzer)
    return source_bytes, tree, static_import_map


def _match_test_method_with_context(
    test_method: FunctionToOptimize,
    source_bytes: bytes,
    tree: object,
    static_import_map: dict[str, str],
    field_type_cache: dict[str | None, dict[str, str]],
    function_map: dict[str, FunctionToOptimize],
    analyzer: JavaAnalyzer,
) -> list[str]:
    """Match a test method using pre-computed per-file context.

    This avoids re-parsing and re-building file-level data for every test method
    in the same file. The field_type_cache is populated lazily per class name.
    """
    class_name = test_method.class_name
    if class_name not in field_type_cache:
        field_type_cache[class_name] = _build_field_type_map(tree.root_node, source_bytes, analyzer, class_name)
    field_types = field_type_cache[class_name]

    local_types = _build_local_type_map(
        tree.root_node, source_bytes, test_method.starting_line, test_method.ending_line, analyzer
    )
    # Locals shadow fields
    type_map = {**field_types, **local_types}

    resolved_calls = _resolve_method_calls_in_range(
        tree.root_node,
        source_bytes,
        test_method.starting_line,
        test_method.ending_line,
        analyzer,
        type_map,
        static_import_map,
    )

    matched: list[str] = []
    for call in resolved_calls:
        if call in function_map and call not in matched:
            matched.append(call)

    return matched


def _match_test_to_functions(
    test_method: FunctionToOptimize,
    test_source: str,
    function_map: dict[str, FunctionToOptimize],
    analyzer: JavaAnalyzer,
) -> list[str]:
    """Match a test method to source functions it exercises.

    Resolves each method invocation in the test to ClassName.methodName by:
    1. Building a variable-to-type map from local declarations and class fields.
    2. Building a static import map (method -> class).
    3. For each method_invocation, resolving the receiver to a class name.
    4. Matching resolved ClassName.methodName against the function map.

    Args:
        test_method: The test method.
        test_source: Full source code of the test file.
        function_map: Map of qualified names to FunctionToOptimize.
        analyzer: JavaAnalyzer instance.

    Returns:
        List of function qualified names that this test exercises.

    """
    source_bytes, tree, static_import_map = _compute_file_context(test_source, analyzer)
    field_type_cache: dict[str | None, dict[str, str]] = {}
    return _match_test_method_with_context(
        test_method, source_bytes, tree, static_import_map, field_type_cache, function_map, analyzer
    )


# ---------------------------------------------------------------------------
# Type resolution helpers
# ---------------------------------------------------------------------------


def _strip_generics(type_name: str) -> str:
    """Strip generic type parameters: ``List<String>`` -> ``List``."""
    idx = type_name.find("<")
    if idx != -1:
        return type_name[:idx].strip()
    return type_name.strip()


def _build_local_type_map(
    node: Node, source_bytes: bytes, start_line: int, end_line: int, analyzer: JavaAnalyzer
) -> dict[str, str]:
    """Map variable names to their declared types within a line range.

    Handles local variable declarations (including ``var`` with constructor
    initializers) and enhanced-for loop variables.
    """
    type_map: dict[str, str] = {}

    def _infer_var_type(declarator: Node) -> str | None:
        value_node = declarator.child_by_field_name("value")
        if value_node is None:
            return None
        if value_node.type == "object_creation_expression":
            type_node = value_node.child_by_field_name("type")
            if type_node:
                return _strip_generics(analyzer.get_node_text(type_node, source_bytes))
        return None

    def visit(n: Node) -> None:
        n_start = n.start_point[0] + 1
        n_end = n.end_point[0] + 1
        if n_end < start_line or n_start > end_line:
            return

        if n.type == "local_variable_declaration":
            type_node = n.child_by_field_name("type")
            if type_node:
                type_name = _strip_generics(analyzer.get_node_text(type_node, source_bytes))
                for child in n.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            var_name = analyzer.get_node_text(name_node, source_bytes)
                            if type_name == "var":
                                resolved = _infer_var_type(child)
                                if resolved:
                                    type_map[var_name] = resolved
                            else:
                                type_map[var_name] = type_name

        elif n.type == "enhanced_for_statement":
            # for (Type item : iterable) -type and name are positional children
            prev_type: str | None = None
            for child in n.children:
                if child.type in ("type_identifier", "generic_type", "scoped_type_identifier", "array_type"):
                    prev_type = _strip_generics(analyzer.get_node_text(child, source_bytes))
                elif child.type == "identifier" and prev_type is not None:
                    type_map[analyzer.get_node_text(child, source_bytes)] = prev_type
                    prev_type = None

        elif n.type == "resource":
            # try-with-resources: try (Type res = ...) { ... }
            type_node = n.child_by_field_name("type")
            name_node = n.child_by_field_name("name")
            if type_node and name_node:
                type_map[analyzer.get_node_text(name_node, source_bytes)] = _strip_generics(
                    analyzer.get_node_text(type_node, source_bytes)
                )

        for child in n.children:
            visit(child)

    visit(node)
    return type_map


def _build_field_type_map(
    node: Node, source_bytes: bytes, analyzer: JavaAnalyzer, test_class_name: str | None
) -> dict[str, str]:
    """Map field names to their declared types for the given class."""
    type_map: dict[str, str] = {}

    def visit(n: Node, current_class: str | None = None) -> None:
        if n.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            name_node = n.child_by_field_name("name")
            if name_node:
                current_class = analyzer.get_node_text(name_node, source_bytes)

        if n.type == "field_declaration" and current_class == test_class_name:
            type_node = n.child_by_field_name("type")
            if type_node:
                type_name = _strip_generics(analyzer.get_node_text(type_node, source_bytes))
                for child in n.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            type_map[analyzer.get_node_text(name_node, source_bytes)] = type_name

        for child in n.children:
            visit(child, current_class)

    visit(node)
    return type_map


def _build_static_import_map(node: Node, source_bytes: bytes, analyzer: JavaAnalyzer) -> dict[str, str]:
    """Map statically imported member names to their declaring class.

    For ``import static com.example.Calculator.add;`` the result is
    ``{"add": "Calculator"}``.
    """
    static_map: dict[str, str] = {}

    def visit(n: Node) -> None:
        if n.type == "import_declaration":
            import_text = analyzer.get_node_text(n, source_bytes)
            if "import static" not in import_text:
                for child in n.children:
                    visit(child)
                return

            path = import_text.replace("import static", "").replace(";", "").strip()
            if path.endswith(".*") or "." not in path:
                for child in n.children:
                    visit(child)
                return

            parts = path.rsplit(".", 2)
            if len(parts) >= 2:
                member_name = parts[-1]
                class_name = parts[-2]
                if class_name and class_name[0].isupper():
                    static_map[member_name] = class_name

        for child in n.children:
            visit(child)

    visit(node)
    return static_map


def _extract_imports(node: Node, source_bytes: bytes, analyzer: JavaAnalyzer) -> set[str]:
    """Extract imported class names (simple names) from a Java file."""
    imports: set[str] = set()

    def visit(n: Node) -> None:
        if n.type == "import_declaration":
            import_text = analyzer.get_node_text(n, source_bytes)

            if import_text.rstrip(";").endswith(".*"):
                if "import static" in import_text:
                    path = import_text.replace("import static ", "").rstrip(";").rstrip(".*")
                    if "." in path:
                        class_name = path.rsplit(".", 1)[-1]
                        if class_name and class_name[0].isupper():
                            imports.add(class_name)
                return

            if "import static" in import_text:
                path = import_text.replace("import static ", "").rstrip(";")
                parts = path.rsplit(".", 2)
                if len(parts) >= 2:
                    class_name = parts[-2]
                    if class_name and class_name[0].isupper():
                        imports.add(class_name)
                return

            for child in n.children:
                if child.type in {"scoped_identifier", "identifier"}:
                    import_path = analyzer.get_node_text(child, source_bytes)
                    if "." in import_path:
                        class_name = import_path.rsplit(".", 1)[-1]
                    else:
                        class_name = import_path
                    if class_name and class_name[0].isupper():
                        imports.add(class_name)

        for child in n.children:
            visit(child)

    visit(node)
    return imports


# ---------------------------------------------------------------------------
# Method call resolution
# ---------------------------------------------------------------------------


def _resolve_method_calls_in_range(
    node: Node,
    source_bytes: bytes,
    start_line: int,
    end_line: int,
    analyzer: JavaAnalyzer,
    type_map: dict[str, str],
    static_import_map: dict[str, str],
) -> set[str]:
    """Resolve method invocations and constructor calls within a line range.

    Returns resolved references as ``ClassName.methodName`` strings.

    Handles method invocations:
    - ``variable.method()`` - looks up variable type in *type_map*.
    - ``ClassName.staticMethod()`` - uppercase-first identifier treated as class.
    - ``new ClassName().method()`` - extracts type from constructor.
    - ``((ClassName) expr).method()`` - extracts type from cast.
    - ``this.field.method()`` - resolves field type via *type_map*.
    - ``method()`` with no receiver - checks *static_import_map*.

    Handles constructor calls:
    - ``new ClassName(...)`` - emits ``ClassName.ClassName`` and ``ClassName.<init>``.
    """
    resolved: set[str] = set()

    def _type_from_object_node(obj: Node) -> str | None:
        """Try to determine the class name from a method invocation's object."""
        if obj.type == "identifier":
            text = analyzer.get_node_text(obj, source_bytes)
            if text in type_map:
                return type_map[text]
            # Uppercase-first identifier without a type mapping → likely a class (static call)
            if text and text[0].isupper():
                return text
            return None

        if obj.type == "object_creation_expression":
            type_node = obj.child_by_field_name("type")
            if type_node:
                return _strip_generics(analyzer.get_node_text(type_node, source_bytes))
            return None

        if obj.type == "field_access":
            # this.field → look up field in type_map
            field_node = obj.child_by_field_name("field")
            obj_child = obj.child_by_field_name("object")
            if field_node and obj_child:
                field_name = analyzer.get_node_text(field_node, source_bytes)
                if obj_child.type == "this" and field_name in type_map:
                    return type_map[field_name]
            return None

        if obj.type == "parenthesized_expression":
            # Unwrap parentheses, look for cast_expression
            for child in obj.children:
                if child.type == "cast_expression":
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        return _strip_generics(analyzer.get_node_text(type_node, source_bytes))
            return None

        return None

    def visit(n: Node) -> None:
        n_start = n.start_point[0] + 1
        n_end = n.end_point[0] + 1
        if n_end < start_line or n_start > end_line:
            return

        if n.type == "method_invocation":
            name_node = n.child_by_field_name("name")
            object_node = n.child_by_field_name("object")

            if name_node:
                method_name = analyzer.get_node_text(name_node, source_bytes)

                if object_node:
                    class_name = _type_from_object_node(object_node)
                    if class_name:
                        resolved.add(f"{class_name}.{method_name}")
                # No receiver - check static imports
                elif method_name in static_import_map:
                    resolved.add(f"{static_import_map[method_name]}.{method_name}")

        elif n.type == "object_creation_expression":
            # Constructor call: new ClassName(...)
            # Emit both common qualified-name conventions so the function_map
            # can use either ClassName.ClassName or ClassName.<init>.
            type_node = n.child_by_field_name("type")
            if type_node:
                class_name = _strip_generics(analyzer.get_node_text(type_node, source_bytes))
                resolved.add(f"{class_name}.{class_name}")
                resolved.add(f"{class_name}.<init>")

        for child in n.children:
            visit(child)

    visit(node)
    return resolved


def _find_method_calls_in_range(
    node: Node, source_bytes: bytes, start_line: int, end_line: int, analyzer: JavaAnalyzer
) -> list[str]:
    """Find bare method call names within a line range (legacy helper)."""
    calls: list[str] = []

    node_start = node.start_point[0] + 1
    node_end = node.end_point[0] + 1

    if node_end < start_line or node_start > end_line:
        return calls

    if node.type == "method_invocation":
        name_node = node.child_by_field_name("name")
        if name_node:
            calls.append(analyzer.get_node_text(name_node, source_bytes))

    for child in node.children:
        calls.extend(_find_method_calls_in_range(child, source_bytes, start_line, end_line, analyzer))

    return calls


def find_tests_for_function(
    function: FunctionToOptimize, test_root: Path, analyzer: JavaAnalyzer | None = None
) -> list[TestInfo]:
    """Find tests that exercise a specific function.

    Args:
        function: The function to find tests for.
        test_root: Root directory containing tests.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of TestInfo for tests that might exercise this function.

    """
    result = discover_tests(test_root, [function], analyzer)
    return result.get(function.qualified_name, [])


def get_test_class_for_source_class(source_class_name: str, test_root: Path) -> Path | None:
    """Find the test class file for a source class.

    Args:
        source_class_name: Name of the source class.
        test_root: Root directory containing tests.

    Returns:
        Path to the test file, or None if not found.

    """
    # Try common naming patterns
    patterns = [f"{source_class_name}Test.java", f"Test{source_class_name}.java", f"{source_class_name}Tests.java"]

    for pattern in patterns:
        matches = list(test_root.rglob(pattern))
        if matches:
            return matches[0]

    return None


def discover_all_tests(test_root: Path, analyzer: JavaAnalyzer | None = None) -> list[FunctionToOptimize]:
    """Discover all test methods in a test directory.

    Args:
        test_root: Root directory containing tests.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionToOptimize for all test methods.

    """
    analyzer = analyzer or get_java_analyzer()
    all_tests: list[FunctionToOptimize] = []

    # Find all test files (various naming conventions)
    test_files = (
        list(test_root.rglob("*Test.java")) + list(test_root.rglob("*Tests.java")) + list(test_root.rglob("Test*.java"))
    )

    for test_file in test_files:
        try:
            tests = discover_test_methods(test_file, analyzer)
            all_tests.extend(tests)
        except Exception as e:
            logger.warning("Failed to analyze test file %s: %s", test_file, e)

    return all_tests


def get_test_file_suffix() -> str:
    """Get the test file suffix for Java.

    Returns:
        Test file suffix.

    """
    return "Test.java"


def is_test_file(file_path: Path) -> bool:
    """Check if a file is a test file.

    Args:
        file_path: Path to check.

    Returns:
        True if this appears to be a test file.

    """
    name = file_path.name

    # Check naming patterns
    if name.endswith(("Test.java", "Tests.java")):
        return True
    if name.startswith("Test") and name.endswith(".java"):
        return True

    # Check if it's in a test directory
    path_parts = file_path.parts
    return any(part in ("test", "tests", "src/test") for part in path_parts)


def get_test_methods_for_class(
    test_file: Path, test_class_name: str | None = None, analyzer: JavaAnalyzer | None = None
) -> list[FunctionToOptimize]:
    """Get all test methods in a specific test class.

    Args:
        test_file: Path to the test file.
        test_class_name: Optional class name to filter (uses file name if not provided).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionToOptimize for test methods.

    """
    tests = discover_test_methods(test_file, analyzer)

    if test_class_name:
        return [t for t in tests if t.class_name == test_class_name]

    return tests


def build_test_mapping_for_project(
    project_root: Path, analyzer: JavaAnalyzer | None = None
) -> dict[str, list[TestInfo]]:
    """Build a complete test mapping for a project.

    Args:
        project_root: Root directory of the project.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Dict mapping qualified function names to lists of TestInfo.

    """
    analyzer = analyzer or get_java_analyzer()

    # Detect project configuration
    config = detect_java_project(project_root)
    if not config:
        return {}

    if not config.source_root or not config.test_root:
        return {}

    # Discover all source functions
    from codeflash.languages.java.discovery import discover_functions

    source_functions: list[FunctionToOptimize] = []
    for java_file in config.source_root.rglob("*.java"):
        funcs = discover_functions(java_file, analyzer=analyzer)
        source_functions.extend(funcs)

    # Map tests to functions
    return discover_tests(config.test_root, source_functions, analyzer)
