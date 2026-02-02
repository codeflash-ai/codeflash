"""Java test discovery for JUnit 5.

This module provides functionality to discover tests that exercise
specific functions, mapping source functions to their tests.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.languages.base import FunctionInfo, TestInfo
from codeflash.languages.java.config import detect_java_project
from codeflash.languages.java.discovery import discover_test_methods
from codeflash.languages.java.parser import JavaAnalyzer, get_java_analyzer

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def discover_tests(
    test_root: Path,
    source_functions: Sequence[FunctionInfo],
    analyzer: JavaAnalyzer | None = None,
) -> dict[str, list[TestInfo]]:
    """Map source functions to their tests via static analysis.

    Uses several heuristics to match tests to functions:
    1. Test method name contains function name
    2. Test class name matches source class name
    3. Imports analysis
    4. Method call analysis in test code

    Args:
        test_root: Root directory containing tests.
        source_functions: Functions to find tests for.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Dict mapping qualified function names to lists of TestInfo.

    """
    analyzer = analyzer or get_java_analyzer()

    # Build a map of function names for quick lookup
    function_map: dict[str, FunctionInfo] = {}
    for func in source_functions:
        function_map[func.name] = func
        function_map[func.qualified_name] = func

    # Find all test files
    test_files = list(test_root.rglob("*Test.java")) + list(test_root.rglob("Test*.java"))

    # Result map
    result: dict[str, list[TestInfo]] = defaultdict(list)

    for test_file in test_files:
        try:
            test_methods = discover_test_methods(test_file, analyzer)
            source = test_file.read_text(encoding="utf-8")

            for test_method in test_methods:
                # Find which source functions this test might exercise
                matched_functions = _match_test_to_functions(
                    test_method, source, function_map, analyzer
                )

                for func_name in matched_functions:
                    result[func_name].append(
                        TestInfo(
                            test_name=test_method.name,
                            test_file=test_file,
                            test_class=test_method.class_name,
                        )
                    )

        except Exception as e:
            logger.warning("Failed to analyze test file %s: %s", test_file, e)

    return dict(result)


def _match_test_to_functions(
    test_method: FunctionInfo,
    test_source: str,
    function_map: dict[str, FunctionInfo],
    analyzer: JavaAnalyzer,
) -> list[str]:
    """Match a test method to source functions it might exercise.

    Args:
        test_method: The test method.
        test_source: Full source code of the test file.
        function_map: Map of function names to FunctionInfo.
        analyzer: JavaAnalyzer instance.

    Returns:
        List of function qualified names that this test might exercise.

    """
    matched: list[str] = []

    # Strategy 1: Test method name contains function name
    # e.g., testAdd -> add, testCalculatorAdd -> Calculator.add
    test_name_lower = test_method.name.lower()

    for func_name, func_info in function_map.items():
        if func_info.name.lower() in test_name_lower:
            matched.append(func_info.qualified_name)

    # Strategy 2: Method call analysis
    # Look for direct method calls in the test code
    source_bytes = test_source.encode("utf8")
    tree = analyzer.parse(source_bytes)

    # Find method calls within the test method's line range
    method_calls = _find_method_calls_in_range(
        tree.root_node,
        source_bytes,
        test_method.start_line,
        test_method.end_line,
        analyzer,
    )

    for call_name in method_calls:
        if call_name in function_map:
            qualified = function_map[call_name].qualified_name
            if qualified not in matched:
                matched.append(qualified)

    # Strategy 3: Test class naming convention
    # e.g., CalculatorTest tests Calculator
    if test_method.class_name:
        # Remove "Test" suffix or prefix
        source_class_name = test_method.class_name
        if source_class_name.endswith("Test"):
            source_class_name = source_class_name[:-4]
        elif source_class_name.startswith("Test"):
            source_class_name = source_class_name[4:]

        # Look for functions in the matching class
        for func_name, func_info in function_map.items():
            if func_info.class_name == source_class_name:
                if func_info.qualified_name not in matched:
                    matched.append(func_info.qualified_name)

    # Strategy 4: Import-based matching
    # If the test file imports a class containing the target function, consider it a match
    # This handles cases like TestQueryBlob importing Buffer and calling Buffer methods
    imported_classes = _extract_imports(tree.root_node, source_bytes, analyzer)

    for func_name, func_info in function_map.items():
        if func_info.qualified_name in matched:
            continue

        # Check if the function's class is imported
        if func_info.class_name and func_info.class_name in imported_classes:
            matched.append(func_info.qualified_name)

    return matched


def _extract_imports(
    node,
    source_bytes: bytes,
    analyzer: JavaAnalyzer,
) -> set[str]:
    """Extract imported class names from a Java file.

    Args:
        node: Tree-sitter root node.
        source_bytes: Source code as bytes.
        analyzer: JavaAnalyzer instance.

    Returns:
        Set of imported class names (simple names, not fully qualified).

    """
    imports: set[str] = set()

    def visit(n):
        if n.type == "import_declaration":
            # Get the full import path
            for child in n.children:
                if child.type == "scoped_identifier" or child.type == "identifier":
                    import_path = analyzer.get_node_text(child, source_bytes)
                    # Extract just the class name (last part)
                    # e.g., "com.example.Buffer" -> "Buffer"
                    if "." in import_path:
                        class_name = import_path.rsplit(".", 1)[-1]
                    else:
                        class_name = import_path
                    # Skip wildcard imports (*)
                    if class_name != "*":
                        imports.add(class_name)

        for child in n.children:
            visit(child)

    visit(node)
    return imports


def _find_method_calls_in_range(
    node,
    source_bytes: bytes,
    start_line: int,
    end_line: int,
    analyzer: JavaAnalyzer,
) -> list[str]:
    """Find method calls within a line range.

    Args:
        node: Tree-sitter node to search.
        source_bytes: Source code as bytes.
        start_line: Start line (1-indexed).
        end_line: End line (1-indexed).
        analyzer: JavaAnalyzer instance.

    Returns:
        List of method names called.

    """
    calls: list[str] = []

    # Check if this node is within the range (convert to 0-indexed)
    node_start = node.start_point[0] + 1
    node_end = node.end_point[0] + 1

    if node_end < start_line or node_start > end_line:
        return calls

    if node.type == "method_invocation":
        name_node = node.child_by_field_name("name")
        if name_node:
            calls.append(analyzer.get_node_text(name_node, source_bytes))

    for child in node.children:
        calls.extend(
            _find_method_calls_in_range(child, source_bytes, start_line, end_line, analyzer)
        )

    return calls


def find_tests_for_function(
    function: FunctionInfo,
    test_root: Path,
    analyzer: JavaAnalyzer | None = None,
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


def get_test_class_for_source_class(
    source_class_name: str,
    test_root: Path,
) -> Path | None:
    """Find the test class file for a source class.

    Args:
        source_class_name: Name of the source class.
        test_root: Root directory containing tests.

    Returns:
        Path to the test file, or None if not found.

    """
    # Try common naming patterns
    patterns = [
        f"{source_class_name}Test.java",
        f"Test{source_class_name}.java",
        f"{source_class_name}Tests.java",
    ]

    for pattern in patterns:
        matches = list(test_root.rglob(pattern))
        if matches:
            return matches[0]

    return None


def discover_all_tests(
    test_root: Path,
    analyzer: JavaAnalyzer | None = None,
) -> list[FunctionInfo]:
    """Discover all test methods in a test directory.

    Args:
        test_root: Root directory containing tests.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionInfo for all test methods.

    """
    analyzer = analyzer or get_java_analyzer()
    all_tests: list[FunctionInfo] = []

    # Find all test files
    test_files = list(test_root.rglob("*Test.java")) + list(test_root.rglob("Test*.java"))

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
    if name.endswith("Test.java") or name.endswith("Tests.java"):
        return True
    if name.startswith("Test") and name.endswith(".java"):
        return True

    # Check if it's in a test directory
    path_parts = file_path.parts
    for part in path_parts:
        if part in ("test", "tests", "src/test"):
            return True

    return False


def get_test_methods_for_class(
    test_file: Path,
    test_class_name: str | None = None,
    analyzer: JavaAnalyzer | None = None,
) -> list[FunctionInfo]:
    """Get all test methods in a specific test class.

    Args:
        test_file: Path to the test file.
        test_class_name: Optional class name to filter (uses file name if not provided).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionInfo for test methods.

    """
    tests = discover_test_methods(test_file, analyzer)

    if test_class_name:
        return [t for t in tests if t.class_name == test_class_name]

    return tests


def build_test_mapping_for_project(
    project_root: Path,
    analyzer: JavaAnalyzer | None = None,
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

    source_functions: list[FunctionInfo] = []
    for java_file in config.source_root.rglob("*.java"):
        funcs = discover_functions(java_file, analyzer=analyzer)
        source_functions.extend(funcs)

    # Map tests to functions
    return discover_tests(config.test_root, source_functions, analyzer)
