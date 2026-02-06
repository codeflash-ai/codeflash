"""Java function and method discovery.

This module provides functionality to discover optimizable functions and methods
in Java source files using the tree-sitter parser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import FunctionFilterCriteria
from codeflash.languages.java.parser import get_java_analyzer
from codeflash.models.function_types import FunctionParent

if TYPE_CHECKING:
    from codeflash.languages.java.parser import JavaAnalyzer, JavaMethodNode

logger = logging.getLogger(__name__)


def discover_functions(
    file_path: Path, filter_criteria: FunctionFilterCriteria | None = None, analyzer: JavaAnalyzer | None = None
) -> list[FunctionToOptimize]:
    """Find all optimizable functions/methods in a Java file.

    Uses tree-sitter to parse the file and find methods that can be optimized.

    Args:
        file_path: Path to the Java file to analyze.
        filter_criteria: Optional criteria to filter functions.
        analyzer: Optional JavaAnalyzer instance (created if not provided).

    Returns:
        List of FunctionToOptimize objects for discovered functions.

    """
    criteria = filter_criteria or FunctionFilterCriteria()

    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return []

    return discover_functions_from_source(source, file_path, criteria, analyzer)


def discover_functions_from_source(
    source: str,
    file_path: Path | None = None,
    filter_criteria: FunctionFilterCriteria | None = None,
    analyzer: JavaAnalyzer | None = None,
) -> list[FunctionToOptimize]:
    """Find all optimizable functions/methods in Java source code.

    Args:
        source: The Java source code to analyze.
        file_path: Optional file path for context.
        filter_criteria: Optional criteria to filter functions.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionToOptimize objects for discovered functions.

    """
    criteria = filter_criteria or FunctionFilterCriteria()
    analyzer = analyzer or get_java_analyzer()

    try:
        # Find all methods
        methods = analyzer.find_methods(
            source,
            include_private=True,  # Include all, filter later
            include_static=True,
        )

        functions: list[FunctionToOptimize] = []

        for method in methods:
            # Apply filters
            if not _should_include_method(method, criteria, source, analyzer):
                continue

            # Build parents list
            parents: list[FunctionParent] = []
            if method.class_name:
                parents.append(FunctionParent(name=method.class_name, type="ClassDef"))

            functions.append(
                FunctionToOptimize(
                    function_name=method.name,
                    file_path=file_path or Path("unknown.java"),
                    starting_line=method.start_line,
                    ending_line=method.end_line,
                    starting_col=method.start_col,
                    ending_col=method.end_col,
                    parents=parents,
                    is_async=False,  # Java doesn't have async keyword
                    is_method=method.class_name is not None,
                    language="java",
                    doc_start_line=method.javadoc_start_line,
                )
            )

        return functions

    except Exception as e:
        logger.warning("Failed to parse Java source: %s", e)
        return []


def _should_include_method(
    method: JavaMethodNode, criteria: FunctionFilterCriteria, source: str, analyzer: JavaAnalyzer
) -> bool:
    """Check if a method should be included based on filter criteria.

    Args:
        method: The method to check.
        criteria: Filter criteria to apply.
        source: Source code for additional analysis.
        analyzer: JavaAnalyzer for additional checks.

    Returns:
        True if the method should be included.

    """
    # Skip abstract methods (no implementation to optimize)
    if method.is_abstract:
        return False

    # Skip constructors (special case - could be optimized but usually not)
    if method.name == method.class_name:
        return False

    # Check include patterns
    if criteria.include_patterns:
        import fnmatch

        if not any(fnmatch.fnmatch(method.name, pattern) for pattern in criteria.include_patterns):
            return False

    # Check exclude patterns
    if criteria.exclude_patterns:
        import fnmatch

        if any(fnmatch.fnmatch(method.name, pattern) for pattern in criteria.exclude_patterns):
            return False

    # Check require_return - void methods don't have return values
    if criteria.require_return:
        if method.return_type == "void":
            return False
        # Also check if the method actually has a return statement
        if not analyzer.has_return_statement(method, source):
            return False

    # Check include_methods - in Java, all functions in classes are methods
    if not criteria.include_methods and method.class_name is not None:
        return False

    # Check line count
    method_lines = method.end_line - method.start_line + 1
    if criteria.min_lines is not None and method_lines < criteria.min_lines:
        return False
    if criteria.max_lines is not None and method_lines > criteria.max_lines:
        return False

    return True


def discover_test_methods(file_path: Path, analyzer: JavaAnalyzer | None = None) -> list[FunctionToOptimize]:
    """Find all JUnit test methods in a Java test file.

    Looks for methods annotated with @Test, @ParameterizedTest, @RepeatedTest, etc.

    Args:
        file_path: Path to the Java test file.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionToOptimize objects for discovered test methods.

    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return []

    analyzer = analyzer or get_java_analyzer()
    source_bytes = source.encode("utf8")
    tree = analyzer.parse(source_bytes)

    test_methods: list[FunctionToOptimize] = []

    # Find methods with test annotations
    _walk_tree_for_test_methods(tree.root_node, source_bytes, file_path, test_methods, analyzer, current_class=None)

    return test_methods


def _walk_tree_for_test_methods(
    node,
    source_bytes: bytes,
    file_path: Path,
    test_methods: list[FunctionToOptimize],
    analyzer: JavaAnalyzer,
    current_class: str | None,
) -> None:
    """Recursively walk the tree to find test methods."""
    new_class = current_class

    if node.type == "class_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            new_class = analyzer.get_node_text(name_node, source_bytes)

    if node.type == "method_declaration":
        # Check for test annotations
        has_test_annotation = False
        for child in node.children:
            if child.type == "modifiers":
                for mod_child in child.children:
                    if mod_child.type in {"marker_annotation", "annotation"}:
                        annotation_text = analyzer.get_node_text(mod_child, source_bytes)
                        # Check for JUnit 5 test annotations
                        if any(
                            ann in annotation_text
                            for ann in ["@Test", "@ParameterizedTest", "@RepeatedTest", "@TestFactory"]
                        ):
                            has_test_annotation = True
                            break

        if has_test_annotation:
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = analyzer.get_node_text(name_node, source_bytes)

                parents: list[FunctionParent] = []
                if current_class:
                    parents.append(FunctionParent(name=current_class, type="ClassDef"))

                test_methods.append(
                    FunctionToOptimize(
                        function_name=method_name,
                        file_path=file_path,
                        starting_line=node.start_point[0] + 1,
                        ending_line=node.end_point[0] + 1,
                        starting_col=node.start_point[1],
                        ending_col=node.end_point[1],
                        parents=list(parents),
                        is_async=False,
                        is_method=current_class is not None,
                        language="java",
                    )
                )

    for child in node.children:
        _walk_tree_for_test_methods(
            child,
            source_bytes,
            file_path,
            test_methods,
            analyzer,
            current_class=new_class if node.type == "class_declaration" else current_class,
        )


def get_method_by_name(
    file_path: Path, method_name: str, class_name: str | None = None, analyzer: JavaAnalyzer | None = None
) -> FunctionToOptimize | None:
    """Find a specific method by name in a Java file.

    Args:
        file_path: Path to the Java file.
        method_name: Name of the method to find.
        class_name: Optional class name to narrow the search.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        FunctionToOptimize for the method, or None if not found.

    """
    functions = discover_functions(file_path, analyzer=analyzer)

    for func in functions:
        if func.function_name == method_name:
            if class_name is None or func.class_name == class_name:
                return func

    return None


def get_class_methods(
    file_path: Path, class_name: str, analyzer: JavaAnalyzer | None = None
) -> list[FunctionToOptimize]:
    """Get all methods in a specific class.

    Args:
        file_path: Path to the Java file.
        class_name: Name of the class.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of FunctionToOptimize objects for methods in the class.

    """
    functions = discover_functions(file_path, analyzer=analyzer)
    return [f for f in functions if f.class_name == class_name]
