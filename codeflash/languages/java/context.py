"""Java code context extraction.

This module provides functionality to extract code context needed for
optimization, including the target function, helper functions, imports,
and other dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.languages.base import CodeContext, FunctionInfo, HelperFunction, Language
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.import_resolver import JavaImportResolver, find_helper_files
from codeflash.languages.java.parser import JavaAnalyzer, get_java_analyzer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def extract_code_context(
    function: FunctionInfo,
    project_root: Path,
    module_root: Path | None = None,
    max_helper_depth: int = 2,
    analyzer: JavaAnalyzer | None = None,
) -> CodeContext:
    """Extract code context for a Java function.

    This extracts:
    - The target function's source code
    - Import statements
    - Helper functions (project-internal dependencies)
    - Read-only context (class fields, constants, etc.)

    Args:
        function: The function to extract context for.
        project_root: Root of the project.
        module_root: Root of the module (defaults to project_root).
        max_helper_depth: Maximum depth to trace helper functions.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        CodeContext with target code and dependencies.

    """
    analyzer = analyzer or get_java_analyzer()
    module_root = module_root or project_root

    # Read the source file
    try:
        source = function.file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read %s: %s", function.file_path, e)
        return CodeContext(
            target_code="",
            target_file=function.file_path,
            language=Language.JAVA,
        )

    # Extract target function code
    target_code = extract_function_source(source, function)

    # Extract imports
    imports = analyzer.find_imports(source)
    import_statements = [_import_to_statement(imp) for imp in imports]

    # Extract helper functions
    helper_functions = find_helper_functions(
        function, project_root, max_helper_depth, analyzer
    )

    # Extract read-only context (class fields, constants, etc.)
    read_only_context = extract_read_only_context(source, function, analyzer)

    return CodeContext(
        target_code=target_code,
        target_file=function.file_path,
        helper_functions=helper_functions,
        read_only_context=read_only_context,
        imports=import_statements,
        language=Language.JAVA,
    )


def extract_function_source(source: str, function: FunctionInfo) -> str:
    """Extract the source code of a function from the full file source.

    Args:
        source: The full file source code.
        function: The function to extract.

    Returns:
        The function's source code.

    """
    lines = source.splitlines(keepends=True)

    # Include Javadoc if present
    start_line = function.doc_start_line or function.start_line
    end_line = function.end_line

    # Convert from 1-indexed to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    return "".join(lines[start_idx:end_idx])


def find_helper_functions(
    function: FunctionInfo,
    project_root: Path,
    max_depth: int = 2,
    analyzer: JavaAnalyzer | None = None,
) -> list[HelperFunction]:
    """Find helper functions that the target function depends on.

    Args:
        function: The target function to analyze.
        project_root: Root of the project.
        max_depth: Maximum depth to trace dependencies.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of HelperFunction objects.

    """
    analyzer = analyzer or get_java_analyzer()
    helpers: list[HelperFunction] = []
    visited_functions: set[str] = set()

    # Find helper files through imports
    helper_files = find_helper_files(
        function.file_path, project_root, max_depth, analyzer
    )

    for file_path, class_names in helper_files.items():
        try:
            source = file_path.read_text(encoding="utf-8")
            file_functions = discover_functions_from_source(source, file_path, analyzer=analyzer)

            for func in file_functions:
                func_id = f"{file_path}:{func.qualified_name}"
                if func_id not in visited_functions:
                    visited_functions.add(func_id)

                    # Extract the function source
                    func_source = extract_function_source(source, func)

                    helpers.append(
                        HelperFunction(
                            name=func.name,
                            qualified_name=func.qualified_name,
                            file_path=file_path,
                            source_code=func_source,
                            start_line=func.start_line,
                            end_line=func.end_line,
                        )
                    )

        except Exception as e:
            logger.warning("Failed to extract helpers from %s: %s", file_path, e)

    # Also find helper methods in the same class
    same_file_helpers = _find_same_class_helpers(function, analyzer)
    for helper in same_file_helpers:
        func_id = f"{function.file_path}:{helper.qualified_name}"
        if func_id not in visited_functions:
            visited_functions.add(func_id)
            helpers.append(helper)

    return helpers


def _find_same_class_helpers(
    function: FunctionInfo,
    analyzer: JavaAnalyzer,
) -> list[HelperFunction]:
    """Find helper methods in the same class as the target function.

    Args:
        function: The target function.
        analyzer: JavaAnalyzer instance.

    Returns:
        List of helper functions in the same class.

    """
    helpers: list[HelperFunction] = []

    if not function.class_name:
        return helpers

    try:
        source = function.file_path.read_text(encoding="utf-8")
        source_bytes = source.encode("utf8")

        # Find all methods in the file
        methods = analyzer.find_methods(source)

        # Find which methods the target function calls
        target_method = None
        for method in methods:
            if method.name == function.name and method.class_name == function.class_name:
                target_method = method
                break

        if not target_method:
            return helpers

        # Get method calls from the target
        called_methods = set(analyzer.find_method_calls(source, target_method))

        # Add called methods from the same class as helpers
        for method in methods:
            if (
                method.name != function.name
                and method.class_name == function.class_name
                and method.name in called_methods
            ):
                func_source = source_bytes[
                    method.node.start_byte : method.node.end_byte
                ].decode("utf8")

                helpers.append(
                    HelperFunction(
                        name=method.name,
                        qualified_name=f"{method.class_name}.{method.name}",
                        file_path=function.file_path,
                        source_code=func_source,
                        start_line=method.start_line,
                        end_line=method.end_line,
                    )
                )

    except Exception as e:
        logger.warning("Failed to find same-class helpers: %s", e)

    return helpers


def extract_read_only_context(
    source: str,
    function: FunctionInfo,
    analyzer: JavaAnalyzer,
) -> str:
    """Extract read-only context (fields, constants, inner classes).

    This extracts class-level context that the function might depend on
    but shouldn't be modified during optimization.

    Args:
        source: The full source code.
        function: The target function.
        analyzer: JavaAnalyzer instance.

    Returns:
        String containing read-only context code.

    """
    if not function.class_name:
        return ""

    context_parts: list[str] = []

    # Find fields in the same class
    fields = analyzer.find_fields(source, function.class_name)
    for field in fields:
        context_parts.append(field.source_text)

    return "\n".join(context_parts)


def _import_to_statement(import_info) -> str:
    """Convert a JavaImportInfo to an import statement string.

    Args:
        import_info: The import info.

    Returns:
        Import statement string.

    """
    if import_info.is_static:
        prefix = "import static "
    else:
        prefix = "import "

    suffix = ".*" if import_info.is_wildcard else ""

    return f"{prefix}{import_info.import_path}{suffix};"


def extract_class_context(
    file_path: Path,
    class_name: str,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Extract the full context of a class.

    Args:
        file_path: Path to the Java file.
        class_name: Name of the class.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        String containing the class code with imports.

    """
    analyzer = analyzer or get_java_analyzer()

    try:
        source = file_path.read_text(encoding="utf-8")

        # Find the class
        classes = analyzer.find_classes(source)
        target_class = None
        for cls in classes:
            if cls.name == class_name:
                target_class = cls
                break

        if not target_class:
            return ""

        # Extract imports
        imports = analyzer.find_imports(source)
        import_statements = [_import_to_statement(imp) for imp in imports]

        # Get package
        package = analyzer.get_package_name(source)
        package_stmt = f"package {package};\n\n" if package else ""

        # Get class source
        class_source = target_class.source_text

        return package_stmt + "\n".join(import_statements) + "\n\n" + class_source

    except Exception as e:
        logger.error("Failed to extract class context: %s", e)
        return ""
