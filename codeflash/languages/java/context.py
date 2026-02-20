"""Java code context extraction.

This module provides functionality to extract code context needed for
optimization, including the target function, helper functions, imports,
and other dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash.code_utils.code_utils import encoded_tokens_len
from codeflash.languages.base import CodeContext, HelperFunction, Language
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.import_resolver import JavaImportResolver, find_helper_files
from codeflash.languages.java.parser import get_java_analyzer

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.parser import JavaAnalyzer, JavaMethodNode

logger = logging.getLogger(__name__)


class InvalidJavaSyntaxError(Exception):
    """Raised when extracted Java code is not syntactically valid."""


def extract_code_context(
    function: FunctionToOptimize,
    project_root: Path,
    module_root: Path | None = None,
    max_helper_depth: int = 2,
    analyzer: JavaAnalyzer | None = None,
    validate_syntax: bool = True,
) -> CodeContext:
    """Extract code context for a Java function.

    This extracts:
    - The target function's source code (wrapped in class/interface/enum skeleton)
    - Import statements
    - Helper functions (project-internal dependencies)
    - Read-only context (only if not already in the skeleton)

    Args:
        function: The function to extract context for.
        project_root: Root of the project.
        module_root: Root of the module (defaults to project_root).
        max_helper_depth: Maximum depth to trace helper functions.
        analyzer: Optional JavaAnalyzer instance.
        validate_syntax: Whether to validate the extracted code syntax.

    Returns:
        CodeContext with target code and dependencies.

    Raises:
        InvalidJavaSyntaxError: If validate_syntax=True and the extracted code is invalid.

    """
    analyzer = analyzer or get_java_analyzer()
    module_root = module_root or project_root

    # Read the source file
    try:
        source = function.file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to read %s: %s", function.file_path, e)
        return CodeContext(target_code="", target_file=function.file_path, language=Language.JAVA)

    # Extract target function code using tree-sitter for resilient name-based lookup
    target_code = extract_function_source(source, function, analyzer=analyzer)

    # Track whether we wrapped in a skeleton (for read_only_context decision)
    wrapped_in_skeleton = False

    # Try to wrap the method in its parent type skeleton (class, interface, or enum)
    # This provides necessary context for optimization
    parent_type_name = _get_parent_type_name(function)
    if parent_type_name:
        type_skeleton = _extract_type_skeleton(source, parent_type_name, function.function_name, analyzer)
        if type_skeleton:
            target_code = _wrap_method_in_type_skeleton(target_code, type_skeleton)
            wrapped_in_skeleton = True

    # Extract imports
    imports = analyzer.find_imports(source)
    import_statements = [_import_to_statement(imp) for imp in imports]

    # Extract helper functions
    helper_functions = find_helper_functions(function, project_root, max_helper_depth, analyzer)

    # Extract read-only context only if fields are NOT already in the skeleton
    # Avoid duplication between target_code and read_only_context
    read_only_context = ""
    if not wrapped_in_skeleton:
        read_only_context = extract_read_only_context(source, function, analyzer)

    # Validate syntax - extracted code must always be valid Java
    if validate_syntax and target_code:
        if not analyzer.validate_syntax(target_code):
            msg = f"Extracted code for {function.function_name} is not syntactically valid Java:\n{target_code}"
            raise InvalidJavaSyntaxError(msg)

    # Extract type skeletons for project-internal imported types
    imported_type_skeletons = get_java_imported_type_skeletons(
        imports, project_root, module_root, analyzer, target_code=target_code
    )

    return CodeContext(
        target_code=target_code,
        target_file=function.file_path,
        helper_functions=helper_functions,
        read_only_context=read_only_context,
        imports=import_statements,
        language=Language.JAVA,
        imported_type_skeletons=imported_type_skeletons,
    )


def _get_parent_type_name(function: FunctionToOptimize) -> str | None:
    """Get the parent type name (class, interface, or enum) for a function.

    Args:
        function: The function to get the parent for.

    Returns:
        The parent type name, or None if not found.

    """
    # First check class_name (set for class methods)
    if function.class_name:
        return function.class_name

    # Check parents for interface/enum
    if function.parents:
        for parent in function.parents:
            if parent.type in ("ClassDef", "InterfaceDef", "EnumDef"):
                return parent.name

    return None


class TypeSkeleton:
    """Represents a type skeleton (class, interface, or enum) for wrapping methods."""

    def __init__(
        self,
        type_declaration: str,
        type_javadoc: str | None,
        fields_code: str,
        constructors_code: str,
        enum_constants: str,
        type_indent: str,
        type_kind: str,  # "class", "interface", or "enum"
        outer_type_skeleton: TypeSkeleton | None = None,
    ) -> None:
        self.type_declaration = type_declaration
        self.type_javadoc = type_javadoc
        self.fields_code = fields_code
        self.constructors_code = constructors_code
        self.enum_constants = enum_constants
        self.type_indent = type_indent
        self.type_kind = type_kind
        self.outer_type_skeleton = outer_type_skeleton


# Keep ClassSkeleton as alias for backwards compatibility
ClassSkeleton = TypeSkeleton


def _extract_type_skeleton(
    source: str, type_name: str, target_method_name: str, analyzer: JavaAnalyzer
) -> TypeSkeleton | None:
    """Extract the type skeleton (class, interface, or enum) for wrapping a method.

    This extracts the type declaration, Javadoc, fields, and constructors
    to provide context for method optimization.

    Args:
        source: The source code.
        type_name: Name of the type containing the method.
        target_method_name: Name of the target method (to exclude from skeleton).
        analyzer: JavaAnalyzer instance.

    Returns:
        TypeSkeleton object or None if type not found.

    """
    source_bytes = source.encode("utf8")
    tree = analyzer.parse(source)
    lines = source.splitlines(keepends=True)

    # Find the type declaration node (class, interface, or enum)
    type_node, type_kind = _find_type_node(tree.root_node, type_name, source_bytes)
    if not type_node:
        return None

    # Check if this is an inner type and get outer type skeleton
    outer_skeleton = _get_outer_type_skeleton(type_node, source_bytes, lines, target_method_name, analyzer)

    # Get type indentation
    type_line_idx = type_node.start_point[0]
    if type_line_idx < len(lines):
        type_line = lines[type_line_idx]
        indent = len(type_line) - len(type_line.lstrip())
        type_indent = " " * indent
    else:
        type_indent = ""

    # Extract type declaration line (modifiers, name, extends, implements)
    type_declaration = _extract_type_declaration(type_node, source_bytes, type_kind)

    # Find preceding Javadoc for type
    type_javadoc = _find_javadoc(type_node, source_bytes)

    # Extract fields, constructors, and enum constants from body
    body_node = type_node.child_by_field_name("body")
    fields_code = ""
    constructors_code = ""
    enum_constants = ""

    if body_node:
        fields_code, constructors_code, enum_constants = _extract_type_body_context(
            body_node, source_bytes, lines, target_method_name, type_kind
        )

    return TypeSkeleton(
        type_declaration=type_declaration,
        type_javadoc=type_javadoc,
        fields_code=fields_code,
        constructors_code=constructors_code,
        enum_constants=enum_constants,
        type_indent=type_indent,
        type_kind=type_kind,
        outer_type_skeleton=outer_skeleton,
    )


# Keep old function name as alias for backwards compatibility
_extract_class_skeleton = _extract_type_skeleton


def _find_type_node(node: Node, type_name: str, source_bytes: bytes) -> tuple[Node | None, str]:
    """Recursively find a type declaration node (class, interface, or enum) with the given name.

    Returns:
        Tuple of (node, type_kind) where type_kind is "class", "interface", or "enum".

    """
    type_declarations = {"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum"}

    if node.type in type_declarations:
        name_node = node.child_by_field_name("name")
        if name_node:
            node_name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
            if node_name == type_name:
                return node, type_declarations[node.type]

    for child in node.children:
        result, kind = _find_type_node(child, type_name, source_bytes)
        if result:
            return result, kind

    return None, ""


# Keep old function name for backwards compatibility
def _find_class_node(node: Node, class_name: str, source_bytes: bytes) -> Node | None:
    """Recursively find a class declaration node with the given name."""
    result, _ = _find_type_node(node, class_name, source_bytes)
    return result


def _get_outer_type_skeleton(
    inner_type_node: Node, source_bytes: bytes, lines: list[str], target_method_name: str, analyzer: JavaAnalyzer
) -> TypeSkeleton | None:
    """Get the outer type skeleton if this is an inner type.

    Args:
        inner_type_node: The inner type node.
        source_bytes: Source code as bytes.
        lines: Source code split into lines.
        target_method_name: Name of target method.
        analyzer: JavaAnalyzer instance.

    Returns:
        TypeSkeleton for the outer type, or None if not an inner type.

    """
    # Walk up to find the parent type
    parent = inner_type_node.parent
    while parent:
        if parent.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            # Found outer type - extract its skeleton
            outer_name_node = parent.child_by_field_name("name")
            if outer_name_node:
                outer_name = source_bytes[outer_name_node.start_byte : outer_name_node.end_byte].decode("utf8")

                type_declarations = {
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "enum_declaration": "enum",
                }
                outer_kind = type_declarations.get(parent.type, "class")

                # Get outer type indentation
                outer_line_idx = parent.start_point[0]
                if outer_line_idx < len(lines):
                    outer_line = lines[outer_line_idx]
                    indent = len(outer_line) - len(outer_line.lstrip())
                    outer_indent = " " * indent
                else:
                    outer_indent = ""

                outer_declaration = _extract_type_declaration(parent, source_bytes, outer_kind)
                outer_javadoc = _find_javadoc(parent, source_bytes)

                # Note: We don't include fields/constructors from outer class in the skeleton
                # to keep the context focused on the inner type
                return TypeSkeleton(
                    type_declaration=outer_declaration,
                    type_javadoc=outer_javadoc,
                    fields_code="",
                    constructors_code="",
                    enum_constants="",
                    type_indent=outer_indent,
                    type_kind=outer_kind,
                    outer_type_skeleton=None,  # Could recurse for deeply nested, but keep simple for now
                )
        parent = parent.parent

    return None


def _extract_type_declaration(type_node: Node, source_bytes: bytes, type_kind: str) -> str:
    """Extract the type declaration line (without body).

    Returns something like: "public class MyClass extends Base implements Interface"

    """
    parts: list[str] = []

    # Determine which body node type to look for
    body_types = {"class": "class_body", "interface": "interface_body", "enum": "enum_body"}
    body_type = body_types.get(type_kind, "class_body")

    for child in type_node.children:
        if child.type == body_type:
            # Stop before the body
            break
        part_text = source_bytes[child.start_byte : child.end_byte].decode("utf8")
        parts.append(part_text)

    return " ".join(parts).strip()


# Keep old function name for backwards compatibility
def _extract_class_declaration(node, source_bytes):
    return _extract_type_declaration(node, source_bytes, "class")


def _find_javadoc(node: Node, source_bytes: bytes) -> str | None:
    """Find Javadoc comment immediately preceding a node."""
    prev_sibling = node.prev_named_sibling

    if prev_sibling and prev_sibling.type == "block_comment":
        comment_text = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf8")
        if comment_text.strip().startswith("/**"):
            return comment_text

    return None


def _extract_type_body_context(
    body_node: Node, source_bytes: bytes, lines: list[str], target_method_name: str, type_kind: str
) -> tuple[str, str, str]:
    """Extract fields, constructors, and enum constants from a type body.

    Args:
        body_node: Tree-sitter node for the type body.
        source_bytes: Source code as bytes.
        lines: Source code split into lines.
        target_method_name: Name of target method to exclude.
        type_kind: Type kind ("class", "interface", or "enum").

    Returns:
        Tuple of (fields_code, constructors_code, enum_constants).

    """
    field_parts: list[str] = []
    constructor_parts: list[str] = []
    enum_constant_parts: list[str] = []

    for child in body_node.children:
        # Skip braces, semicolons, and commas
        if child.type in ("{", "}", ";", ","):
            continue

        # Handle enum constants (only for enums)
        # Extract just the constant name/text, not the whole line
        if child.type == "enum_constant" and type_kind == "enum":
            constant_text = source_bytes[child.start_byte : child.end_byte].decode("utf8")
            enum_constant_parts.append(constant_text)

        # Handle field declarations
        elif child.type == "field_declaration":
            start_line = child.start_point[0]
            end_line = child.end_point[0]

            # Check for preceding Javadoc/comment
            javadoc_start = start_line
            prev_sibling = child.prev_named_sibling
            if prev_sibling and prev_sibling.type == "block_comment":
                comment_text = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf8")
                if comment_text.strip().startswith("/**"):
                    javadoc_start = prev_sibling.start_point[0]

            field_lines = lines[javadoc_start : end_line + 1]
            field_parts.append("".join(field_lines))

        # Handle constant declarations (for interfaces)
        elif child.type == "constant_declaration" and type_kind == "interface":
            start_line = child.start_point[0]
            end_line = child.end_point[0]
            constant_lines = lines[start_line : end_line + 1]
            field_parts.append("".join(constant_lines))

        # Handle constructor declarations
        elif child.type == "constructor_declaration":
            start_line = child.start_point[0]
            end_line = child.end_point[0]

            # Check for preceding Javadoc
            javadoc_start = start_line
            prev_sibling = child.prev_named_sibling
            if prev_sibling and prev_sibling.type == "block_comment":
                comment_text = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf8")
                if comment_text.strip().startswith("/**"):
                    javadoc_start = prev_sibling.start_point[0]

            constructor_lines = lines[javadoc_start : end_line + 1]
            constructor_parts.append("".join(constructor_lines))

    fields_code = "".join(field_parts)
    constructors_code = "".join(constructor_parts)
    # Join enum constants with commas
    enum_constants = ", ".join(enum_constant_parts) if enum_constant_parts else ""

    return (fields_code, constructors_code, enum_constants)


# Keep old function name for backwards compatibility
def _extract_class_body_context(
    body_node: Node, source_bytes: bytes, lines: list[str], target_method_name: str
) -> tuple[str, str]:
    """Extract fields and constructors from a class body."""
    fields, constructors, _ = _extract_type_body_context(body_node, source_bytes, lines, target_method_name, "class")
    return (fields, constructors)


def _wrap_method_in_type_skeleton(method_code: str, skeleton: TypeSkeleton) -> str:
    """Wrap a method in its type skeleton (class, interface, or enum).

    Args:
        method_code: The method source code.
        skeleton: The type skeleton.

    Returns:
        The method wrapped in the type skeleton.

    """
    parts: list[str] = []

    # If there's an outer type, wrap in that first
    if skeleton.outer_type_skeleton:
        outer = skeleton.outer_type_skeleton
        if outer.type_javadoc:
            parts.append(outer.type_javadoc)
            parts.append("\n")
        parts.append(f"{outer.type_indent}{outer.type_declaration} {{\n")

    # Add type Javadoc if present
    if skeleton.type_javadoc:
        parts.append(skeleton.type_javadoc)
        parts.append("\n")

    # Add type declaration and opening brace
    parts.append(f"{skeleton.type_indent}{skeleton.type_declaration} {{\n")

    # For enums, add constants first
    if skeleton.enum_constants:
        # Calculate method indentation (one level deeper than type)
        method_indent = skeleton.type_indent + "    "
        parts.append(f"{method_indent}{skeleton.enum_constants};\n")
        parts.append("\n")  # Blank line after enum constants

    # Add fields if present
    if skeleton.fields_code:
        parts.append(skeleton.fields_code)
        if not skeleton.fields_code.endswith("\n"):
            parts.append("\n")

    # Add constructors if present
    if skeleton.constructors_code:
        parts.append(skeleton.constructors_code)
        if not skeleton.constructors_code.endswith("\n"):
            parts.append("\n")

    # Add blank line before method if there were fields or constructors
    if skeleton.fields_code or skeleton.constructors_code or skeleton.enum_constants:
        # Check if the method code doesn't already start with a blank line
        if method_code and not method_code.lstrip().startswith("\n"):
            # The fields/constructors already have their own newline, just ensure separation
            pass

    # Add the target method
    parts.append(method_code)
    if not method_code.endswith("\n"):
        parts.append("\n")

    # Add closing brace for this type
    parts.append(f"{skeleton.type_indent}}}\n")

    # Close outer type if present
    if skeleton.outer_type_skeleton:
        parts.append(f"{skeleton.outer_type_skeleton.type_indent}}}\n")

    return "".join(parts)


# Keep old function name for backwards compatibility
_wrap_method_in_class_skeleton = _wrap_method_in_type_skeleton


def extract_function_source(source: str, function: FunctionToOptimize, analyzer: JavaAnalyzer | None = None) -> str:
    """Extract the source code of a function from the full file source.

    Uses tree-sitter to locate the function by name in the current source,
    which is resilient to file modifications (e.g., when a prior optimization
    in --all mode changed line counts in the same file). Falls back to
    pre-computed line numbers if tree-sitter lookup fails.

    Args:
        source: The full file source code.
        function: The function to extract.
        analyzer: Optional JavaAnalyzer for tree-sitter based lookup.

    Returns:
        The function's source code.

    """
    # Try tree-sitter based extraction first — resilient to stale line numbers
    if analyzer is not None:
        result = _extract_function_source_by_name(source, function, analyzer)
        if result is not None:
            return result

    # Fallback: use pre-computed line numbers
    return _extract_function_source_by_lines(source, function)


def _extract_function_source_by_name(source: str, function: FunctionToOptimize, analyzer: JavaAnalyzer) -> str | None:
    """Extract function source using tree-sitter to find the method by name.

    This re-parses the source and finds the method by name and class,
    so it works correctly even if the file has been modified since
    the function was originally discovered.

    Args:
        source: The full file source code.
        function: The function to extract.
        analyzer: JavaAnalyzer for parsing.

    Returns:
        The function's source code including Javadoc, or None if not found.

    """
    methods = analyzer.find_methods(source)
    lines = source.splitlines(keepends=True)

    # Find matching methods by name and class
    matching = [
        m
        for m in methods
        if m.name == function.function_name and (function.class_name is None or m.class_name == function.class_name)
    ]

    if not matching:
        logger.debug(
            "Tree-sitter lookup failed: no method '%s' (class=%s) found in source",
            function.function_name,
            function.class_name,
        )
        return None

    if len(matching) == 1:
        method = matching[0]
    else:
        # Multiple overloads — use original line number as proximity hint
        method = _find_closest_overload(matching, function.starting_line)

    # Determine start line (include Javadoc if present)
    start_line = method.javadoc_start_line or method.start_line
    end_line = method.end_line

    # Convert from 1-indexed to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    return "".join(lines[start_idx:end_idx])


def _find_closest_overload(methods: list[JavaMethodNode], original_start_line: int | None) -> JavaMethodNode:
    """Pick the overload whose start_line is closest to the original."""
    if not original_start_line:
        return methods[0]

    return min(methods, key=lambda m: abs(m.start_line - original_start_line))


def _extract_function_source_by_lines(source: str, function: FunctionToOptimize) -> str:
    """Extract function source using pre-computed line numbers (fallback)."""
    lines = source.splitlines(keepends=True)

    start_line = function.doc_start_line or function.starting_line
    end_line = function.ending_line

    # Convert from 1-indexed to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    return "".join(lines[start_idx:end_idx])


def find_helper_functions(
    function: FunctionToOptimize, project_root: Path, max_depth: int = 2, analyzer: JavaAnalyzer | None = None
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
    helper_files = find_helper_files(function.file_path, project_root, max_depth, analyzer)

    for file_path in helper_files:
        try:
            source = file_path.read_text(encoding="utf-8")
            file_functions = discover_functions_from_source(source, file_path, analyzer=analyzer)

            for func in file_functions:
                func_id = f"{file_path}:{func.qualified_name}"
                if func_id not in visited_functions:
                    visited_functions.add(func_id)

                    # Extract the function source using tree-sitter for resilient lookup
                    func_source = extract_function_source(source, func, analyzer=analyzer)

                    helpers.append(
                        HelperFunction(
                            name=func.function_name,
                            qualified_name=func.qualified_name,
                            file_path=file_path,
                            source_code=func_source,
                            start_line=func.starting_line,
                            end_line=func.ending_line,
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


def _find_same_class_helpers(function: FunctionToOptimize, analyzer: JavaAnalyzer) -> list[HelperFunction]:
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
            if method.name == function.function_name and method.class_name == function.class_name:
                target_method = method
                break

        if not target_method:
            return helpers

        # Get method calls from the target
        called_methods = set(analyzer.find_method_calls(source, target_method))

        # Add called methods from the same class as helpers
        for method in methods:
            if (
                method.name != function.function_name
                and method.class_name == function.class_name
                and method.name in called_methods
            ):
                func_source = source_bytes[method.node.start_byte : method.node.end_byte].decode("utf8")

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


def extract_read_only_context(source: str, function: FunctionToOptimize, analyzer: JavaAnalyzer) -> str:
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


def extract_class_context(file_path: Path, class_name: str, analyzer: JavaAnalyzer | None = None) -> str:
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
        logger.exception("Failed to extract class context: %s", e)
        return ""


# Maximum token budget for imported type skeletons to avoid bloating testgen context
IMPORTED_SKELETON_TOKEN_BUDGET = 4000


def _extract_type_names_from_code(code: str, analyzer: JavaAnalyzer) -> set[str]:
    """Extract type names referenced in Java code (method parameters, field types, etc.).

    Parses the code and collects type_identifier nodes to find which types
    are directly used. This is used to prioritize skeletons for types the
    target method actually references.
    """
    if not code:
        return set()

    type_names: set[str] = set()
    try:
        source_bytes = code.encode("utf8")
        tree = analyzer.parse(source_bytes)

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "type_identifier":
                name = source_bytes[node.start_byte : node.end_byte].decode("utf8")
                type_names.add(name)
            stack.extend(node.children)
    except Exception:
        pass

    return type_names


def get_java_imported_type_skeletons(
    imports: list, project_root: Path, module_root: Path | None, analyzer: JavaAnalyzer, target_code: str = ""
) -> str:
    """Extract type skeletons for project-internal imported types.

    Analogous to Python's get_imported_class_definitions() — resolves each import
    to a project file, extracts class declaration + constructors + fields + public
    method signatures, and returns them concatenated. This gives the testgen AI
    real type information instead of forcing it to hallucinate constructors.

    Types referenced in the target method (parameter types, field types used in
    the method body) are prioritized to ensure the AI always has context for
    the types it must construct in tests.

    Args:
        imports: List of JavaImportInfo objects from analyzer.find_imports().
        project_root: Root of the project.
        module_root: Root of the module (defaults to project_root).
        analyzer: JavaAnalyzer instance.
        target_code: The target method's source code (used for type prioritization).

    Returns:
        Concatenated type skeletons as a string, within token budget.

    """
    module_root = module_root or project_root
    resolver = JavaImportResolver(project_root)

    seen: set[tuple[str, str]] = set()  # (file_path_str, type_name) for dedup
    skeleton_parts: list[str] = []
    total_tokens = 0

    # Extract type names from target code for priority ordering
    priority_types = _extract_type_names_from_code(target_code, analyzer)

    # Pre-resolve all imports, expanding wildcards into individual types
    resolved_imports: list = []
    for imp in imports:
        if imp.is_wildcard:
            # Expand wildcard imports (e.g., com.aerospike.client.policy.*) into individual types
            expanded = resolver.expand_wildcard_import(imp.import_path)
            if expanded:
                resolved_imports.extend(expanded)
                logger.debug("Expanded wildcard import %s.* into %d types", imp.import_path, len(expanded))
            continue

        resolved = resolver.resolve_import(imp)

        # Skip external/unresolved imports
        if resolved.is_external or resolved.file_path is None:
            continue

        if not resolved.class_name:
            continue

        resolved_imports.append(resolved)

    # Sort: types referenced in the target method come first (priority), rest after
    if priority_types:
        resolved_imports.sort(key=lambda r: 0 if r.class_name in priority_types else 1)

    for resolved in resolved_imports:
        class_name = resolved.class_name
        if not class_name:
            continue

        dedup_key = (str(resolved.file_path), class_name)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        try:
            source = resolved.file_path.read_text(encoding="utf-8")
        except Exception:
            logger.debug("Could not read imported file %s", resolved.file_path)
            continue

        skeleton = _extract_type_skeleton(source, class_name, "", analyzer)
        if not skeleton:
            continue

        # Build a minimal skeleton string: declaration + fields + constructors + method signatures
        skeleton_str = _format_skeleton_for_context(skeleton, source, class_name, analyzer)
        if not skeleton_str:
            continue

        skeleton_tokens = encoded_tokens_len(skeleton_str)
        if total_tokens + skeleton_tokens > IMPORTED_SKELETON_TOKEN_BUDGET:
            logger.debug("Imported type skeleton token budget exceeded, stopping")
            break

        total_tokens += skeleton_tokens
        skeleton_parts.append(skeleton_str)

    return "\n\n".join(skeleton_parts)


def _extract_constructor_summaries(skeleton: TypeSkeleton) -> list[str]:
    """Extract one-line constructor signature summaries from a TypeSkeleton.

    Returns lines like "ClassName(Type1 param1, Type2 param2)" for each constructor.
    """
    if not skeleton.constructors_code:
        return []

    import re

    summaries: list[str] = []
    # Match constructor declarations: optional modifiers, then ClassName(params)
    # The pattern captures the constructor name and parameter list
    for match in re.finditer(r"(?:public|protected|private)?\s*(\w+)\s*\(([^)]*)\)", skeleton.constructors_code):
        name = match.group(1)
        params = match.group(2).strip()
        if params:
            summaries.append(f"{name}({params})")
        else:
            summaries.append(f"{name}()")

    return summaries


def _format_skeleton_for_context(skeleton: TypeSkeleton, source: str, class_name: str, analyzer: JavaAnalyzer) -> str:
    """Format a TypeSkeleton into a context string with method signatures.

    Includes: type declaration, fields, constructors, and public method signatures
    (signature only, no body).

    """
    parts: list[str] = []

    # Constructor summary header — makes constructor signatures unambiguous for the AI
    constructor_summaries = _extract_constructor_summaries(skeleton)
    if constructor_summaries:
        for summary in constructor_summaries:
            parts.append(f"// Constructors: {summary}")

    # Type declaration
    parts.append(f"{skeleton.type_declaration} {{")

    # Enum constants
    if skeleton.enum_constants:
        parts.append(f"    {skeleton.enum_constants};")

    # Fields
    if skeleton.fields_code:
        # avoid repeated strip() calls inside loop
        fields_lines = skeleton.fields_code.strip().splitlines()
        for line in fields_lines:
            parts.append(f"    {line.strip()}")

    # Constructors
    if skeleton.constructors_code:
        constructors_lines = skeleton.constructors_code.strip().splitlines()
        for line in constructors_lines:
            stripped = line.strip()
            if stripped:
                parts.append(f"    {stripped}")

    # Public method signatures (no body)
    method_sigs = _extract_public_method_signatures(source, class_name, analyzer)
    for sig in method_sigs:
        parts.append(f"    {sig};")

    parts.append("}")

    return "\n".join(parts)


def _extract_public_method_signatures(source: str, class_name: str, analyzer: JavaAnalyzer) -> list[str]:
    """Extract public method signatures (without body) from a class."""
    methods = analyzer.find_methods(source)
    signatures: list[str] = []

    if not methods:
        return signatures

    source_bytes = source.encode("utf8")

    pub_token = b"public"

    for method in methods:
        if method.class_name != class_name:
            continue

        node = method.node
        if not node:
            continue

        # Check if the method is public
        is_public = False
        sig_parts_bytes: list[bytes] = []
        # Single pass over children: detect modifiers and collect parts up to the body
        for child in node.children:
            ctype = child.type
            if ctype == "modifiers":
                # Check modifiers for 'public' using bytes to avoid decoding each time
                mod_slice = source_bytes[child.start_byte : child.end_byte]
                if pub_token in mod_slice:
                    is_public = True
                sig_parts_bytes.append(mod_slice)
                continue

            if ctype in {"block", "constructor_body"}:
                break

            sig_parts_bytes.append(source_bytes[child.start_byte : child.end_byte])

        if not is_public:
            continue

        if sig_parts_bytes:
            sig = b" ".join(sig_parts_bytes).decode("utf8").strip()
            # Skip constructors (already included via constructors_code)
            if node.type != "constructor_declaration":
                signatures.append(sig)

    return signatures
