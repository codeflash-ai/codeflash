"""Tree-sitter utilities for cross-language code analysis.

This module provides a unified interface for parsing and analyzing code
across multiple languages using tree-sitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


class TreeSitterLanguage(Enum):
    """Supported tree-sitter languages."""

    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"


# Lazy-loaded language instances
_LANGUAGE_CACHE: dict[TreeSitterLanguage, Language] = {}


def _get_language(lang: TreeSitterLanguage) -> Language:
    """Get a tree-sitter Language instance, with lazy loading."""
    if lang not in _LANGUAGE_CACHE:
        if lang == TreeSitterLanguage.JAVASCRIPT:
            import tree_sitter_javascript

            _LANGUAGE_CACHE[lang] = Language(tree_sitter_javascript.language())
        elif lang == TreeSitterLanguage.TYPESCRIPT:
            import tree_sitter_typescript

            _LANGUAGE_CACHE[lang] = Language(tree_sitter_typescript.language_typescript())
        elif lang == TreeSitterLanguage.TSX:
            import tree_sitter_typescript

            _LANGUAGE_CACHE[lang] = Language(tree_sitter_typescript.language_tsx())
    return _LANGUAGE_CACHE[lang]


@dataclass
class FunctionNode:
    """Represents a function found by tree-sitter analysis."""

    name: str
    node: Node
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_async: bool
    is_method: bool
    is_arrow: bool
    is_generator: bool
    class_name: str | None
    parent_function: str | None
    source_text: str
    doc_start_line: int | None = None  # Line where JSDoc comment starts (or None if no JSDoc)
    is_exported: bool = False  # Whether the function is exported


@dataclass
class ImportInfo:
    """Represents an import statement."""

    module_path: str  # The path being imported from
    default_import: str | None  # Default import name (import X from ...)
    named_imports: list[tuple[str, str | None]]  # [(name, alias), ...]
    namespace_import: str | None  # Namespace import (import * as X from ...)
    is_type_only: bool  # TypeScript type-only import
    start_line: int
    end_line: int


@dataclass
class ExportInfo:
    """Represents an export statement."""

    exported_names: list[tuple[str, str | None]]  # [(name, alias), ...] for named exports
    default_export: str | None  # Name of default exported function/class/value
    is_reexport: bool  # Whether this is a re-export (export { x } from './other')
    reexport_source: str | None  # Module path for re-exports
    start_line: int
    end_line: int
    # Functions passed as arguments to wrapper calls in default exports
    # e.g., export default curry(traverseEntity) -> ["traverseEntity"]
    wrapped_default_args: list[str] | None = None


@dataclass
class ModuleLevelDeclaration:
    """Represents a module-level (global) variable or constant declaration."""

    name: str  # Variable/constant name
    declaration_type: str  # "const", "let", "var", "class", "enum", "type", "interface"
    source_code: str  # Full declaration source code
    start_line: int
    end_line: int
    is_exported: bool  # Whether the declaration is exported


@dataclass
class TypeDefinition:
    """Represents a type definition (interface, type alias, class, or enum)."""

    name: str  # Type name
    definition_type: str  # "interface", "type", "class", "enum"
    source_code: str  # Full definition source code
    start_line: int
    end_line: int
    is_exported: bool  # Whether the definition is exported
    file_path: Path | None = None  # File where the type is defined


class TreeSitterAnalyzer:
    """Cross-language code analysis using tree-sitter.

    This class provides methods to parse and analyze JavaScript/TypeScript code,
    finding functions, imports, and other code structures.
    """

    def __init__(self, language: TreeSitterLanguage | str) -> None:
        """Initialize the analyzer for a specific language.

        Args:
            language: The language to analyze (TreeSitterLanguage enum or string).

        """
        if isinstance(language, str):
            language = TreeSitterLanguage(language)
        self.language = language
        self._parser: Parser | None = None

    @property
    def parser(self) -> Parser:
        """Get the parser, creating it lazily."""
        if self._parser is None:
            self._parser = Parser(_get_language(self.language))
        return self._parser

    def parse(self, source: str | bytes) -> Tree:
        """Parse source code into a tree-sitter tree.

        Args:
            source: Source code as string or bytes.

        Returns:
            The parsed tree.

        """
        if isinstance(source, str):
            source_bytes = source.encode("utf8")
        else:
            source_bytes = source
        return self.parser.parse(source_bytes)

    def get_node_text(self, node: Node, source: bytes) -> str:
        """Extract the source text for a tree-sitter node.

        Args:
            node: The tree-sitter node.
            source: The source code as bytes.

        Returns:
            The text content of the node.

        """
        return source[node.start_byte : node.end_byte].decode("utf8")

    def find_functions(
        self, source: str, include_methods: bool = True, include_arrow_functions: bool = True, require_name: bool = True
    ) -> list[FunctionNode]:
        """Find all function definitions in source code.

        Args:
            source: The source code to analyze.
            include_methods: Whether to include class methods.
            include_arrow_functions: Whether to include arrow functions.
            require_name: Whether to require functions to have names.

        Returns:
            List of FunctionNode objects describing found functions.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        functions: list[FunctionNode] = []

        self._walk_tree_for_functions(
            tree.root_node,
            source_bytes,
            functions,
            include_methods=include_methods,
            include_arrow_functions=include_arrow_functions,
            require_name=require_name,
            current_class=None,
            current_function=None,
        )

        return functions

    def _walk_tree_for_functions(
        self,
        node: Node,
        source_bytes: bytes,
        functions: list[FunctionNode],
        include_methods: bool,
        include_arrow_functions: bool,
        require_name: bool,
        current_class: str | None,
        current_function: str | None,
    ) -> None:
        """Recursively walk the tree to find function definitions."""
        # Function types in JavaScript/TypeScript
        function_types = {
            "function_declaration",
            "function_expression",
            "generator_function_declaration",
            "generator_function",
        }

        if include_arrow_functions:
            function_types.add("arrow_function")

        if include_methods:
            function_types.add("method_definition")

        # Track class context
        new_class = current_class
        new_function = current_function

        if node.type in {"class_declaration", "class"}:
            # Get class name
            name_node = node.child_by_field_name("name")
            if name_node:
                new_class = self.get_node_text(name_node, source_bytes)

        if node.type in function_types:
            func_info = self._extract_function_info(node, source_bytes, current_class, current_function)

            if func_info:
                # Check if we should include this function
                should_include = True

                if require_name and not func_info.name:
                    should_include = False

                if func_info.is_method and not include_methods:
                    should_include = False

                if func_info.is_arrow and not include_arrow_functions:
                    should_include = False

                # Skip arrow functions that are object properties (e.g., { foo: () => {} })
                # These are not standalone functions - they're values in object literals
                if func_info.is_arrow and node.parent and node.parent.type == "pair":
                    should_include = False

                if should_include:
                    functions.append(func_info)

                # Track as current function for nested functions
                if func_info.name:
                    new_function = func_info.name

        # Recurse into children
        for child in node.children:
            self._walk_tree_for_functions(
                child,
                source_bytes,
                functions,
                include_methods=include_methods,
                include_arrow_functions=include_arrow_functions,
                require_name=require_name,
                current_class=new_class,
                current_function=new_function if node.type in function_types else current_function,
            )

    def _extract_function_info(
        self, node: Node, source_bytes: bytes, current_class: str | None, current_function: str | None
    ) -> FunctionNode | None:
        """Extract function information from a tree-sitter node."""
        name = ""
        is_async = False
        is_generator = False
        is_method = False
        is_arrow = node.type == "arrow_function"
        is_exported = False

        # Check for async modifier
        for child in node.children:
            if child.type == "async":
                is_async = True
                break

        # Check for generator
        if "generator" in node.type:
            is_generator = True

        # Check if function is exported
        # For function_declaration: check if parent is export_statement
        # For arrow functions: check if parent variable_declarator's grandparent is export_statement
        # For CommonJS: check module.exports = { name } or exports.name = ...
        is_exported = self._is_node_exported(node, source_bytes)

        # Get function name based on node type
        if node.type in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
            else:
                # Fallback: search for identifier child (some tree-sitter versions)
                for child in node.children:
                    if child.type == "identifier":
                        name = self.get_node_text(child, source_bytes)
                        break
        elif node.type == "method_definition":
            is_method = True
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
        elif node.type in ("function_expression", "generator_function"):
            # Check if assigned to a variable
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
            else:
                # Try to get name from parent assignment
                name = self._get_name_from_assignment(node, source_bytes)
        elif node.type == "arrow_function":
            # Arrow functions get names from variable declarations
            name = self._get_name_from_assignment(node, source_bytes)

        # Get source text
        source_text = self.get_node_text(node, source_bytes)

        # Find preceding JSDoc comment
        doc_start_line = self._find_preceding_jsdoc(node, source_bytes)

        return FunctionNode(
            name=name,
            node=node,
            start_line=node.start_point[0] + 1,  # Convert to 1-indexed
            end_line=node.end_point[0] + 1,
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            is_async=is_async,
            is_method=is_method,
            is_arrow=is_arrow,
            is_generator=is_generator,
            class_name=current_class if is_method else None,
            parent_function=current_function,
            source_text=source_text,
            doc_start_line=doc_start_line,
            is_exported=is_exported,
        )

    def _is_node_exported(self, node: Node, source_bytes: bytes | None = None) -> bool:
        """Check if a function node is exported.

        Handles various export patterns:
        - export function foo() {}
        - export const foo = () => {}
        - export default function foo() {}
        - Class methods in exported classes
        - module.exports = { foo } (CommonJS)
        - exports.foo = ... (CommonJS)

        Args:
            node: The function node to check.
            source_bytes: Source code bytes (needed for CommonJS export detection).

        Returns:
            True if the function is exported, False otherwise.

        """
        # Check direct parent for export_statement
        if node.parent and node.parent.type == "export_statement":
            return True

        # For arrow functions and function expressions assigned to variables
        # e.g., export const foo = () => {}
        if node.type in ("arrow_function", "function_expression", "generator_function"):
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                grandparent = parent.parent
                if grandparent and grandparent.type in ("lexical_declaration", "variable_declaration"):
                    great_grandparent = grandparent.parent
                    if great_grandparent and great_grandparent.type == "export_statement":
                        return True

        # For methods in exported classes
        if node.type == "method_definition":
            # Walk up to find class_declaration
            current = node.parent
            while current:
                if current.type in ("class_declaration", "class"):
                    # Check if this class is exported via ES module export
                    if current.parent and current.parent.type == "export_statement":
                        return True
                    # Check if class is exported via CommonJS
                    if source_bytes:
                        class_name_node = current.child_by_field_name("name")
                        if class_name_node:
                            class_name = self.get_node_text(class_name_node, source_bytes)
                            if self._is_name_in_commonjs_exports(node, class_name, source_bytes):
                                return True
                    break
                current = current.parent

        # Check CommonJS exports: module.exports = { foo } or exports.foo = ...
        if source_bytes:
            func_name = self._get_function_name_for_export_check(node, source_bytes)
            if func_name and self._is_name_in_commonjs_exports(node, func_name, source_bytes):
                return True

        return False

    def _get_function_name_for_export_check(self, node: Node, source_bytes: bytes) -> str | None:
        """Get the function name for export checking."""
        if node.type in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                return self.get_node_text(name_node, source_bytes)
        elif node.type in ("arrow_function", "function_expression", "generator_function"):
            # Get name from variable assignment
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node and name_node.type == "identifier":
                    return self.get_node_text(name_node, source_bytes)
        return None

    def _is_name_in_commonjs_exports(self, node: Node, name: str, source_bytes: bytes) -> bool:
        """Check if a name is exported via CommonJS module.exports or exports.

        Handles patterns like:
        - module.exports = { foo, bar }
        - module.exports = { foo: someFunc }
        - exports.foo = ...
        - module.exports.foo = ...

        Args:
            node: Any node in the tree (used to find the program root).
            name: The name to check for in exports.
            source_bytes: Source code bytes.

        Returns:
            True if the name is in CommonJS exports.

        """
        # Walk up to find program root
        root = node
        while root.parent:
            root = root.parent

        # Search for CommonJS export patterns in program children
        for child in root.children:
            if child.type == "expression_statement":
                # Look for assignment expressions
                for expr in child.children:
                    if expr.type == "assignment_expression":
                        if self._check_commonjs_assignment_exports(expr, name, source_bytes):
                            return True

        return False

    def _check_commonjs_assignment_exports(self, node: Node, name: str, source_bytes: bytes) -> bool:
        """Check if a CommonJS assignment exports the given name."""
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")

        if not left_node or not right_node:
            return False

        left_text = self.get_node_text(left_node, source_bytes)

        # Check module.exports = { name, ... } or module.exports = { key: name, ... }
        if left_text == "module.exports" and right_node.type == "object":
            for child in right_node.children:
                if child.type == "shorthand_property_identifier":
                    # { foo } - shorthand export
                    if self.get_node_text(child, source_bytes) == name:
                        return True
                elif child.type == "pair":
                    # { key: value } - check both key and value
                    key_node = child.child_by_field_name("key")
                    value_node = child.child_by_field_name("value")
                    if key_node and self.get_node_text(key_node, source_bytes) == name:
                        return True
                    if value_node and value_node.type == "identifier":
                        if self.get_node_text(value_node, source_bytes) == name:
                            return True

        # Check module.exports = name (single export)
        if left_text == "module.exports" and right_node.type == "identifier":
            if self.get_node_text(right_node, source_bytes) == name:
                return True

        # Check module.exports.name = ... or exports.name = ...
        if left_text in {f"module.exports.{name}", f"exports.{name}"}:
            return True

        return False

    def _find_preceding_jsdoc(self, node: Node, source_bytes: bytes) -> int | None:
        """Find JSDoc comment immediately preceding a function node.

        For regular functions, looks at the previous sibling of the function node.
        For arrow functions assigned to variables, looks at the previous sibling
        of the variable declaration.

        Args:
            node: The function node to find JSDoc for.
            source_bytes: The source code as bytes.

        Returns:
            The start line (1-indexed) of the JSDoc, or None if no JSDoc found.

        """
        target_node = node

        # For arrow functions, look at parent variable declaration
        if node.type == "arrow_function":
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                grandparent = parent.parent
                if grandparent and grandparent.type in ("lexical_declaration", "variable_declaration"):
                    target_node = grandparent

        # For function expressions assigned to variables, also look at parent
        if node.type in ("function_expression", "generator_function"):
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                grandparent = parent.parent
                if grandparent and grandparent.type in ("lexical_declaration", "variable_declaration"):
                    target_node = grandparent

        # Get the previous sibling node
        prev_sibling = target_node.prev_named_sibling

        # Check if it's a comment node with JSDoc pattern
        if prev_sibling and prev_sibling.type == "comment":
            comment_text = self.get_node_text(prev_sibling, source_bytes)
            if comment_text.strip().startswith("/**"):
                # Verify it's immediately preceding (no blank lines between)
                comment_end_line = prev_sibling.end_point[0]
                function_start_line = target_node.start_point[0]
                if function_start_line - comment_end_line <= 1:
                    return prev_sibling.start_point[0] + 1  # 1-indexed

        return None

    def _get_name_from_assignment(self, node: Node, source_bytes: bytes) -> str:
        """Try to extract function name from parent variable declaration or assignment.

        Handles patterns like:
        - const foo = () => {}
        - const foo = function() {}
        - let bar = function() {}
        - obj.method = () => {}
        """
        parent = node.parent
        if parent is None:
            return ""

        # Check for variable declarator: const foo = ...
        if parent.type == "variable_declarator":
            name_node = parent.child_by_field_name("name")
            if name_node:
                return self.get_node_text(name_node, source_bytes)

        # Check for assignment expression: foo = ...
        if parent.type == "assignment_expression":
            left_node = parent.child_by_field_name("left")
            if left_node:
                if left_node.type == "identifier":
                    return self.get_node_text(left_node, source_bytes)
                if left_node.type == "member_expression":
                    # For obj.method = ..., get the property name
                    prop_node = left_node.child_by_field_name("property")
                    if prop_node:
                        return self.get_node_text(prop_node, source_bytes)

        # Check for property in object: { foo: () => {} }
        if parent.type == "pair":
            key_node = parent.child_by_field_name("key")
            if key_node:
                return self.get_node_text(key_node, source_bytes)

        return ""

    def find_imports(self, source: str) -> list[ImportInfo]:
        """Find all import statements in source code.

        Args:
            source: The source code to analyze.

        Returns:
            List of ImportInfo objects describing imports.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        imports: list[ImportInfo] = []

        self._walk_tree_for_imports(tree.root_node, source_bytes, imports)

        return imports

    def _walk_tree_for_imports(
        self, node: Node, source_bytes: bytes, imports: list[ImportInfo], in_function: bool = False
    ) -> None:
        """Recursively walk the tree to find import statements.

        Args:
            node: Current node to check.
            source_bytes: Source code bytes.
            imports: List to append found imports to.
            in_function: Whether we're currently inside a function/method body.

        """
        # Track when we enter function/method bodies
        # These node types contain function/method bodies where require() should not be treated as imports
        function_body_types = {
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
            "function",  # Generic function in some grammars
        }

        if node.type == "import_statement":
            import_info = self._extract_import_info(node, source_bytes)
            if import_info:
                imports.append(import_info)

        # Also handle require() calls for CommonJS, but only at module level
        # require() inside functions is a dynamic import, not a module import
        if node.type == "call_expression" and not in_function:
            func_node = node.child_by_field_name("function")
            if func_node and self.get_node_text(func_node, source_bytes) == "require":
                import_info = self._extract_require_info(node, source_bytes)
                if import_info:
                    imports.append(import_info)

        # Update in_function flag for children
        child_in_function = in_function or node.type in function_body_types

        for child in node.children:
            self._walk_tree_for_imports(child, source_bytes, imports, child_in_function)

    def _extract_import_info(self, node: Node, source_bytes: bytes) -> ImportInfo | None:
        """Extract import information from an import statement node."""
        module_path = ""
        default_import = None
        named_imports: list[tuple[str, str | None]] = []
        namespace_import = None
        is_type_only = False

        # Get the module path (source)
        source_node = node.child_by_field_name("source")
        if source_node:
            # Remove quotes from string
            module_path = self.get_node_text(source_node, source_bytes).strip("'\"")

        # Check for type-only import (TypeScript)
        for child in node.children:
            if child.type == "type" or self.get_node_text(child, source_bytes) == "type":
                is_type_only = True
                break

        # Process import clause
        for child in node.children:
            if child.type == "import_clause":
                self._process_import_clause(child, source_bytes, default_import, named_imports, namespace_import)
                # Re-extract after processing
                for clause_child in child.children:
                    if clause_child.type == "identifier":
                        default_import = self.get_node_text(clause_child, source_bytes)
                    elif clause_child.type == "named_imports":
                        for spec in clause_child.children:
                            if spec.type == "import_specifier":
                                name_node = spec.child_by_field_name("name")
                                alias_node = spec.child_by_field_name("alias")
                                if name_node:
                                    name = self.get_node_text(name_node, source_bytes)
                                    alias = self.get_node_text(alias_node, source_bytes) if alias_node else None
                                    named_imports.append((name, alias))
                    elif clause_child.type == "namespace_import":
                        # import * as X
                        for ns_child in clause_child.children:
                            if ns_child.type == "identifier":
                                namespace_import = self.get_node_text(ns_child, source_bytes)

        if not module_path:
            return None

        return ImportInfo(
            module_path=module_path,
            default_import=default_import,
            named_imports=named_imports,
            namespace_import=namespace_import,
            is_type_only=is_type_only,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

    def _process_import_clause(
        self,
        node: Node,
        source_bytes: bytes,
        default_import: str | None,
        named_imports: list[tuple[str, str | None]],
        namespace_import: str | None,
    ) -> None:
        """Process an import clause to extract imports."""
        # This is a helper that modifies the lists in place
        # Processing is done inline in _extract_import_info

    def _extract_require_info(self, node: Node, source_bytes: bytes) -> ImportInfo | None:
        """Extract import information from a require() call.

        Handles various CommonJS require patterns:
        - const foo = require('./module')           -> default import
        - const { a, b } = require('./module')      -> named imports
        - const { a: aliasA } = require('./module') -> named imports with alias
        - const foo = require('./module').bar       -> property access (named import)
        - require('./module')                       -> side effect import
        """
        # Handle require().property pattern - the call_expression is inside member_expression
        actual_require_node = node
        property_access = None

        # Check if this require is part of a member_expression like require('./m').foo
        if node.parent and node.parent.type == "member_expression":
            member_node = node.parent
            prop_node = member_node.child_by_field_name("property")
            if prop_node:
                property_access = self.get_node_text(prop_node, source_bytes)
            # Use the member expression's parent for variable assignment lookup
            node = member_node

        args_node = actual_require_node.child_by_field_name("arguments")
        if not args_node:
            return None

        # Get the first argument (module path)
        module_path = ""
        for child in args_node.children:
            if child.type == "string":
                module_path = self.get_node_text(child, source_bytes).strip("'\"")
                break

        if not module_path:
            return None

        # Try to get the variable name from assignment
        default_import = None
        named_imports: list[tuple[str, str | None]] = []

        parent = node.parent
        if parent and parent.type == "variable_declarator":
            name_node = parent.child_by_field_name("name")
            if name_node:
                if name_node.type == "identifier":
                    var_name = self.get_node_text(name_node, source_bytes)
                    if property_access:
                        # const foo = require('./module').bar
                        # This imports 'bar' from the module and assigns to 'foo'
                        named_imports.append((property_access, var_name if var_name != property_access else None))
                    else:
                        # const foo = require('./module')
                        default_import = var_name
                elif name_node.type == "object_pattern":
                    # Destructuring: const { a, b } = require('...')
                    named_imports = self._extract_object_pattern_names(name_node, source_bytes)
        elif property_access:
            # require('./module').foo without assignment - still track the property access
            named_imports.append((property_access, None))

        return ImportInfo(
            module_path=module_path,
            default_import=default_import,
            named_imports=named_imports,
            namespace_import=None,
            is_type_only=False,
            start_line=actual_require_node.start_point[0] + 1,
            end_line=actual_require_node.end_point[0] + 1,
        )

    def _extract_object_pattern_names(self, node: Node, source_bytes: bytes) -> list[tuple[str, str | None]]:
        """Extract names from an object pattern (destructuring).

        Handles patterns like:
        - { a, b }         -> [('a', None), ('b', None)]
        - { a: aliasA }    -> [('a', 'aliasA')]
        - { a, b: aliasB } -> [('a', None), ('b', 'aliasB')]
        """
        names: list[tuple[str, str | None]] = []

        for child in node.children:
            if child.type == "shorthand_property_identifier_pattern":
                # { a } - shorthand, name equals value
                name = self.get_node_text(child, source_bytes)
                names.append((name, None))
            elif child.type == "pair_pattern":
                # { a: aliasA } - renamed import
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node and value_node:
                    original_name = self.get_node_text(key_node, source_bytes)
                    alias = self.get_node_text(value_node, source_bytes)
                    names.append((original_name, alias))

        return names

    def find_exports(self, source: str) -> list[ExportInfo]:
        """Find all export statements in source code.

        Args:
            source: The source code to analyze.

        Returns:
            List of ExportInfo objects describing exports.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        exports: list[ExportInfo] = []

        self._walk_tree_for_exports(tree.root_node, source_bytes, exports)

        return exports

    def _walk_tree_for_exports(self, node: Node, source_bytes: bytes, exports: list[ExportInfo]) -> None:
        """Recursively walk the tree to find export statements."""
        # Handle ES module export statements
        if node.type == "export_statement":
            export_info = self._extract_export_info(node, source_bytes)
            if export_info:
                exports.append(export_info)

        # Handle CommonJS exports: module.exports = ... or exports.foo = ...
        if node.type == "assignment_expression":
            export_info = self._extract_commonjs_export(node, source_bytes)
            if export_info:
                exports.append(export_info)

        for child in node.children:
            self._walk_tree_for_exports(child, source_bytes, exports)

    def _extract_export_info(self, node: Node, source_bytes: bytes) -> ExportInfo | None:
        """Extract export information from an export statement node."""
        exported_names: list[tuple[str, str | None]] = []
        default_export: str | None = None
        is_reexport = False
        reexport_source: str | None = None
        wrapped_default_args: list[str] | None = None

        # Check for re-export source (export { x } from './other')
        source_node = node.child_by_field_name("source")
        if source_node:
            is_reexport = True
            reexport_source = self.get_node_text(source_node, source_bytes).strip("'\"")

        for child in node.children:
            # Handle 'export default'
            if child.type == "default":
                # Find what's being exported as default
                for sibling in node.children:
                    if sibling.type in {"function_declaration", "class_declaration"}:
                        name_node = sibling.child_by_field_name("name")
                        default_export = self.get_node_text(name_node, source_bytes) if name_node else "default"
                    elif sibling.type == "identifier":
                        default_export = self.get_node_text(sibling, source_bytes)
                    elif sibling.type in ("arrow_function", "function_expression", "object", "array"):
                        default_export = "default"
                    elif sibling.type == "call_expression":
                        # Handle wrapped exports: export default curry(traverseEntity)
                        # The default export is the result of the call, but we track
                        # the wrapped function names for export checking
                        default_export = "default"
                        wrapped_default_args = self._extract_call_expression_identifiers(sibling, source_bytes)
                break

            # Handle named exports: export { a, b as c }
            if child.type == "export_clause":
                for spec in child.children:
                    if spec.type == "export_specifier":
                        name_node = spec.child_by_field_name("name")
                        alias_node = spec.child_by_field_name("alias")
                        if name_node:
                            name = self.get_node_text(name_node, source_bytes)
                            alias = self.get_node_text(alias_node, source_bytes) if alias_node else None
                            exported_names.append((name, alias))

            # Handle direct exports: export function foo() {}
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self.get_node_text(name_node, source_bytes)
                    exported_names.append((name, None))

            # Handle direct class exports: export class Foo {}
            if child.type == "class_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self.get_node_text(name_node, source_bytes)
                    exported_names.append((name, None))

            # Handle variable exports: export const foo = ...
            if child.type == "lexical_declaration":
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        if name_node and name_node.type == "identifier":
                            name = self.get_node_text(name_node, source_bytes)
                            exported_names.append((name, None))

        # Skip if no exports found
        if not exported_names and not default_export:
            return None

        return ExportInfo(
            exported_names=exported_names,
            default_export=default_export,
            is_reexport=is_reexport,
            reexport_source=reexport_source,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            wrapped_default_args=wrapped_default_args,
        )

    def _extract_call_expression_identifiers(self, node: Node, source_bytes: bytes) -> list[str]:
        """Extract identifier names from arguments of a call expression.

        For patterns like curry(traverseEntity) or compose(fn1, fn2), this extracts
        the function names passed as arguments: ["traverseEntity"] or ["fn1", "fn2"].

        Args:
            node: A call_expression node.
            source_bytes: The source code as bytes.

        Returns:
            List of identifier names found in the call arguments.

        """
        identifiers: list[str] = []

        # Get the arguments node
        args_node = node.child_by_field_name("arguments")
        if args_node:
            for child in args_node.children:
                if child.type == "identifier":
                    identifiers.append(self.get_node_text(child, source_bytes))
                # Also handle nested call expressions: compose(curry(fn))
                elif child.type == "call_expression":
                    identifiers.extend(self._extract_call_expression_identifiers(child, source_bytes))

        return identifiers

    def _extract_commonjs_export(self, node: Node, source_bytes: bytes) -> ExportInfo | None:
        """Extract export information from CommonJS module.exports or exports.* patterns.

        Handles patterns like:
        - module.exports = function() {}       -> default export
        - module.exports = { foo, bar }        -> named exports
        - module.exports.foo = function() {}   -> named export 'foo'
        - exports.foo = function() {}          -> named export 'foo'
        - module.exports = require('./other')  -> re-export
        """
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")

        if not left_node or not right_node:
            return None

        # Check if this is a module.exports or exports.* pattern
        if left_node.type != "member_expression":
            return None

        left_text = self.get_node_text(left_node, source_bytes)

        exported_names: list[tuple[str, str | None]] = []
        default_export: str | None = None
        is_reexport = False
        reexport_source: str | None = None

        if left_text == "module.exports":
            # module.exports = something
            if right_node.type in {"function_expression", "arrow_function"}:
                # module.exports = function foo() {} or module.exports = () => {}
                name_node = right_node.child_by_field_name("name")
                default_export = self.get_node_text(name_node, source_bytes) if name_node else "default"
            elif right_node.type == "identifier":
                # module.exports = someFunction
                default_export = self.get_node_text(right_node, source_bytes)
            elif right_node.type == "object":
                # module.exports = { foo, bar, baz: qux }
                for child in right_node.children:
                    if child.type == "shorthand_property_identifier":
                        # { foo } - exports function named foo
                        name = self.get_node_text(child, source_bytes)
                        exported_names.append((name, None))
                    elif child.type == "pair":
                        # { baz: qux } - exports qux as baz
                        key_node = child.child_by_field_name("key")
                        value_node = child.child_by_field_name("value")
                        if key_node and value_node:
                            export_name = self.get_node_text(key_node, source_bytes)
                            local_name = self.get_node_text(value_node, source_bytes)
                            # In CommonJS { baz: qux }, baz is the exported name, qux is local
                            exported_names.append((local_name, export_name))
            elif right_node.type == "call_expression":
                # module.exports = require('./other') - re-export
                func_node = right_node.child_by_field_name("function")
                if func_node and self.get_node_text(func_node, source_bytes) == "require":
                    is_reexport = True
                    args_node = right_node.child_by_field_name("arguments")
                    if args_node:
                        for arg in args_node.children:
                            if arg.type == "string":
                                reexport_source = self.get_node_text(arg, source_bytes).strip("'\"")
                                break
                    default_export = "default"
            else:
                # module.exports = something else (class, etc.)
                default_export = "default"

        elif left_text.startswith("module.exports."):
            # module.exports.foo = something
            prop_name = left_text.split(".", 2)[2]  # Get 'foo' from 'module.exports.foo'
            exported_names.append((prop_name, None))

        elif left_text.startswith("exports."):
            # exports.foo = something
            prop_name = left_text.split(".", 1)[1]  # Get 'foo' from 'exports.foo'
            exported_names.append((prop_name, None))

        else:
            # Not a CommonJS export pattern
            return None

        # Skip if no exports found
        if not exported_names and not default_export:
            return None

        return ExportInfo(
            exported_names=exported_names,
            default_export=default_export,
            is_reexport=is_reexport,
            reexport_source=reexport_source,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

    def is_function_exported(
        self, source: str, function_name: str, class_name: str | None = None
    ) -> tuple[bool, str | None]:
        """Check if a function is exported and get its export name.

        For class methods, also checks if the containing class is exported.
        Also handles wrapped exports like: export default curry(traverseEntity)

        Args:
            source: The source code to analyze.
            function_name: The name of the function to check.
            class_name: For class methods, the name of the containing class.

        Returns:
            Tuple of (is_exported, export_name). export_name may differ from
            function_name if exported with an alias. For class methods,
            returns the class export name.

        """
        exports = self.find_exports(source)

        # First, check if the function itself is directly exported
        for export in exports:
            # Check default export
            if export.default_export == function_name:
                return (True, "default")

            # Check named exports
            for name, alias in export.exported_names:
                if name == function_name:
                    return (True, alias if alias else name)

            # Check wrapped default exports: export default curry(traverseEntity)
            # The function is exported via wrapper, so it's accessible as "default"
            if export.wrapped_default_args and function_name in export.wrapped_default_args:
                return (True, "default")

        # For class methods, check if the containing class is exported
        if class_name:
            for export in exports:
                # Check if class is default export
                if export.default_export == class_name:
                    return (True, class_name)

                # Check if class is in named exports
                for name, alias in export.exported_names:
                    if name == class_name:
                        return (True, alias if alias else name)

        return (False, None)

    def find_function_calls(self, source: str, within_function: FunctionNode) -> list[str]:
        """Find all function calls within a specific function's body.

        Args:
            source: The full source code.
            within_function: The function to search within.

        Returns:
            List of function names that are called.

        """
        calls: list[str] = []
        source_bytes = source.encode("utf8")

        # Get the body of the function
        body_node = within_function.node.child_by_field_name("body")
        if body_node is None:
            # For arrow functions, the body might be the last child
            for child in within_function.node.children:
                if child.type in ("statement_block", "expression_statement") or (
                    child.type not in ("identifier", "formal_parameters", "async", "=>")
                ):
                    body_node = child
                    break

        if body_node:
            self._walk_tree_for_calls(body_node, source_bytes, calls)

        return list(set(calls))  # Remove duplicates

    def _walk_tree_for_calls(self, node: Node, source_bytes: bytes, calls: list[str]) -> None:
        """Recursively find function calls in a subtree."""
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    calls.append(self.get_node_text(func_node, source_bytes))
                elif func_node.type == "member_expression":
                    # For method calls like obj.method(), get the method name
                    prop_node = func_node.child_by_field_name("property")
                    if prop_node:
                        calls.append(self.get_node_text(prop_node, source_bytes))

        for child in node.children:
            self._walk_tree_for_calls(child, source_bytes, calls)

    def find_module_level_declarations(self, source: str) -> list[ModuleLevelDeclaration]:
        """Find all module-level variable/constant declarations.

        This finds global variables, constants, classes, enums, type aliases,
        and interfaces defined at the top level of the module (not inside functions).

        Args:
            source: The source code to analyze.

        Returns:
            List of ModuleLevelDeclaration objects.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        declarations: list[ModuleLevelDeclaration] = []

        # Only look at direct children of the program/module node (top-level)
        for child in tree.root_node.children:
            self._extract_module_level_declaration(child, source_bytes, declarations)

        return declarations

    def _extract_module_level_declaration(
        self, node: Node, source_bytes: bytes, declarations: list[ModuleLevelDeclaration]
    ) -> None:
        """Extract module-level declarations from a node."""
        is_exported = False

        # Handle export statements - unwrap to get the actual declaration
        if node.type == "export_statement":
            is_exported = True
            # Find the actual declaration inside the export
            for child in node.children:
                if child.type in ("lexical_declaration", "variable_declaration"):
                    self._extract_declaration(child, source_bytes, declarations, is_exported, node)
                    return
                if child.type == "class_declaration":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        declarations.append(
                            ModuleLevelDeclaration(
                                name=self.get_node_text(name_node, source_bytes),
                                declaration_type="class",
                                source_code=self.get_node_text(node, source_bytes),
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                is_exported=is_exported,
                            )
                        )
                    return
                if child.type in ("type_alias_declaration", "interface_declaration", "enum_declaration"):
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        decl_type = child.type.replace("_declaration", "").replace("_alias", "")
                        declarations.append(
                            ModuleLevelDeclaration(
                                name=self.get_node_text(name_node, source_bytes),
                                declaration_type=decl_type,
                                source_code=self.get_node_text(node, source_bytes),
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                is_exported=is_exported,
                            )
                        )
                    return
            return

        # Handle non-exported declarations
        if node.type in (
            "lexical_declaration",  # const/let
            "variable_declaration",  # var
        ):
            self._extract_declaration(node, source_bytes, declarations, is_exported, node)
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                declarations.append(
                    ModuleLevelDeclaration(
                        name=self.get_node_text(name_node, source_bytes),
                        declaration_type="class",
                        source_code=self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )
        elif node.type in ("type_alias_declaration", "interface_declaration", "enum_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                decl_type = node.type.replace("_declaration", "").replace("_alias", "")
                declarations.append(
                    ModuleLevelDeclaration(
                        name=self.get_node_text(name_node, source_bytes),
                        declaration_type=decl_type,
                        source_code=self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )

    def _extract_declaration(
        self,
        node: Node,
        source_bytes: bytes,
        declarations: list[ModuleLevelDeclaration],
        is_exported: bool,
        source_node: Node,
    ) -> None:
        """Extract variable declarations (const/let/var)."""
        # Determine declaration type (const, let, var)
        decl_type = "var"
        for child in node.children:
            if child.type in ("const", "let", "var"):
                decl_type = child.type
                break

        # Find variable declarators
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    # Handle destructuring patterns
                    if name_node.type == "identifier":
                        declarations.append(
                            ModuleLevelDeclaration(
                                name=self.get_node_text(name_node, source_bytes),
                                declaration_type=decl_type,
                                source_code=self.get_node_text(source_node, source_bytes),
                                start_line=source_node.start_point[0] + 1,
                                end_line=source_node.end_point[0] + 1,
                                is_exported=is_exported,
                            )
                        )
                    elif name_node.type in ("object_pattern", "array_pattern"):
                        # For destructuring, extract all bound identifiers
                        identifiers = self._extract_pattern_identifiers(name_node, source_bytes)
                        for ident in identifiers:
                            declarations.append(
                                ModuleLevelDeclaration(
                                    name=ident,
                                    declaration_type=decl_type,
                                    source_code=self.get_node_text(source_node, source_bytes),
                                    start_line=source_node.start_point[0] + 1,
                                    end_line=source_node.end_point[0] + 1,
                                    is_exported=is_exported,
                                )
                            )

    def _extract_pattern_identifiers(self, pattern_node: Node, source_bytes: bytes) -> list[str]:
        """Extract all identifier names from a destructuring pattern."""
        identifiers: list[str] = []

        def walk(n: Node) -> None:
            if n.type in {"identifier", "shorthand_property_identifier_pattern"}:
                identifiers.append(self.get_node_text(n, source_bytes))
            for child in n.children:
                walk(child)

        walk(pattern_node)
        return identifiers

    def find_referenced_identifiers(self, source: str) -> set[str]:
        """Find all identifiers referenced in the source code.

        This finds all identifier references, excluding:
        - Declaration names (left side of assignments)
        - Property names in object literals
        - Function/class names at definition site

        Args:
            source: The source code to analyze.

        Returns:
            Set of referenced identifier names.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        references: set[str] = set()

        self._walk_tree_for_references(tree.root_node, source_bytes, references)

        return references

    def _walk_tree_for_references(self, node: Node, source_bytes: bytes, references: set[str]) -> None:
        """Walk tree to collect identifier references."""
        if node.type == "identifier":
            # Check if this identifier is a reference (not a declaration)
            parent = node.parent
            if parent is None:
                return

            # Skip function/class/method names at definition
            if parent.type in ("function_declaration", "class_declaration", "method_definition", "function_expression"):
                if parent.child_by_field_name("name") == node:
                    # Don't recurse into parent's children - the parent will be visited separately
                    return

            # Skip variable declarator names (left side of declaration)
            if parent.type == "variable_declarator" and parent.child_by_field_name("name") == node:
                # Don't recurse - the value will be visited when we visit the declarator
                return

            # Skip property names in object literals (keys)
            if parent.type == "pair" and parent.child_by_field_name("key") == node:
                # Don't recurse - the value will be visited when we visit the pair
                return

            # Skip property access property names (obj.property - skip 'property')
            if parent.type == "member_expression" and parent.child_by_field_name("property") == node:
                # Don't recurse - the object will be visited when we visit the member_expression
                return

            # Skip import specifier names
            if parent.type in ("import_specifier", "import_clause", "namespace_import"):
                return

            # Skip export specifier names
            if parent.type == "export_specifier":
                return

            # Skip parameter names in function definitions (but NOT default values)
            if parent.type == "formal_parameters":
                return
            if parent.type == "required_parameter":
                # Only skip if this is the parameter name (pattern field), not the default value
                if parent.child_by_field_name("pattern") == node:
                    return
                # If it's the value field (default value), it's a reference - don't skip

            # This is a reference
            references.add(self.get_node_text(node, source_bytes))
            return

        # Recurse into children
        for child in node.children:
            self._walk_tree_for_references(child, source_bytes, references)

    def has_return_statement(self, function_node: FunctionNode, source: str) -> bool:
        """Check if a function has a return statement.

        Args:
            function_node: The function to check.
            source: The source code.

        Returns:
            True if the function has a return statement.

        """
        source_bytes = source.encode("utf8")

        # Generator functions always implicitly return a Generator/Iterator
        if function_node.is_generator:
            return True

        # For arrow functions with expression body, there's an implicit return
        if function_node.is_arrow:
            body_node = function_node.node.child_by_field_name("body")
            if body_node and body_node.type != "statement_block":
                # Expression body (implicit return)
                return True

        return self._node_has_return(function_node.node)

    def _node_has_return(self, node: Node) -> bool:
        """Recursively check if a node contains a return statement."""
        if node.type == "return_statement":
            return True

        # Don't recurse into nested function definitions
        if node.type in ("function_declaration", "function_expression", "arrow_function", "method_definition"):
            # Only check the current function, not nested ones
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    if self._node_has_return(child):
                        return True
            return False

        return any(self._node_has_return(child) for child in node.children)

    def extract_type_annotations(self, source: str, function_name: str, function_line: int) -> set[str]:
        """Extract type annotation names from a function's parameters and return type.

        Finds the function by name and line number, then extracts all user-defined type names
        from its type annotations (parameters and return type).

        Args:
            source: The source code to analyze.
            function_name: Name of the function to find.
            function_line: Start line of the function (1-indexed).

        Returns:
            Set of type names found in the function's annotations.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        type_names: set[str] = set()

        # Find the function node
        func_node = self._find_function_node(tree.root_node, source_bytes, function_name, function_line)
        if not func_node:
            return type_names

        # Extract type annotations from parameters
        params_node = func_node.child_by_field_name("parameters")
        if params_node:
            self._extract_type_names_from_node(params_node, source_bytes, type_names)

        # Extract return type annotation
        return_type_node = func_node.child_by_field_name("return_type")
        if return_type_node:
            self._extract_type_names_from_node(return_type_node, source_bytes, type_names)

        return type_names

    def extract_class_field_types(self, source: str, class_name: str) -> set[str]:
        """Extract type annotation names from class field declarations.

        Args:
            source: The source code to analyze.
            class_name: Name of the class to analyze.

        Returns:
            Set of type names found in class field annotations.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        type_names: set[str] = set()

        # Find the class node
        class_node = self._find_class_node(tree.root_node, source_bytes, class_name)
        if not class_node:
            return type_names

        # Find class body and extract field type annotations
        body_node = class_node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                # Handle public_field_definition (JS/TS class fields)
                if child.type in ("public_field_definition", "field_definition"):
                    type_annotation = child.child_by_field_name("type")
                    if type_annotation:
                        self._extract_type_names_from_node(type_annotation, source_bytes, type_names)

        return type_names

    def _find_function_node(
        self, node: Node, source_bytes: bytes, function_name: str, function_line: int
    ) -> Node | None:
        """Find a function/method node by name and line number."""
        if node.type in (
            "function_declaration",
            "method_definition",
            "function_expression",
            "generator_function_declaration",
        ):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
                # Line is 1-indexed, tree-sitter is 0-indexed
                if name == function_name and (node.start_point[0] + 1) == function_line:
                    return node

        # Check arrow functions assigned to variables
        if node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    if name_node and value_node and value_node.type == "arrow_function":
                        name = self.get_node_text(name_node, source_bytes)
                        if name == function_name and (node.start_point[0] + 1) == function_line:
                            return value_node

        # Recurse into children
        for child in node.children:
            result = self._find_function_node(child, source_bytes, function_name, function_line)
            if result:
                return result

        return None

    def _find_class_node(self, node: Node, source_bytes: bytes, class_name: str) -> Node | None:
        """Find a class node by name."""
        if node.type in ("class_declaration", "class"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
                if name == class_name:
                    return node

        for child in node.children:
            result = self._find_class_node(child, source_bytes, class_name)
            if result:
                return result

        return None

    def _extract_type_names_from_node(self, node: Node, source_bytes: bytes, type_names: set[str]) -> None:
        """Recursively extract type names from a type annotation node.

        Handles various TypeScript type annotation patterns:
        - Simple types: number, string, Point
        - Generic types: Array<T>, Promise<T>
        - Union types: A | B
        - Intersection types: A & B
        - Array types: T[]
        - Tuple types: [A, B]
        - Object/mapped types: { key: Type }

        Args:
            node: Tree-sitter node to analyze.
            source_bytes: Source code as bytes.
            type_names: Set to add found type names to.

        """
        # Handle type identifiers (the actual type name references)
        if node.type == "type_identifier":
            type_name = self.get_node_text(node, source_bytes)
            # Skip primitive types
            if type_name not in (
                "number",
                "string",
                "boolean",
                "void",
                "null",
                "undefined",
                "any",
                "never",
                "unknown",
                "object",
                "symbol",
                "bigint",
            ):
                type_names.add(type_name)
            return

        # Handle regular identifiers in type position (can happen in some contexts)
        if node.type == "identifier" and node.parent and node.parent.type in ("type_annotation", "generic_type"):
            type_name = self.get_node_text(node, source_bytes)
            if type_name not in (
                "number",
                "string",
                "boolean",
                "void",
                "null",
                "undefined",
                "any",
                "never",
                "unknown",
                "object",
                "symbol",
                "bigint",
            ):
                type_names.add(type_name)
            return

        # Handle nested_type_identifier (e.g., Namespace.Type)
        if node.type == "nested_type_identifier":
            # Get the full qualified name
            type_name = self.get_node_text(node, source_bytes)
            # Add both the full name and the first part (namespace)
            type_names.add(type_name)
            # Also extract the module/namespace part
            module_node = node.child_by_field_name("module")
            if module_node:
                type_names.add(self.get_node_text(module_node, source_bytes))
            return

        # Recurse into all children for compound types
        for child in node.children:
            self._extract_type_names_from_node(child, source_bytes, type_names)

    def find_type_definitions(self, source: str) -> list[TypeDefinition]:
        """Find all type definitions (interface, type, class, enum) in source code.

        Args:
            source: The source code to analyze.

        Returns:
            List of TypeDefinition objects.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        definitions: list[TypeDefinition] = []

        # Walk through top-level nodes
        for child in tree.root_node.children:
            self._extract_type_definition(child, source_bytes, definitions)

        return definitions

    def _extract_type_definition(
        self, node: Node, source_bytes: bytes, definitions: list[TypeDefinition], is_exported: bool = False
    ) -> None:
        """Extract type definitions from a node."""
        # Handle export statements - unwrap to get the actual definition
        if node.type == "export_statement":
            for child in node.children:
                if child.type in (
                    "interface_declaration",
                    "type_alias_declaration",
                    "class_declaration",
                    "enum_declaration",
                ):
                    self._extract_type_definition(child, source_bytes, definitions, is_exported=True)
            return

        # Extract interface definitions
        if node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                # Look for preceding JSDoc comment
                jsdoc = ""
                prev_sibling = node.prev_named_sibling
                if prev_sibling and prev_sibling.type == "comment":
                    comment_text = self.get_node_text(prev_sibling, source_bytes)
                    if comment_text.strip().startswith("/**"):
                        jsdoc = comment_text + "\n"

                definitions.append(
                    TypeDefinition(
                        name=self.get_node_text(name_node, source_bytes),
                        definition_type="interface",
                        source_code=jsdoc + self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )

        # Extract type alias definitions
        elif node.type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                # Look for preceding JSDoc comment
                jsdoc = ""
                prev_sibling = node.prev_named_sibling
                if prev_sibling and prev_sibling.type == "comment":
                    comment_text = self.get_node_text(prev_sibling, source_bytes)
                    if comment_text.strip().startswith("/**"):
                        jsdoc = comment_text + "\n"

                definitions.append(
                    TypeDefinition(
                        name=self.get_node_text(name_node, source_bytes),
                        definition_type="type",
                        source_code=jsdoc + self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )

        # Extract enum definitions
        elif node.type == "enum_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                # Look for preceding JSDoc comment
                jsdoc = ""
                prev_sibling = node.prev_named_sibling
                if prev_sibling and prev_sibling.type == "comment":
                    comment_text = self.get_node_text(prev_sibling, source_bytes)
                    if comment_text.strip().startswith("/**"):
                        jsdoc = comment_text + "\n"

                definitions.append(
                    TypeDefinition(
                        name=self.get_node_text(name_node, source_bytes),
                        definition_type="enum",
                        source_code=jsdoc + self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )

        # Extract class definitions (as types)
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                # Look for preceding JSDoc comment
                jsdoc = ""
                prev_sibling = node.prev_named_sibling
                if prev_sibling and prev_sibling.type == "comment":
                    comment_text = self.get_node_text(prev_sibling, source_bytes)
                    if comment_text.strip().startswith("/**"):
                        jsdoc = comment_text + "\n"

                definitions.append(
                    TypeDefinition(
                        name=self.get_node_text(name_node, source_bytes),
                        definition_type="class",
                        source_code=jsdoc + self.get_node_text(node, source_bytes),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        is_exported=is_exported,
                    )
                )


    @property
    def parser(self) -> Parser:
        # Lazy-initialize the Parser to avoid doing work until parsing is needed.
        if self._parser is None:
            self._parser = Parser()
        return self._parser


def get_analyzer_for_file(file_path: Path) -> TreeSitterAnalyzer:
    """Get the appropriate TreeSitterAnalyzer for a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        TreeSitterAnalyzer configured for the file's language.

    """
    suffix = file_path.suffix.lower()

    if suffix == ".ts":
        return TreeSitterAnalyzer(TreeSitterLanguage.TYPESCRIPT)
    if suffix == ".tsx":
        return TreeSitterAnalyzer(TreeSitterLanguage.TSX)
    # Default to JavaScript for .js, .jsx, .mjs, .cjs
    return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)


# Author: Saurabh Misra <misra.saurabh1@gmail.com>
def extract_calling_function_source(source_code: str, function_name: str, ref_line: int) -> str | None:
    """Extract the source code of a calling function in JavaScript/TypeScript.

    Args:
        source_code: Full source code of the file.
        function_name: Name of the function to extract.
        ref_line: Line number where the reference is (helps identify the right function).

    Returns:
        Source code of the function, or None if not found.

    """
    try:
        from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer, TreeSitterLanguage

        # Try TypeScript first, fall back to JavaScript
        for lang in [TreeSitterLanguage.TYPESCRIPT, TreeSitterLanguage.TSX, TreeSitterLanguage.JAVASCRIPT]:
            try:
                analyzer = TreeSitterAnalyzer(lang)
                functions = analyzer.find_functions(source_code, include_methods=True)

                for func in functions:
                    if func.name == function_name:
                        # Check if the reference line is within this function
                        if func.start_line <= ref_line <= func.end_line:
                            return func.source_text
                break
            except Exception:
                continue

        return None
    except Exception:
        return None
