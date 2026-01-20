"""Tree-sitter utilities for cross-language code analysis.

This module provides a unified interface for parsing and analyzing code
across multiple languages using tree-sitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from tree_sitter import Language, Node, Parser

if TYPE_CHECKING:
    from tree_sitter import Tree

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


class TreeSitterAnalyzer:
    """Cross-language code analysis using tree-sitter.

    This class provides methods to parse and analyze JavaScript/TypeScript code,
    finding functions, imports, and other code structures.
    """

    def __init__(self, language: TreeSitterLanguage | str):
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
            source = source.encode("utf8")
        return self.parser.parse(source)

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

        if node.type == "class_declaration" or node.type == "class":
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

        # Check for async modifier
        for child in node.children:
            if child.type == "async":
                is_async = True
                break

        # Check for generator
        if "generator" in node.type:
            is_generator = True

        # Get function name based on node type
        if node.type in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self.get_node_text(name_node, source_bytes)
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
        )

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

    def _walk_tree_for_imports(self, node: Node, source_bytes: bytes, imports: list[ImportInfo]) -> None:
        """Recursively walk the tree to find import statements."""
        if node.type == "import_statement":
            import_info = self._extract_import_info(node, source_bytes)
            if import_info:
                imports.append(import_info)

        # Also handle require() calls for CommonJS
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and self.get_node_text(func_node, source_bytes) == "require":
                import_info = self._extract_require_info(node, source_bytes)
                if import_info:
                    imports.append(import_info)

        for child in node.children:
            self._walk_tree_for_imports(child, source_bytes, imports)

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
                    if sibling.type == "function_declaration" or sibling.type == "class_declaration":
                        name_node = sibling.child_by_field_name("name")
                        if name_node:
                            default_export = self.get_node_text(name_node, source_bytes)
                        else:
                            default_export = "default"
                    elif sibling.type == "identifier":
                        default_export = self.get_node_text(sibling, source_bytes)
                    elif sibling.type in ("arrow_function", "function_expression", "object", "array"):
                        default_export = "default"
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
        )

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
            if right_node.type == "function_expression" or right_node.type == "arrow_function":
                # module.exports = function foo() {} or module.exports = () => {}
                name_node = right_node.child_by_field_name("name")
                if name_node:
                    default_export = self.get_node_text(name_node, source_bytes)
                else:
                    default_export = "default"
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

    def is_function_exported(self, source: str, function_name: str) -> tuple[bool, str | None]:
        """Check if a function is exported and get its export name.

        Args:
            source: The source code to analyze.
            function_name: The name of the function to check.

        Returns:
            Tuple of (is_exported, export_name). export_name may differ from
            function_name if exported with an alias.

        """
        exports = self.find_exports(source)

        for export in exports:
            # Check default export
            if export.default_export == function_name:
                return (True, "default")

            # Check named exports
            for name, alias in export.exported_names:
                if name == function_name:
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

    def has_return_statement(self, function_node: FunctionNode, source: str) -> bool:
        """Check if a function has a return statement.

        Args:
            function_node: The function to check.
            source: The source code.

        Returns:
            True if the function has a return statement.

        """
        source_bytes = source.encode("utf8")

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

        for child in node.children:
            if self._node_has_return(child):
                return True

        return False


def get_analyzer_for_file(file_path: Path) -> TreeSitterAnalyzer:
    """Get the appropriate TreeSitterAnalyzer for a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        TreeSitterAnalyzer configured for the file's language.

    """
    suffix = file_path.suffix.lower()

    if suffix in (".ts",):
        return TreeSitterAnalyzer(TreeSitterLanguage.TYPESCRIPT)
    if suffix in (".tsx",):
        return TreeSitterAnalyzer(TreeSitterLanguage.TSX)
    # Default to JavaScript for .js, .jsx, .mjs, .cjs
    return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)
