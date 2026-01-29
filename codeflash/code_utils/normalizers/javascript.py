"""JavaScript/TypeScript code normalizer using tree-sitter."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from codeflash.code_utils.normalizers.base import CodeNormalizer

if TYPE_CHECKING:
    from tree_sitter import Node


# TODO:{claude} move to language support directory to keep the directory structure clean
class JavaScriptVariableNormalizer:
    """Normalizes JavaScript/TypeScript code for duplicate detection using tree-sitter.

    Normalizes local variable names while preserving function names, class names,
    parameters, and imported names.
    """

    def __init__(self) -> None:
        self.var_counter = 0
        self.var_mapping: dict[str, str] = {}
        self.preserved_names: set[str] = set()
        # Common JavaScript builtins
        self.builtins = {
            "console",
            "window",
            "document",
            "Math",
            "JSON",
            "Object",
            "Array",
            "String",
            "Number",
            "Boolean",
            "Date",
            "RegExp",
            "Error",
            "Promise",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "Symbol",
            "Proxy",
            "Reflect",
            "undefined",
            "null",
            "NaN",
            "Infinity",
            "globalThis",
            "parseInt",
            "parseFloat",
            "isNaN",
            "isFinite",
            "eval",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            "fetch",
            "require",
            "module",
            "exports",
            "process",
            "__dirname",
            "__filename",
            "Buffer",
        }

    def get_normalized_name(self, name: str) -> str:
        """Get or create normalized name for a variable."""
        if name in self.builtins or name in self.preserved_names:
            return name
        if name not in self.var_mapping:
            self.var_mapping[name] = f"var_{self.var_counter}"
            self.var_counter += 1
        return self.var_mapping[name]

    def collect_preserved_names(self, node: Node, source_code: bytes) -> None:
        """Collect names that should be preserved (function names, class names, imports, params)."""
        # Function declarations and expressions - preserve the function name
        if node.type in ("function_declaration", "function_expression", "method_definition", "arrow_function"):
            name_node = node.child_by_field_name("name")
            if name_node:
                self.preserved_names.add(source_code[name_node.start_byte : name_node.end_byte].decode("utf-8"))
            # Preserve parameters
            params_node = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
            if params_node:
                self._collect_parameter_names(params_node, source_code)

        # Class declarations
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                self.preserved_names.add(source_code[name_node.start_byte : name_node.end_byte].decode("utf-8"))

        # Import declarations
        elif node.type in ("import_statement", "import_declaration"):
            for child in node.children:
                if child.type == "import_clause":
                    self._collect_import_names(child, source_code)
                elif child.type == "identifier":
                    self.preserved_names.add(source_code[child.start_byte : child.end_byte].decode("utf-8"))

        # Recurse
        for child in node.children:
            self.collect_preserved_names(child, source_code)

    def _collect_parameter_names(self, node: Node, source_code: bytes) -> None:
        """Collect parameter names from a parameters node."""
        for child in node.children:
            if child.type == "identifier":
                self.preserved_names.add(source_code[child.start_byte : child.end_byte].decode("utf-8"))
            elif child.type in ("required_parameter", "optional_parameter", "rest_parameter"):
                pattern_node = child.child_by_field_name("pattern")
                if pattern_node and pattern_node.type == "identifier":
                    self.preserved_names.add(
                        source_code[pattern_node.start_byte : pattern_node.end_byte].decode("utf-8")
                    )
            # Recurse for nested patterns
            self._collect_parameter_names(child, source_code)

    def _collect_import_names(self, node: Node, source_code: bytes) -> None:
        """Collect imported names from import clause."""
        for child in node.children:
            if child.type == "identifier":
                self.preserved_names.add(source_code[child.start_byte : child.end_byte].decode("utf-8"))
            elif child.type == "import_specifier":
                # Get the local name (alias or original)
                alias_node = child.child_by_field_name("alias")
                name_node = child.child_by_field_name("name")
                if alias_node:
                    self.preserved_names.add(source_code[alias_node.start_byte : alias_node.end_byte].decode("utf-8"))
                elif name_node:
                    self.preserved_names.add(source_code[name_node.start_byte : name_node.end_byte].decode("utf-8"))
            self._collect_import_names(child, source_code)

    def normalize_tree(self, node: Node, source_code: bytes) -> str:
        """Normalize the AST tree to a string representation for comparison."""
        parts: list[str] = []
        self._normalize_node(node, source_code, parts)
        return " ".join(parts)

    def _normalize_node(self, node: Node, source_code: bytes, parts: list[str]) -> None:
        """Recursively normalize a node."""
        # Skip comments
        if node.type in ("comment", "line_comment", "block_comment"):
            return

        # Handle identifiers - normalize variable names
        if node.type == "identifier":
            name = source_code[node.start_byte : node.end_byte].decode("utf-8")
            normalized = self.get_normalized_name(name)
            parts.append(normalized)
            return

        # Handle type identifiers (TypeScript) - preserve as-is
        if node.type == "type_identifier":
            parts.append(source_code[node.start_byte : node.end_byte].decode("utf-8"))
            return

        # Handle string literals - normalize to placeholder
        if node.type in ("string", "template_string", "string_fragment"):
            parts.append('"STR"')
            return

        # Handle number literals - normalize to placeholder
        if node.type == "number":
            parts.append("NUM")
            return

        # For leaf nodes, output the node type
        if len(node.children) == 0:
            text = source_code[node.start_byte : node.end_byte].decode("utf-8")
            parts.append(text)
            return

        # Output node type for structure
        parts.append(f"({node.type}")

        # Recurse into children
        for child in node.children:
            self._normalize_node(child, source_code, parts)

        parts.append(")")


def _basic_normalize(code: str) -> str:
    """Basic normalization: remove comments and normalize whitespace."""
    # Remove single-line comments
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Normalize whitespace
    return " ".join(code.split())


class JavaScriptNormalizer(CodeNormalizer):
    """JavaScript code normalizer using tree-sitter.

    Normalizes JavaScript code by:
    - Replacing local variable names with canonical forms (var_0, var_1, etc.)
    - Preserving function names, class names, parameters, and imports
    - Removing comments
    - Normalizing string and number literals
    """

    @property
    def language(self) -> str:
        """Return the language this normalizer handles."""
        return "javascript"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return file extensions this normalizer can handle."""
        return (".js", ".jsx", ".mjs", ".cjs")

    def _get_tree_sitter_language(self) -> str:
        """Get the tree-sitter language identifier."""
        return "javascript"

    def normalize(self, code: str) -> str:
        """Normalize JavaScript code to a canonical form.

        Args:
            code: JavaScript source code to normalize

        Returns:
            Normalized representation of the code

        """
        try:
            from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage

            lang_map = {"javascript": TreeSitterLanguage.JAVASCRIPT, "typescript": TreeSitterLanguage.TYPESCRIPT}
            lang = lang_map.get(self._get_tree_sitter_language(), TreeSitterLanguage.JAVASCRIPT)
            analyzer = TreeSitterAnalyzer(lang)
            tree = analyzer.parse(code)

            if tree.root_node.has_error:
                return _basic_normalize(code)

            normalizer = JavaScriptVariableNormalizer()
            source_bytes = code.encode("utf-8")

            # First pass: collect preserved names
            normalizer.collect_preserved_names(tree.root_node, source_bytes)

            # Second pass: normalize and build representation
            return normalizer.normalize_tree(tree.root_node, source_bytes)
        except Exception:
            return _basic_normalize(code)

    def normalize_for_hash(self, code: str) -> str:
        """Normalize JavaScript code optimized for hashing.

        For JavaScript, this is the same as normalize().

        Args:
            code: JavaScript source code to normalize

        Returns:
            Normalized representation suitable for hashing

        """
        return self.normalize(code)


class TypeScriptNormalizer(JavaScriptNormalizer):
    """TypeScript code normalizer using tree-sitter.

    Inherits from JavaScriptNormalizer and overrides language-specific settings.
    """

    @property
    def language(self) -> str:
        """Return the language this normalizer handles."""
        return "typescript"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return file extensions this normalizer can handle."""
        return (".ts", ".tsx", ".mts", ".cts")

    def _get_tree_sitter_language(self) -> str:
        """Get the tree-sitter language identifier."""
        return "typescript"
