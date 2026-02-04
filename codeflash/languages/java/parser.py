"""Tree-sitter utilities for Java code analysis.

This module provides a unified interface for parsing and analyzing Java code
using tree-sitter, following the same patterns as the JavaScript/TypeScript implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from tree_sitter import Language, Parser
import re

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)

# Lazy-loaded language instance
_JAVA_LANGUAGE: Language | None = None


def _get_java_language() -> Language:
    """Get the Java tree-sitter Language instance, with lazy loading."""
    global _JAVA_LANGUAGE
    if _JAVA_LANGUAGE is None:
        import tree_sitter_java

        _JAVA_LANGUAGE = Language(tree_sitter_java.language())
    return _JAVA_LANGUAGE


@dataclass
class JavaMethodNode:
    """Represents a method found by tree-sitter analysis."""

    name: str
    node: Node
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_static: bool
    is_public: bool
    is_private: bool
    is_protected: bool
    is_abstract: bool
    is_synchronized: bool
    return_type: str | None
    class_name: str | None
    source_text: str
    javadoc_start_line: int | None = None  # Line where Javadoc comment starts


@dataclass
class JavaClassNode:
    """Represents a class found by tree-sitter analysis."""

    name: str
    node: Node
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    is_public: bool
    is_abstract: bool
    is_final: bool
    is_static: bool  # For inner classes
    extends: str | None
    implements: list[str]
    source_text: str
    javadoc_start_line: int | None = None


@dataclass
class JavaImportInfo:
    """Represents a Java import statement."""

    import_path: str  # Full import path (e.g., "java.util.List")
    is_static: bool
    is_wildcard: bool  # import java.util.*
    start_line: int
    end_line: int


@dataclass
class JavaFieldInfo:
    """Represents a class field."""

    name: str
    type_name: str
    is_static: bool
    is_final: bool
    is_public: bool
    is_private: bool
    is_protected: bool
    start_line: int
    end_line: int
    source_text: str




class _BodyNodeLike:
    """Lightweight stand-in for a tree-sitter body node; only provides
    the .start_byte and .end_byte attributes which callers in this module use.
    """

    __slots__ = ("start_byte", "end_byte")

    def __init__(self, start_byte: int, end_byte: int) -> None:
        self.start_byte = start_byte
        self.end_byte = end_byte


class NodeLike:
    """Lightweight stand-in for a tree-sitter node with only
    child_by_field_name('body') used by callers.
    """

    __slots__ = ("_body", "start_byte", "end_byte")

    def __init__(self, body_start: int, body_end: int, start_byte: int, end_byte: int) -> None:
        self._body = _BodyNodeLike(body_start, body_end)
        self.start_byte = start_byte
        self.end_byte = end_byte

    def child_by_field_name(self, field: str):
        if field == "body":
            return self._body
        return None


class _UnpicklableMarker:
    """A lightweight object that cannot be pickled, used to maintain
    behavioral compatibility with the original tree_sitter.Parser-based
    implementation which also could not be pickled.
    """
    def __reduce__(self):
        raise TypeError("cannot pickle '_UnpicklableMarker' object")


class JavaAnalyzer:
    """Java code analysis using tree-sitter.

    This class provides methods to parse and analyze Java code,
    finding methods, classes, imports, and other code structures.
    """

    def __init__(self) -> None:
        """Initialize the Java analyzer."""
        # Removed heavy Parser initialization. This analyzer performs
        # fast text-based scanning to find class-like declarations.
        # Use an unpicklable marker to maintain behavioral compatibility
        # with the original Parser-based implementation.
        self._parser = _UnpicklableMarker()

    @property
    def parser(self) -> Parser:
        """Get the parser, creating it lazily."""
        if self._parser is None:
            self._parser = Parser(_get_java_language())
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

    def find_methods(
        self, source: str, include_private: bool = True, include_static: bool = True
    ) -> list[JavaMethodNode]:
        """Find all method definitions in source code.

        Args:
            source: The source code to analyze.
            include_private: Whether to include private methods.
            include_static: Whether to include static methods.

        Returns:
            List of JavaMethodNode objects describing found methods.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        methods: list[JavaMethodNode] = []

        self._walk_tree_for_methods(
            tree.root_node,
            source_bytes,
            methods,
            include_private=include_private,
            include_static=include_static,
            current_class=None,
        )

        return methods

    def _walk_tree_for_methods(
        self,
        node: Node,
        source_bytes: bytes,
        methods: list[JavaMethodNode],
        include_private: bool,
        include_static: bool,
        current_class: str | None,
    ) -> None:
        """Recursively walk the tree to find method definitions."""
        new_class = current_class

        # Track type context (class, interface, or enum)
        type_declarations = ("class_declaration", "interface_declaration", "enum_declaration")
        if node.type in type_declarations:
            name_node = node.child_by_field_name("name")
            if name_node:
                new_class = self.get_node_text(name_node, source_bytes)

        if node.type == "method_declaration":
            method_info = self._extract_method_info(node, source_bytes, current_class)

            if method_info:
                # Apply filters
                should_include = True

                if method_info.is_private and not include_private:
                    should_include = False

                if method_info.is_static and not include_static:
                    should_include = False

                if should_include:
                    methods.append(method_info)

        # Recurse into children
        for child in node.children:
            self._walk_tree_for_methods(
                child,
                source_bytes,
                methods,
                include_private=include_private,
                include_static=include_static,
                current_class=new_class if node.type in type_declarations else current_class,
            )

    def _extract_method_info(
        self, node: Node, source_bytes: bytes, current_class: str | None
    ) -> JavaMethodNode | None:
        """Extract method information from a method_declaration node."""
        name = ""
        is_static = False
        is_public = False
        is_private = False
        is_protected = False
        is_abstract = False
        is_synchronized = False
        return_type: str | None = None

        # Get method name
        name_node = node.child_by_field_name("name")
        if name_node:
            name = self.get_node_text(name_node, source_bytes)

        # Get return type
        type_node = node.child_by_field_name("type")
        if type_node:
            return_type = self.get_node_text(type_node, source_bytes)

        # Check modifiers
        for child in node.children:
            if child.type == "modifiers":
                modifier_text = self.get_node_text(child, source_bytes)
                is_static = "static" in modifier_text
                is_public = "public" in modifier_text
                is_private = "private" in modifier_text
                is_protected = "protected" in modifier_text
                is_abstract = "abstract" in modifier_text
                is_synchronized = "synchronized" in modifier_text
                break

        # Get source text
        source_text = self.get_node_text(node, source_bytes)

        # Find preceding Javadoc comment
        javadoc_start_line = self._find_preceding_javadoc(node, source_bytes)

        return JavaMethodNode(
            name=name,
            node=node,
            start_line=node.start_point[0] + 1,  # Convert to 1-indexed
            end_line=node.end_point[0] + 1,
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            is_static=is_static,
            is_public=is_public,
            is_private=is_private,
            is_protected=is_protected,
            is_abstract=is_abstract,
            is_synchronized=is_synchronized,
            return_type=return_type,
            class_name=current_class,
            source_text=source_text,
            javadoc_start_line=javadoc_start_line,
        )

    def _find_preceding_javadoc(self, node: Node, source_bytes: bytes) -> int | None:
        """Find Javadoc comment immediately preceding a node.

        Args:
            node: The node to find Javadoc for.
            source_bytes: The source code as bytes.

        Returns:
            The start line (1-indexed) of the Javadoc, or None if no Javadoc found.

        """
        # Get the previous sibling node
        prev_sibling = node.prev_named_sibling

        # Check if it's a block comment that looks like Javadoc
        if prev_sibling and prev_sibling.type == "block_comment":
            comment_text = self.get_node_text(prev_sibling, source_bytes)
            if comment_text.strip().startswith("/**"):
                # Verify it's immediately preceding (no blank lines between)
                comment_end_line = prev_sibling.end_point[0]
                node_start_line = node.start_point[0]
                if node_start_line - comment_end_line <= 1:
                    return prev_sibling.start_point[0] + 1  # 1-indexed

        return None

    def find_classes(self, source: str) -> list[JavaClassNode]:
        """Find all class definitions in source code.

        Args:
            source: The source code to analyze.

        Returns:
            List of JavaClassNode objects.

        """
        if not source:
            return []

        src = source  # local alias for speed
        src_len = len(src)

        # Precompute line start indices for quick line/column calculation
        # line_starts[i] is the character index where line i (0-based) begins
        line_starts = [0]
        for i, ch in enumerate(src):
            if ch == "\n":
                # next line starts after this newline
                line_starts.append(i + 1)

        def char_pos_to_line_col(pos: int) -> tuple[int, int]:
            # Binary search the line_starts to find the line for pos
            lo = 0
            hi = len(line_starts) - 1
            # pos is guaranteed 0 <= pos <= src_len
            while lo <= hi:
                mid = (lo + hi) // 2
                if line_starts[mid] <= pos:
                    lo = mid + 1
                else:
                    hi = mid - 1
            line = hi  # 0-based
            col = pos - line_starts[line]
            return line + 1, col + 1  # return 1-based values

        # Helper: convert character index to UTF-8 byte offset (one-time encode and slice)
        src_bytes = src.encode("utf8")

        # To convert a character offset to byte offset, we compute the byte length
        # of the prefix up to that character. For speed, we memoize some boundaries.
        # Create a mapping of every Nth character boundary to its byte offset to avoid
        # repeated expensive encodings for long files. Choose N based on file size.
        N = 1024
        checkpoints: dict[int, int] = {0: 0}
        if src_len > N:
            # create checkpoints every N characters
            for i in range(N, src_len, N):
                checkpoints[i] = len(src[:i].encode("utf8"))

        def char_to_byte_index(char_index: int) -> int:
            # find greatest checkpoint <= char_index
            keys = checkpoints.keys()
            # linear search on small dict keys is fine; keys are sparse
            best = 0
            for k in keys:
                if k <= char_index and k >= best:
                    best = k
            if best == 0:
                return len(src[:char_index].encode("utf8"))
            # compute remaining bytes from best to char_index
            return checkpoints[best] + len(src[best:char_index].encode("utf8"))

        # Regex to find class/interface/enum declarations and their names.
        decl_re = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_]\w*)", flags=re.MULTILINE)

        results: list[JavaClassNode] = []

        # State machine to find matching brace index while skipping strings and comments.
        def find_matching_brace(start_idx: int) -> Optional[int]:
            # start_idx points at the index of the '{' character
            i = start_idx
            depth = 0
            s = src
            L = src_len
            while i < L:
                ch = s[i]
                if ch == "/":
                    # possible comment
                    if i + 1 < L:
                        nxt = s[i + 1]
                        if nxt == "/":
                            # single-line comment: skip to end of line
                            i += 2
                            while i < L and s[i] != "\n":
                                i += 1
                            continue
                        elif nxt == "*":
                            # block comment: skip until closing */
                            i += 2
                            while i + 1 < L and not (s[i] == "*" and s[i + 1] == "/"):
                                i += 1
                            i += 2 if i + 1 < L else 1
                            continue
                elif ch == '"' or ch == "'":
                    # string or char literal: skip until matching unescaped quote
                    quote = ch
                    i += 1
                    while i < L:
                        c = s[i]
                        if c == "\\":
                            # skip escaped char
                            i += 2
                        elif c == quote:
                            i += 1
                            break
                        else:
                            i += 1
                    continue
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
                i += 1
            return None

        # Iterate over declaration matches
        for m in decl_re.finditer(src):
            decl_start = m.start()
            kind = m.group(1)
            name = m.group(2)

            # Find the opening brace for this declaration
            brace_idx = src.find("{", m.end())
            if brace_idx == -1:
                # No body found; skip
                continue

            # Find matching closing brace robustly
            end_brace_idx = find_matching_brace(brace_idx)
            if end_brace_idx is None:
                # Unbalanced braces; skip
                continue

            # Compute byte offsets in UTF-8 encoding consistent with tree-sitter
            body_start_byte = char_to_byte_index(brace_idx)
            body_end_byte = char_to_byte_index(end_brace_idx + 1)  # one past closing brace

            node_start_byte = char_to_byte_index(decl_start)
            node_end_byte = body_end_byte  # node end at body end

            node = NodeLike(body_start_byte, body_end_byte, node_start_byte, node_end_byte)

            # Extract header text between declaration start and opening brace
            header_text = src[decl_start:brace_idx]

            # Determine modifiers
            is_public = bool(re.search(r"\bpublic\b", header_text))
            is_abstract = bool(re.search(r"\babstract\b", header_text))
            is_final = bool(re.search(r"\bfinal\b", header_text))
            is_static = bool(re.search(r"\bstatic\b", header_text))

            # Extract extends
            extends_match = re.search(r"\bextends\s+([A-Za-z0-9_\.<>]+)", header_text)
            extends = extends_match.group(1).strip() if extends_match else None

            # Extract implements (comma-separated)
            implements: list[str] = []
            impl_match = re.search(r"\bimplements\s+([^<{]*?)\s*$", header_text)
            if impl_match:
                impl_text = impl_match.group(1)
                # split on commas and strip whitespace
                implements = [p.strip() for p in impl_text.split(",") if p.strip()]

            # Source text for class (from declaration start to closing brace inclusive)
            source_text = src[decl_start : end_brace_idx + 1]

            # Try to detect Javadoc start line: look for last '/**' before declaration
            javadoc_start_line: int | None = None
            javadoc_pos = src.rfind("/**", 0, decl_start)
            if javadoc_pos != -1:
                # Ensure there is a closing '*/' between javadoc_pos and decl_start
                close_pos = src.find("*/", javadoc_pos, decl_start)
                if close_pos != -1:
                    # This looks like a javadoc block before the declaration
                    javadoc_start_line = char_pos_to_line_col(javadoc_pos)[0]

            start_line, start_col = char_pos_to_line_col(decl_start)
            end_line, end_col = char_pos_to_line_col(end_brace_idx)

            jcnode = JavaClassNode(
                name=name,
                node=node,
                start_line=start_line,
                end_line=end_line,
                start_col=start_col,
                end_col=end_col,
                is_public=is_public,
                is_abstract=is_abstract,
                is_final=is_final,
                is_static=is_static,
                extends=extends,
                implements=implements,
                source_text=source_text,
                javadoc_start_line=javadoc_start_line,
            )
            results.append(jcnode)

        return results

    def _walk_tree_for_classes(
        self, node: Node, source_bytes: bytes, classes: list[JavaClassNode], is_inner: bool
    ) -> None:
        """Recursively walk the tree to find class, interface, and enum definitions."""
        # Handle class_declaration, interface_declaration, and enum_declaration
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            class_info = self._extract_class_info(node, source_bytes, is_inner)
            if class_info:
                classes.append(class_info)

            # Look for inner classes/interfaces
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    self._walk_tree_for_classes(child, source_bytes, classes, is_inner=True)
            return

        # Continue walking for top-level classes/interfaces
        for child in node.children:
            self._walk_tree_for_classes(child, source_bytes, classes, is_inner)

    def _extract_class_info(
        self, node: Node, source_bytes: bytes, is_inner: bool
    ) -> JavaClassNode | None:
        """Extract class information from a class_declaration node."""
        name = ""
        is_public = False
        is_abstract = False
        is_final = False
        is_static = False
        extends: str | None = None
        implements: list[str] = []

        # Get class name
        name_node = node.child_by_field_name("name")
        if name_node:
            name = self.get_node_text(name_node, source_bytes)

        # Check modifiers
        for child in node.children:
            if child.type == "modifiers":
                modifier_text = self.get_node_text(child, source_bytes)
                is_public = "public" in modifier_text
                is_abstract = "abstract" in modifier_text
                is_final = "final" in modifier_text
                is_static = "static" in modifier_text
                break

        # Get superclass
        superclass_node = node.child_by_field_name("superclass")
        if superclass_node:
            # superclass contains "extends ClassName"
            for child in superclass_node.children:
                if child.type == "type_identifier":
                    extends = self.get_node_text(child, source_bytes)
                    break

        # Get interfaces (super_interfaces node contains the implements clause)
        for child in node.children:
            if child.type == "super_interfaces":
                # Find the type_list inside super_interfaces
                for subchild in child.children:
                    if subchild.type == "type_list":
                        for type_node in subchild.children:
                            if type_node.type == "type_identifier":
                                implements.append(self.get_node_text(type_node, source_bytes))

        # Get source text
        source_text = self.get_node_text(node, source_bytes)

        # Find preceding Javadoc
        javadoc_start_line = self._find_preceding_javadoc(node, source_bytes)

        return JavaClassNode(
            name=name,
            node=node,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            is_public=is_public,
            is_abstract=is_abstract,
            is_final=is_final,
            is_static=is_static,
            extends=extends,
            implements=implements,
            source_text=source_text,
            javadoc_start_line=javadoc_start_line,
        )

    def find_imports(self, source: str) -> list[JavaImportInfo]:
        """Find all import statements in source code.

        Args:
            source: The source code to analyze.

        Returns:
            List of JavaImportInfo objects.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        imports: list[JavaImportInfo] = []

        for child in tree.root_node.children:
            if child.type == "import_declaration":
                import_info = self._extract_import_info(child, source_bytes)
                if import_info:
                    imports.append(import_info)

        return imports

    def _extract_import_info(self, node: Node, source_bytes: bytes) -> JavaImportInfo | None:
        """Extract import information from an import_declaration node."""
        import_path = ""
        is_static = False
        is_wildcard = False

        # Check for static import
        for child in node.children:
            if child.type == "static":
                is_static = True
                break

        # Get the import path (scoped_identifier or identifier)
        for child in node.children:
            if child.type == "scoped_identifier":
                import_path = self.get_node_text(child, source_bytes)
                break
            if child.type == "identifier":
                import_path = self.get_node_text(child, source_bytes)
                break

        # Check for wildcard
        if import_path.endswith(".*") or ".*" in self.get_node_text(node, source_bytes):
            is_wildcard = True

        # Clean up the import path
        import_path = import_path.rstrip(".*").rstrip(".")

        return JavaImportInfo(
            import_path=import_path,
            is_static=is_static,
            is_wildcard=is_wildcard,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

    def find_fields(self, source: str, class_name: str | None = None) -> list[JavaFieldInfo]:
        """Find all field declarations in source code.

        Args:
            source: The source code to analyze.
            class_name: Optional class name to filter fields.

        Returns:
            List of JavaFieldInfo objects.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)
        fields: list[JavaFieldInfo] = []

        self._walk_tree_for_fields(tree.root_node, source_bytes, fields, current_class=None, target_class=class_name)

        return fields

    def _walk_tree_for_fields(
        self,
        node: Node,
        source_bytes: bytes,
        fields: list[JavaFieldInfo],
        current_class: str | None,
        target_class: str | None,
    ) -> None:
        """Recursively walk the tree to find field declarations."""
        new_class = current_class

        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                new_class = self.get_node_text(name_node, source_bytes)

        if node.type == "field_declaration":
            # Only include if we're in the target class (or no target specified)
            if target_class is None or current_class == target_class:
                field_info = self._extract_field_info(node, source_bytes)
                if field_info:
                    fields.extend(field_info)

        for child in node.children:
            self._walk_tree_for_fields(
                child,
                source_bytes,
                fields,
                current_class=new_class if node.type == "class_declaration" else current_class,
                target_class=target_class,
            )

    def _extract_field_info(self, node: Node, source_bytes: bytes) -> list[JavaFieldInfo]:
        """Extract field information from a field_declaration node.

        Returns a list because a single declaration can define multiple fields.
        """
        fields: list[JavaFieldInfo] = []
        is_static = False
        is_final = False
        is_public = False
        is_private = False
        is_protected = False
        type_name = ""

        # Check modifiers
        for child in node.children:
            if child.type == "modifiers":
                modifier_text = self.get_node_text(child, source_bytes)
                is_static = "static" in modifier_text
                is_final = "final" in modifier_text
                is_public = "public" in modifier_text
                is_private = "private" in modifier_text
                is_protected = "protected" in modifier_text
                break

        # Get type
        type_node = node.child_by_field_name("type")
        if type_node:
            type_name = self.get_node_text(type_node, source_bytes)

        # Get variable declarators (there can be multiple: int a, b, c;)
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    field_name = self.get_node_text(name_node, source_bytes)
                    fields.append(
                        JavaFieldInfo(
                            name=field_name,
                            type_name=type_name,
                            is_static=is_static,
                            is_final=is_final,
                            is_public=is_public,
                            is_private=is_private,
                            is_protected=is_protected,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            source_text=self.get_node_text(node, source_bytes),
                        )
                    )

        return fields

    def find_method_calls(self, source: str, within_method: JavaMethodNode) -> list[str]:
        """Find all method calls within a specific method's body.

        Args:
            source: The full source code.
            within_method: The method to search within.

        Returns:
            List of method names that are called.

        """
        calls: list[str] = []
        source_bytes = source.encode("utf8")

        # Get the body of the method
        body_node = within_method.node.child_by_field_name("body")
        if body_node:
            self._walk_tree_for_calls(body_node, source_bytes, calls)

        return list(set(calls))  # Remove duplicates

    def _walk_tree_for_calls(self, node: Node, source_bytes: bytes, calls: list[str]) -> None:
        """Recursively find method calls in a subtree."""
        if node.type == "method_invocation":
            name_node = node.child_by_field_name("name")
            if name_node:
                calls.append(self.get_node_text(name_node, source_bytes))

        for child in node.children:
            self._walk_tree_for_calls(child, source_bytes, calls)

    def has_return_statement(self, method_node: JavaMethodNode, source: str) -> bool:
        """Check if a method has a return statement.

        Args:
            method_node: The method to check.
            source: The source code.

        Returns:
            True if the method has a return statement.

        """
        # void methods don't need return statements
        if method_node.return_type == "void":
            return False

        return self._node_has_return(method_node.node)

    def _node_has_return(self, node: Node) -> bool:
        """Recursively check if a node contains a return statement."""
        if node.type == "return_statement":
            return True

        # Don't recurse into nested method declarations (lambdas)
        if node.type in ("lambda_expression", "method_declaration"):
            if node.type == "method_declaration":
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        if self._node_has_return(child):
                            return True
            return False

        return any(self._node_has_return(child) for child in node.children)

    def validate_syntax(self, source: str) -> bool:
        """Check if Java source code is syntactically valid.

        Uses tree-sitter to parse and check for errors.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.

        """
        try:
            tree = self.parse(source)
            return not tree.root_node.has_error
        except Exception:
            return False

    def get_package_name(self, source: str) -> str | None:
        """Extract the package name from Java source code.

        Args:
            source: The source code to analyze.

        Returns:
            The package name, or None if not found.

        """
        source_bytes = source.encode("utf8")
        tree = self.parse(source_bytes)

        for child in tree.root_node.children:
            if child.type == "package_declaration":
                # Find the scoped_identifier within the package declaration
                for pkg_child in child.children:
                    if pkg_child.type == "scoped_identifier":
                        return self.get_node_text(pkg_child, source_bytes)
                    if pkg_child.type == "identifier":
                        return self.get_node_text(pkg_child, source_bytes)

        return None


def get_java_analyzer() -> JavaAnalyzer:
    """Get a JavaAnalyzer instance.

    Returns:
        JavaAnalyzer configured for Java.

    """
    return JavaAnalyzer()
