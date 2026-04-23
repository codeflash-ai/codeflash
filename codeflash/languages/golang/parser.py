from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)

_GO_LANGUAGE: Language | None = None
_GO_PARSER: Parser | None = None


def _get_go_language() -> Language:
    global _GO_LANGUAGE
    if _GO_LANGUAGE is None:
        import tree_sitter_go

        _GO_LANGUAGE = Language(tree_sitter_go.language())
    return _GO_LANGUAGE


def _get_go_parser() -> Parser:
    global _GO_PARSER
    if _GO_PARSER is None:
        _GO_PARSER = Parser(_get_go_language())
    return _GO_PARSER


@dataclass(frozen=True)
class GoFunctionNode:
    name: str
    starting_line: int
    ending_line: int
    starting_col: int
    ending_col: int
    is_exported: bool
    has_return_type: bool
    doc_start_line: int | None = None


@dataclass(frozen=True)
class GoMethodNode:
    name: str
    receiver_name: str
    receiver_is_pointer: bool
    starting_line: int
    ending_line: int
    starting_col: int
    ending_col: int
    is_exported: bool
    has_return_type: bool
    doc_start_line: int | None = None


@dataclass(frozen=True)
class GoStructNode:
    name: str
    starting_line: int
    ending_line: int
    fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoInterfaceNode:
    name: str
    starting_line: int
    ending_line: int
    methods: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoImportInfo:
    path: str
    alias: str | None
    starting_line: int
    ending_line: int


class GoAnalyzer:
    def __init__(self) -> None:
        self._parser = _get_go_parser()
        self._source_bytes: bytes | None = None
        self._tree: Tree | None = None

    @property
    def last_tree(self) -> Tree | None:
        return self._tree

    def parse(self, source: str) -> Tree:
        self._source_bytes = source.encode("utf-8")
        self._tree = self._parser.parse(self._source_bytes)
        return self._tree

    def get_node_text(self, node: Node) -> str:
        if self._source_bytes is None:
            return ""
        return self._source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def validate_syntax(self, source: str) -> bool:
        tree = self.parse(source)
        return not tree.root_node.has_error

    def find_functions(self, source: str) -> list[GoFunctionNode]:
        tree = self.parse(source)
        results: list[GoFunctionNode] = []
        for node in tree.root_node.children:
            if node.type == "function_declaration":
                func = self._parse_function_node(node)
                if func is not None:
                    results.append(func)
        return results

    def find_methods(self, source: str) -> list[GoMethodNode]:
        tree = self.parse(source)
        results: list[GoMethodNode] = []
        for node in tree.root_node.children:
            if node.type == "method_declaration":
                method = self._parse_method_node(node)
                if method is not None:
                    results.append(method)
        return results

    def find_structs(self, source: str) -> list[GoStructNode]:
        tree = self.parse(source)
        results: list[GoStructNode] = []
        for node in tree.root_node.children:
            if node.type == "type_declaration":
                for spec in _children_of_type(node, "type_spec"):
                    type_node = spec.child_by_field_name("type")
                    if type_node is not None and type_node.type == "struct_type":
                        name_node = spec.child_by_field_name("name")
                        if name_node is not None:
                            fields = self._extract_struct_fields(type_node)
                            results.append(
                                GoStructNode(
                                    name=self.get_node_text(name_node),
                                    starting_line=node.start_point.row + 1,
                                    ending_line=node.end_point.row + 1,
                                    fields=fields,
                                )
                            )
        return results

    def find_interfaces(self, source: str) -> list[GoInterfaceNode]:
        tree = self.parse(source)
        results: list[GoInterfaceNode] = []
        for node in tree.root_node.children:
            if node.type == "type_declaration":
                for spec in _children_of_type(node, "type_spec"):
                    type_node = spec.child_by_field_name("type")
                    if type_node is not None and type_node.type == "interface_type":
                        name_node = spec.child_by_field_name("name")
                        if name_node is not None:
                            methods = self._extract_interface_methods(type_node)
                            results.append(
                                GoInterfaceNode(
                                    name=self.get_node_text(name_node),
                                    starting_line=node.start_point.row + 1,
                                    ending_line=node.end_point.row + 1,
                                    methods=methods,
                                )
                            )
        return results

    def find_imports(self, source: str) -> list[GoImportInfo]:
        tree = self.parse(source)
        results: list[GoImportInfo] = []
        for node in tree.root_node.children:
            if node.type == "import_declaration":
                for spec in _iter_import_specs(node):
                    path_node = spec.child_by_field_name("path")
                    if path_node is None:
                        continue
                    import_path = self.get_node_text(path_node).strip('"')
                    alias_node = spec.child_by_field_name("name")
                    alias = self.get_node_text(alias_node) if alias_node is not None else None
                    results.append(
                        GoImportInfo(
                            path=import_path,
                            alias=alias,
                            starting_line=spec.start_point.row + 1,
                            ending_line=spec.end_point.row + 1,
                        )
                    )
        return results

    def find_package_name(self, source: str) -> str | None:
        tree = self.parse(source)
        for node in tree.root_node.children:
            if node.type == "package_clause":
                for child in node.children:
                    if child.type == "package_identifier":
                        return self.get_node_text(child)
        return None

    def _parse_function_node(self, node: Node) -> GoFunctionNode | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self.get_node_text(name_node)
        result_node = node.child_by_field_name("result")
        doc_line = _find_preceding_comment_line(node)
        return GoFunctionNode(
            name=name,
            starting_line=node.start_point.row + 1,
            ending_line=node.end_point.row + 1,
            starting_col=node.start_point.column,
            ending_col=node.end_point.column,
            is_exported=name[0].isupper(),
            has_return_type=result_node is not None,
            doc_start_line=doc_line,
        )

    def _parse_method_node(self, node: Node) -> GoMethodNode | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self.get_node_text(name_node)

        receiver_node = node.child_by_field_name("receiver")
        if receiver_node is None:
            return None
        receiver_name, receiver_is_pointer = self.parse_receiver(receiver_node)
        if receiver_name is None:
            return None

        result_node = node.child_by_field_name("result")
        doc_line = _find_preceding_comment_line(node)
        return GoMethodNode(
            name=name,
            receiver_name=receiver_name,
            receiver_is_pointer=receiver_is_pointer,
            starting_line=node.start_point.row + 1,
            ending_line=node.end_point.row + 1,
            starting_col=node.start_point.column,
            ending_col=node.end_point.column,
            is_exported=name[0].isupper(),
            has_return_type=result_node is not None,
            doc_start_line=doc_line,
        )

    def parse_receiver(self, receiver_node: Node) -> tuple[str | None, bool]:
        for param in _children_of_type(receiver_node, "parameter_declaration"):
            type_node = param.child_by_field_name("type")
            if type_node is None:
                continue
            if type_node.type == "pointer_type":
                inner = type_node.child(1)
                if inner is not None:
                    return self.get_node_text(inner), True
            elif type_node.type == "type_identifier":
                return self.get_node_text(type_node), False
        return None, False

    def _extract_struct_fields(self, struct_node: Node) -> list[str]:
        fields: list[str] = []
        for child in struct_node.children:
            if child.type == "field_declaration_list":
                for fc in child.children:
                    if fc.type == "field_declaration":
                        fields.append(self.get_node_text(fc).strip())
                break
        return fields

    def _extract_interface_methods(self, iface_node: Node) -> list[str]:
        methods: list[str] = []
        for child in iface_node.children:
            if child.type == "method_elem":
                methods.append(self.get_node_text(child).strip())
        return methods

    def extract_function_source(self, source: str, func_name: str, receiver_type: str | None = None) -> str | None:
        tree = self.parse(source)
        for node in tree.root_node.children:
            if receiver_type is None and node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node is not None and self.get_node_text(name_node) == func_name:
                    return self._get_source_with_doc(node)

            if receiver_type is not None and node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node is None or self.get_node_text(name_node) != func_name:
                    continue
                recv_node = node.child_by_field_name("receiver")
                if recv_node is not None:
                    recv_name, _ = self.parse_receiver(recv_node)
                    if recv_name == receiver_type:
                        return self._get_source_with_doc(node)
        return None

    def _get_source_with_doc(self, node: Node) -> str:
        doc_line = _find_preceding_comment_line(node)
        if doc_line is not None and self._source_bytes is not None:
            lines = self._source_bytes.decode("utf-8").splitlines(keepends=True)
            start = doc_line - 1
            end = node.end_point.row + 1
            return "".join(lines[start:end])
        return self.get_node_text(node)


def _children_of_type(node: Node, type_name: str) -> list[Node]:
    return [child for child in node.children if child.type == type_name]


def _iter_import_specs(import_node: Node) -> list[Node]:
    results: list[Node] = []
    for child in import_node.children:
        if child.type == "import_spec":
            results.append(child)
        elif child.type == "import_spec_list":
            results.extend(c for c in child.children if c.type == "import_spec")
    return results


def _find_preceding_comment_line(node: Node) -> int | None:
    prev = node.prev_named_sibling
    if prev is None:
        return None
    if prev.type != "comment":
        return None
    if prev.end_point.row + 1 != node.start_point.row:
        return None
    comment_start = prev.start_point.row + 1
    current = prev
    while True:
        earlier = current.prev_named_sibling
        if earlier is None or earlier.type != "comment":
            break
        if earlier.end_point.row + 1 != current.start_point.row:
            break
        comment_start = earlier.start_point.row + 1
        current = earlier
    return comment_start
