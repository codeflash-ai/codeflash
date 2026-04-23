from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash.languages.golang.parser import GoAnalyzer

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

logger = logging.getLogger(__name__)


def replace_function(
    source: str, function: FunctionToOptimize, new_source: str, analyzer: GoAnalyzer | None = None
) -> str:
    analyzer = analyzer or GoAnalyzer()
    receiver_type = function.parents[0].name if function.parents else None

    tree = analyzer.parse(source)
    target_node = None

    for node in tree.root_node.children:
        if receiver_type is None and node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is not None and analyzer.get_node_text(name_node) == function.function_name:
                target_node = node
                break
        elif receiver_type is not None and node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None or analyzer.get_node_text(name_node) != function.function_name:
                continue
            recv_node = node.child_by_field_name("receiver")
            if recv_node is not None:
                recv_name, _ = analyzer.parse_receiver(recv_node)
                if recv_name == receiver_type:
                    target_node = node
                    break

    if target_node is None:
        logger.warning("Could not find function %s in source for replacement", function.function_name)
        return source

    lines = source.splitlines(keepends=True)
    doc_line = _find_doc_comment_start(target_node)
    start_line = (doc_line if doc_line is not None else target_node.start_point.row + 1) - 1
    end_line = target_node.end_point.row + 1

    new_source_stripped = new_source.rstrip("\n") + "\n"

    result_lines = [*lines[:start_line], new_source_stripped, *lines[end_line:]]
    return "".join(result_lines)


def add_global_declarations(optimized_code: str, original_source: str, analyzer: GoAnalyzer | None = None) -> str:
    analyzer = analyzer or GoAnalyzer()

    merged = _merge_imports(optimized_code, original_source, analyzer)
    return _merge_global_var_const(optimized_code, merged, analyzer)


def _merge_imports(optimized_code: str, original_source: str, analyzer: GoAnalyzer) -> str:
    opt_imports = analyzer.find_imports(optimized_code)
    orig_imports = analyzer.find_imports(original_source)
    orig_paths = {imp.path for imp in orig_imports}

    new_imports = [imp for imp in opt_imports if imp.path not in orig_paths]
    if not new_imports:
        return original_source

    lines = original_source.splitlines(keepends=True)

    import_block_end = _find_import_block_end(original_source, analyzer)

    new_import_lines = []
    for imp in new_imports:
        if imp.alias:
            new_import_lines.append(f'\t{imp.alias} "{imp.path}"\n')
        else:
            new_import_lines.append(f'\t"{imp.path}"\n')

    if orig_imports:
        last_import = max(orig_imports, key=lambda i: i.ending_line)
        insert_at = last_import.ending_line
        for node in analyzer.last_tree.root_node.children if analyzer.last_tree else []:
            if node.type == "import_declaration":
                for child in node.children:
                    if child.type == "import_spec_list":
                        close_paren_line = child.end_point.row
                        insert_at = close_paren_line
                        break
        return "".join([*lines[:insert_at], *new_import_lines, *lines[insert_at:]])

    insert_at = import_block_end
    import_block = "import (\n" + "".join(new_import_lines) + ")\n\n"
    return "".join([*lines[:insert_at], import_block, *lines[insert_at:]])


def _merge_global_var_const(optimized_code: str, original_source: str, analyzer: GoAnalyzer) -> str:
    opt_decls = analyzer.find_global_declarations(optimized_code)
    if not opt_decls:
        return original_source

    orig_decls = analyzer.find_global_declarations(original_source)
    orig_names_to_decl: dict[str, object] = {}
    for decl in orig_decls:
        for name in decl.names:
            orig_names_to_decl[name] = decl

    new_decls: list[str] = []
    replaced_decls: set[int] = set()

    for opt_decl in opt_decls:
        overlapping_orig = None
        for name in opt_decl.names:
            if name in orig_names_to_decl:
                overlapping_orig = orig_names_to_decl[name]
                break

        if overlapping_orig is None:
            new_decls.append(opt_decl.source_code)
        elif overlapping_orig.source_code.strip() != opt_decl.source_code.strip():
            orig_id = id(overlapping_orig)
            if orig_id not in replaced_decls:
                replaced_decls.add(orig_id)
                original_source = _replace_declaration_block(original_source, overlapping_orig, opt_decl.source_code)

    if new_decls:
        original_source = _insert_new_declarations(original_source, new_decls, analyzer)

    return original_source


def _replace_declaration_block(source: str, orig_decl: object, new_source_code: str) -> str:
    lines = source.splitlines(keepends=True)
    start = orig_decl.starting_line - 1
    end = orig_decl.ending_line
    replacement = new_source_code.rstrip("\n") + "\n"
    return "".join([*lines[:start], replacement, *lines[end:]])


def _insert_new_declarations(source: str, new_decls: list[str], analyzer: GoAnalyzer) -> str:
    lines = source.splitlines(keepends=True)

    insert_at = _find_declarations_insert_point(source, analyzer)

    block = "\n".join(new_decls) + "\n\n"
    return "".join([*lines[:insert_at], block, *lines[insert_at:]])


def _find_declarations_insert_point(source: str, analyzer: GoAnalyzer) -> int:
    tree = analyzer.parse(source)
    last_line = 0
    for node in tree.root_node.children:
        if node.type in ("import_declaration", "var_declaration", "const_declaration"):
            candidate = node.end_point.row + 1
            last_line = max(last_line, candidate)
    if last_line > 0:
        return last_line

    for node in tree.root_node.children:
        if node.type == "package_clause":
            return node.end_point.row + 1
    return 0


def remove_test_functions(test_source: str, functions_to_remove: list[str], analyzer: GoAnalyzer | None = None) -> str:
    analyzer = analyzer or GoAnalyzer()
    tree = analyzer.parse(test_source)
    lines = test_source.splitlines(keepends=True)

    regions_to_remove: list[tuple[int, int]] = []

    for node in tree.root_node.children:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is not None and analyzer.get_node_text(name_node) in functions_to_remove:
                doc_start = _find_doc_comment_start(node)
                start = (doc_start if doc_start is not None else node.start_point.row + 1) - 1
                end = node.end_point.row + 1
                regions_to_remove.append((start, end))

    for start, end in reversed(regions_to_remove):
        del lines[start:end]

    return "".join(lines)


def _find_doc_comment_start(node: object) -> int | None:
    prev = getattr(node, "prev_named_sibling", None)
    if prev is None:
        return None
    if getattr(prev, "type", None) != "comment":
        return None
    if prev.end_point.row + 1 != node.start_point.row:
        return None
    comment_start = prev.start_point.row + 1
    current = prev
    while True:
        earlier = getattr(current, "prev_named_sibling", None)
        if earlier is None or getattr(earlier, "type", None) != "comment":
            break
        if earlier.end_point.row + 1 != current.start_point.row:
            break
        comment_start = earlier.start_point.row + 1
        current = earlier
    return comment_start


def _find_import_block_end(source: str, analyzer: GoAnalyzer) -> int:
    tree = analyzer.parse(source)
    for node in tree.root_node.children:
        if node.type == "package_clause":
            return node.end_point.row + 1
    return 0
