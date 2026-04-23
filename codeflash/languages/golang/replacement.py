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
