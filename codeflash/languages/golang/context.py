from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash.languages.base import CodeContext, HelperFunction, Language
from codeflash.languages.golang.parser import GoAnalyzer

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

logger = logging.getLogger(__name__)


def extract_code_context(
    function: FunctionToOptimize,
    project_root: Path,
    module_root: Path | None = None,
    analyzer: GoAnalyzer | None = None,
) -> CodeContext:
    analyzer = analyzer or GoAnalyzer()

    try:
        source = function.file_path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read %s", function.file_path)
        return CodeContext(target_code="", target_file=function.file_path, language=Language.GO)

    receiver_type = _get_receiver_type(function)
    target_code = analyzer.extract_function_source(source, function.function_name, receiver_type=receiver_type)
    if target_code is None:
        target_code = ""

    imports = analyzer.find_imports(source)
    import_lines = [_import_to_line(imp) for imp in imports]

    read_only_parts: list[str] = []
    if receiver_type:
        struct_ctx = _extract_struct_context(source, receiver_type, analyzer)
        if struct_ctx:
            read_only_parts.append(struct_ctx)

    init_ctx = _extract_init_context(source, analyzer)
    if init_ctx:
        read_only_parts.append(init_ctx)

    helpers = find_helper_functions(source, function, analyzer)

    return CodeContext(
        target_code=target_code,
        target_file=function.file_path,
        helper_functions=helpers,
        read_only_context="\n\n".join(read_only_parts),
        imports=import_lines,
        language=Language.GO,
    )


def find_helper_functions(
    source: str, function: FunctionToOptimize, analyzer: GoAnalyzer | None = None
) -> list[HelperFunction]:
    analyzer = analyzer or GoAnalyzer()
    target_name = function.function_name

    functions = analyzer.find_functions(source)
    methods = analyzer.find_methods(source)

    helpers: list[HelperFunction] = []

    for func in functions:
        if func.name == target_name:
            continue
        if func.name in ("init", "main"):
            continue
        extracted = analyzer.extract_function_source(source, func.name)
        if extracted is None:
            continue
        helpers.append(
            HelperFunction(
                name=func.name,
                qualified_name=func.name,
                file_path=function.file_path,
                source_code=extracted,
                start_line=func.starting_line,
                end_line=func.ending_line,
            )
        )

    receiver_type = _get_receiver_type(function)
    for method in methods:
        if method.name == target_name and method.receiver_name == receiver_type:
            continue
        extracted = analyzer.extract_function_source(source, method.name, receiver_type=method.receiver_name)
        if extracted is None:
            continue
        qualified = f"{method.receiver_name}.{method.name}"
        helpers.append(
            HelperFunction(
                name=method.name,
                qualified_name=qualified,
                file_path=function.file_path,
                source_code=extracted,
                start_line=method.starting_line,
                end_line=method.ending_line,
            )
        )

    return helpers


def _get_receiver_type(function: FunctionToOptimize) -> str | None:
    if function.parents:
        return function.parents[0].name
    return None


def _import_to_line(imp: object) -> str:
    path = getattr(imp, "path", "")
    alias = getattr(imp, "alias", None)
    if alias:
        return f'{alias} "{path}"'
    return f'"{path}"'


def _extract_struct_context(source: str, struct_name: str, analyzer: GoAnalyzer) -> str:
    structs = analyzer.find_structs(source)
    for s in structs:
        if s.name == struct_name:
            lines = source.splitlines()
            return "\n".join(lines[s.starting_line - 1 : s.ending_line])
    return ""


def _extract_init_context(source: str, analyzer: GoAnalyzer) -> str:
    init_source = analyzer.extract_function_source(source, "init")
    if init_source is None:
        return ""

    init_ids = analyzer.collect_body_identifiers(source, "init")
    if not init_ids:
        return init_source

    parts: list[str] = []

    for decl in analyzer.find_global_declarations(source):
        if init_ids & set(decl.names):
            parts.append(decl.source_code)

    for struct in analyzer.find_structs(source):
        if struct.name in init_ids:
            lines = source.splitlines()
            parts.append("\n".join(lines[struct.starting_line - 1 : struct.ending_line]))

    parts.append(init_source)
    return "\n\n".join(parts)
