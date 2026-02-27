"""JavaScript/TypeScript code replacement helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
import weakref

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.languages.base import Language
    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

_imports_cache: "weakref.WeakKeyDictionary[TreeSitterAnalyzer, dict[str, list]]" = weakref.WeakKeyDictionary()

_decls_cache: "weakref.WeakKeyDictionary[TreeSitterAnalyzer, dict[str, list]]" = weakref.WeakKeyDictionary()


# Author: ali <mohammed18200118@gmail.com>
def _add_global_declarations_for_language(
    optimized_code: str, original_source: str, module_abspath: Path, language: Language
) -> str:
    """Add new global declarations from optimized code to original source.

    Finds module-level declarations (const, let, var, class, type, interface, enum)
    in the optimized code that don't exist in the original source and adds them.

    New declarations are inserted after any existing declarations they depend on.
    For example, if optimized code has `const _has = FOO.bar.bind(FOO)`, and `FOO`
    is already declared in the original source, `_has` will be inserted after `FOO`.

    Args:
        optimized_code: The optimized code that may contain new declarations.
        original_source: The original source code.
        module_abspath: Path to the module file (for parser selection).
        language: The language of the code.

    Returns:
        Original source with new declarations added in dependency order.

    """
    from codeflash.languages.base import Language

    if language not in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        return original_source

    try:
        from codeflash.languages.javascript.treesitter import get_analyzer_for_file

        analyzer = get_analyzer_for_file(module_abspath)

        # Merge imports from optimized code into original source
        result = _merge_imports(original_source, optimized_code, analyzer)

        # Use cached declaration retrieval to reduce parse overhead
        original_declarations = _cached_find_module_level_declarations(analyzer, result)
        optimized_declarations = _cached_find_module_level_declarations(analyzer, optimized_code)


        if not optimized_declarations:
            return result

        existing_names = _get_existing_names(original_declarations, analyzer, result)
        new_declarations = _filter_new_declarations(optimized_declarations, existing_names)

        if not new_declarations:
            return result

        # Build a map of existing declaration names to their end lines (1-indexed)
        existing_decl_end_lines = {decl.name: decl.end_line for decl in original_declarations}

        for decl in new_declarations:
            result = _insert_declaration_after_dependencies(
                result, decl, existing_decl_end_lines, analyzer, module_abspath
            )
            # Update the map with the newly inserted declaration for subsequent insertions
            # Re-parse to get accurate line numbers after insertion
            updated_declarations = _cached_find_module_level_declarations(analyzer, result)
            existing_decl_end_lines = {d.name: d.end_line for d in updated_declarations}

        return result

    except Exception as e:
        logger.debug(f"Error adding global declarations: {e}")
        return original_source


# Author: ali <mohammed18200118@gmail.com>
def _get_existing_names(original_declarations: list, analyzer: TreeSitterAnalyzer, original_source: str) -> set[str]:
    """Get all names that already exist in the original source (declarations + imports)."""
    existing_names = {decl.name for decl in original_declarations}

    # Use cached find_imports to avoid re-parsing the same source
    original_imports = _cached_find_imports(analyzer, original_source)
    for imp in original_imports:
        if imp.default_import:
            existing_names.add(imp.default_import)
        for name, alias in imp.named_imports:
            existing_names.add(alias if alias else name)
        if imp.namespace_import:
            existing_names.add(imp.namespace_import)

    return existing_names


# Author: ali <mohammed18200118@gmail.com>
def _filter_new_declarations(optimized_declarations: list, existing_names: set[str]) -> list:
    """Filter declarations to only those that don't exist in the original source."""
    new_declarations = []
    seen_sources: set[str] = set()

    # Sort by line number to maintain order from optimized code
    sorted_declarations = sorted(optimized_declarations, key=lambda d: d.start_line)

    for decl in sorted_declarations:
        if decl.name not in existing_names and decl.source_code not in seen_sources:
            new_declarations.append(decl)
            seen_sources.add(decl.source_code)

    return new_declarations


# Author: ali <mohammed18200118@gmail.com>
def _insert_declaration_after_dependencies(
    source: str,
    declaration,
    existing_decl_end_lines: dict[str, int],
    analyzer: TreeSitterAnalyzer,
    module_abspath: Path,
) -> str:
    """Insert a declaration after the last existing declaration it depends on.

    Args:
        source: Current source code.
        declaration: The declaration to insert.
        existing_decl_end_lines: Map of existing declaration names to their end lines.
        analyzer: TreeSitter analyzer.
        module_abspath: Path to the module file.

    Returns:
        Source code with the declaration inserted at the correct position.

    """
    # Find identifiers referenced in this declaration
    referenced_names = analyzer.find_referenced_identifiers(declaration.source_code)

    # Find the latest end line among all referenced declarations
    insertion_line = _find_insertion_line_for_declaration(source, referenced_names, existing_decl_end_lines, analyzer)

    lines = source.splitlines(keepends=True)

    # Ensure proper spacing
    decl_code = declaration.source_code
    if not decl_code.endswith("\n"):
        decl_code += "\n"

    # Add blank line before if inserting after content
    if insertion_line > 0 and lines[insertion_line - 1].strip():
        decl_code = "\n" + decl_code

    before = lines[:insertion_line]
    after = lines[insertion_line:]

    return "".join([*before, decl_code, *after])


# Author: ali <mohammed18200118@gmail.com>
def _find_insertion_line_for_declaration(
    source: str, referenced_names: set[str], existing_decl_end_lines: dict[str, int], analyzer: TreeSitterAnalyzer
) -> int:
    """Find the line where a declaration should be inserted based on its dependencies.

    Args:
        source: Source code.
        referenced_names: Names referenced by the declaration.
        existing_decl_end_lines: Map of declaration names to their end lines (1-indexed).
        analyzer: TreeSitter analyzer.

    Returns:
        Line index (0-based) where the declaration should be inserted.

    """
    # Find the maximum end line among referenced declarations
    max_dependency_line = 0
    for name in referenced_names:
        if name in existing_decl_end_lines:
            max_dependency_line = max(max_dependency_line, existing_decl_end_lines[name])

    if max_dependency_line > 0:
        # Insert after the last dependency (end_line is 1-indexed, we need 0-indexed)
        return max_dependency_line

    # No dependencies found - insert after imports
    lines = source.splitlines(keepends=True)
    return _find_line_after_imports(lines, analyzer, source)


# Author: ali <mohammed18200118@gmail.com>
def _find_line_after_imports(lines: list[str], analyzer: TreeSitterAnalyzer, source: str) -> int:
    """Find the line index after all imports.

    Args:
        lines: Source lines.
        analyzer: TreeSitter analyzer.
        source: Full source code.

    Returns:
        Line index (0-based) for insertion after imports.

    """
    try:
        imports = analyzer.find_imports(source)
        if imports:
            return max(imp.end_line for imp in imports)
    except Exception as exc:
        logger.debug(f"Exception in _find_line_after_imports: {exc}")

    # Default: insert at beginning (after shebang/directive comments)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("//") and not stripped.startswith("#!"):
            return i

    return 0


def _merge_imports(source: str, new_source: str, analyzer: TreeSitterAnalyzer) -> str:
    """Merge imports from new_source into source.

    For imports from the same module, merges named imports so that any new
    named imports from new_source are added to the existing import in source.
    Also merges default imports and namespace imports when the source import
    is missing them.
    """
    try:
        # Fast path: if there are no import tokens in new_source, avoid parsing
        if "import" not in new_source:
            return source

        source_imports = _cached_find_imports(analyzer, source)
        new_imports = _cached_find_imports(analyzer, new_source)
    except Exception:
        return source

    if not new_imports:
        return source

    source_import_map: dict[str, list] = {}
    for imp in source_imports:
        source_import_map.setdefault(imp.module_path, []).append(imp)

    lines = source.splitlines(keepends=True)

    replacements: list[tuple[int, int, str]] = []
    for new_imp in new_imports:
        matching = source_import_map.get(new_imp.module_path)
        if not matching:
            continue

        for src_imp in matching:
            existing_names = {name for name, _ in src_imp.named_imports}
            new_names = [(name, alias) for name, alias in new_imp.named_imports if name not in existing_names]

            new_default = new_imp.default_import if not src_imp.default_import and new_imp.default_import else None
            new_namespace = (
                new_imp.namespace_import if not src_imp.namespace_import and new_imp.namespace_import else None
            )

            if not new_names and not new_default and not new_namespace:
                continue

            merged_named = list(src_imp.named_imports) + new_names
            default_part = new_default or src_imp.default_import
            namespace_part = new_namespace or src_imp.namespace_import

            parts = []
            if default_part:
                parts.append(default_part)
            if namespace_part:
                parts.append(f"* as {namespace_part}")
            if merged_named:
                named_str = ", ".join(f"{name} as {alias}" if alias else name for name, alias in merged_named)
                parts.append("{ " + named_str + " }")

            orig_line_idx = src_imp.start_line - 1
            orig_line = lines[orig_line_idx] if orig_line_idx < len(lines) else ""
            quote = "'" if "'" in orig_line else '"'
            semicolon = ";" if orig_line.rstrip().endswith(";") else ""
            type_prefix = "type " if src_imp.is_type_only else ""

            merged_line = (
                f"import {type_prefix}{', '.join(parts)} from {quote}{src_imp.module_path}{quote}{semicolon}\n"
            )
            replacements.append((src_imp.start_line, src_imp.end_line, merged_line))

    if not replacements:
        return source

    replacements.sort(key=lambda r: r[0], reverse=True)
    for start_line, end_line, new_line in replacements:
        lines[start_line - 1 : end_line] = [new_line]

    return "".join(lines)



def _cached_find_imports(analyzer: TreeSitterAnalyzer, source: str):
    """Cached wrapper for analyzer.find_imports to avoid repeated parses."""
    a_cache = _imports_cache.get(analyzer)
    if a_cache is None:
        a_cache = {}
        _imports_cache[analyzer] = a_cache
    res = a_cache.get(source)
    if res is None:
        res = analyzer.find_imports(source)
        a_cache[source] = res
    return res



def _cached_find_module_level_declarations(analyzer: TreeSitterAnalyzer, source: str):
    """Cached wrapper for analyzer.find_module_level_declarations to avoid repeated parses."""
    a_cache = _decls_cache.get(analyzer)
    if a_cache is None:
        a_cache = {}
        _decls_cache[analyzer] = a_cache
    res = a_cache.get(source)
    if res is None:
        res = analyzer.find_module_level_declarations(source)
        a_cache[source] = res
    return res


def _cached_find_imports(analyzer: TreeSitterAnalyzer, source: str):
    """Cached wrapper for analyzer.find_imports to avoid repeated parses."""
    a_cache = _imports_cache.get(analyzer)
    if a_cache is None:
        a_cache = {}
        _imports_cache[analyzer] = a_cache
    res = a_cache.get(source)
    if res is None:
        res = analyzer.find_imports(source)
        a_cache[source] = res
    return res


def _cached_find_module_level_declarations(analyzer: TreeSitterAnalyzer, source: str):
    """Cached wrapper for analyzer.find_module_level_declarations to avoid repeated parses."""
    a_cache = _decls_cache.get(analyzer)
    if a_cache is None:
        a_cache = {}
        _decls_cache[analyzer] = a_cache
    res = a_cache.get(source)
    if res is None:
        res = analyzer.find_module_level_declarations(source)
        a_cache[source] = res
    return res
