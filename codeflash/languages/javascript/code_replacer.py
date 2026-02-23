"""JavaScript/TypeScript code replacement helpers."""

from __future__ import annotations
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.languages.base import Language
    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer


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
        from codeflash.languages.javascript.treesitter import \
            get_analyzer_for_file

        analyzer = get_analyzer_for_file(module_abspath)

        # Merge imports from optimized code into original source
        result = _merge_imports(original_source, optimized_code, analyzer)

        original_declarations = analyzer.find_module_level_declarations(result)
        optimized_declarations = analyzer.find_module_level_declarations(optimized_code)

        if not optimized_declarations:
            return result

        existing_names = _get_existing_names(original_declarations, analyzer, result)
        new_declarations = _filter_new_declarations(optimized_declarations, existing_names)

        if not new_declarations:
            return result

        # Build a map of existing declaration names to their end lines (1-indexed)
        existing_decl_end_lines = {decl.name: decl.end_line for decl in original_declarations}

        # Insert each new declaration after its dependencies

        # Work with line list to avoid repeated full-string splits/joins and reparses.
        result_lines = result.splitlines(keepends=True)

        # Insert each new declaration after its dependencies
        for decl in new_declarations:
            # Find identifiers referenced in this declaration
            referenced_names = analyzer.find_referenced_identifiers(decl.source_code)

            # Find insertion line using the current result (join on demand)
            insertion_line = _find_insertion_line_for_declaration("".join(result_lines), referenced_names, existing_decl_end_lines, analyzer)

            # Ensure proper spacing and newline termination
            decl_code = decl.source_code
            if not decl_code.endswith("\n"):
                decl_code += "\n"

            # Add blank line before if inserting after content
            if insertion_line > 0 and result_lines[insertion_line - 1].strip():
                decl_code = "\n" + decl_code

            # Split the declaration into actual lines matching result_lines structure
            added_lines = decl_code.splitlines(keepends=True)
            num_added = len(added_lines)

            # Insert into result_lines
            if insertion_line < 0:
                insertion_line = 0
            if insertion_line > len(result_lines):
                insertion_line = len(result_lines)
            result_lines[insertion_line:insertion_line] = added_lines

            # Update existing declaration end lines: any declaration with end_line >= insertion_line+1 shifts down
            # and add the newly inserted declaration with its end line.
            threshold = insertion_line + 1  # end_line is 1-indexed
            for name, end_line in list(existing_decl_end_lines.items()):
                if end_line >= threshold:
                    existing_decl_end_lines[name] = end_line + num_added
            existing_decl_end_lines[decl.name] = insertion_line + num_added

        # Return the updated source
        return "".join(result_lines)


    except Exception as e:
        logger.debug(f"Error adding global declarations: {e}")
        return original_source


# Author: ali <mohammed18200118@gmail.com>
def _get_existing_names(original_declarations: list, analyzer: TreeSitterAnalyzer, original_source: str) -> set[str]:
    """Get all names that already exist in the original source (declarations + imports)."""
    existing_names = {decl.name for decl in original_declarations}

    original_imports = analyzer.find_imports(original_source)
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


def _merge_imports(original_source: str, optimized_code: str, analyzer: TreeSitterAnalyzer) -> str:
    """Merge imports from optimized code into original source.

    For each import in the optimized code that shares a module path with an existing
    import in the original source, adds any new named imports to the original import line.
    """
    try:
        original_imports = analyzer.find_imports(original_source)
        optimized_imports = analyzer.find_imports(optimized_code)
    except Exception:
        return original_source

    if not optimized_imports:
        return original_source

    # Build a map of module_path -> ImportInfo for original imports
    original_import_map: dict[str, list] = {}
    for imp in original_imports:
        original_import_map.setdefault(imp.module_path, []).append(imp)

    # Work on a line list to avoid repeated splitting/joining
    result_lines = original_source.splitlines(keepends=True)

    for opt_imp in optimized_imports:
        if opt_imp.module_path not in original_import_map:
            continue

        # Get new named imports that don't exist in the original
        for orig_imp in original_import_map[opt_imp.module_path]:
            existing_names = {name for name, _ in orig_imp.named_imports}
            new_names = [(name, alias) for name, alias in opt_imp.named_imports if name not in existing_names]

            if not new_names:
                continue

            # Find the original import line and add new named imports
            if orig_imp.start_line <= len(result_lines):
                # Reconstruct the import statement lines
                import_text = "".join(result_lines[orig_imp.start_line - 1 : orig_imp.end_line])

                # Find the closing brace of named imports and insert new names before it
                insert_pos = import_text.rfind("}")
                if insert_pos != -1:
                    new_imports_str = ", ".join(
                        f"{name} as {alias}" if alias else name for name, alias in new_names
                    )
                    # Check if there's already content before the brace
                    before_brace = import_text[:insert_pos].rstrip()
                    if before_brace and not before_brace.endswith(","):
                        new_imports_str = ", " + new_imports_str
                    else:
                        new_imports_str = " " + new_imports_str

                    updated_import = import_text[:insert_pos] + new_imports_str + " " + import_text[insert_pos:]
                    # Replace the original import lines with the updated import text (single element)
                    result_lines[orig_imp.start_line - 1 : orig_imp.end_line] = [updated_import]


                    logger.debug(f"Merged imports for {opt_imp.module_path}: added {[n for n, _ in new_names]}")

    return "".join(result_lines)
