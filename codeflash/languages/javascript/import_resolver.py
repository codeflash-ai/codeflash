"""Import resolution for JavaScript/TypeScript.

This module provides utilities to resolve JavaScript/TypeScript import paths
to actual file paths, enabling multi-file context extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import HelperFunction
    from codeflash.languages.treesitter_utils import ImportInfo, TreeSitterAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ResolvedImport:
    """Result of resolving an import to a file path."""

    file_path: Path  # Resolved absolute file path
    module_path: str  # Original import path (e.g., './utils')
    imported_names: list[str]  # Names imported (for named imports)
    is_default_import: bool  # Whether it's a default import
    is_namespace_import: bool  # Whether it's import * as X
    namespace_name: str | None  # The namespace alias (X in import * as X)


class ImportResolver:
    """Resolves JavaScript/TypeScript import paths to file paths."""

    # Supported extensions in resolution order (prefer TS over JS)
    EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

    def __init__(self, project_root: Path) -> None:
        """Initialize the resolver.

        Args:
            project_root: Root directory of the project.

        """
        # Resolve to real path to handle macOS symlinks like /var -> /private/var
        self.project_root = project_root.resolve()
        self._resolution_cache: dict[tuple[Path, str], Path | None] = {}

    def resolve_import(self, import_info: ImportInfo, source_file: Path) -> ResolvedImport | None:
        """Resolve an import to its actual file path.

        Args:
            import_info: The import statement information.
            source_file: The file containing the import statement.

        Returns:
            ResolvedImport if resolution successful, None otherwise.

        """
        module_path = import_info.module_path

        # Skip external packages (node_modules)
        if self._is_external_package(module_path):
            logger.debug("Skipping external package: %s", module_path)
            return None

        # Check cache
        cache_key = (source_file, module_path)
        if cache_key in self._resolution_cache:
            cached_path = self._resolution_cache[cache_key]
            if cached_path is None:
                return None
            return self._build_resolved_import(import_info, cached_path)

        # Resolve the path
        resolved_path = self._resolve_module_path(module_path, source_file.parent)

        # Cache the result
        self._resolution_cache[cache_key] = resolved_path

        if resolved_path is None:
            logger.debug("Could not resolve import: %s from %s", module_path, source_file)
            return None

        return self._build_resolved_import(import_info, resolved_path)

    def _build_resolved_import(self, import_info: ImportInfo, resolved_path: Path) -> ResolvedImport:
        """Build a ResolvedImport from import info and resolved path."""
        imported_names = []

        # Collect named imports
        for name, alias in import_info.named_imports:
            imported_names.append(alias if alias else name)

        # Add default import if present
        if import_info.default_import:
            imported_names.append(import_info.default_import)

        return ResolvedImport(
            file_path=resolved_path,
            module_path=import_info.module_path,
            imported_names=imported_names,
            is_default_import=import_info.default_import is not None,
            is_namespace_import=import_info.namespace_import is not None,
            namespace_name=import_info.namespace_import,
        )

    def _resolve_module_path(self, module_path: str, source_dir: Path) -> Path | None:
        """Resolve a module path to an absolute file path.

        Args:
            module_path: The import path (e.g., './utils', '../lib/helper').
            source_dir: Directory of the file containing the import.

        Returns:
            Resolved absolute path, or None if not found.

        """
        # Handle relative imports
        if module_path.startswith("."):
            return self._resolve_relative_import(module_path, source_dir)

        # Handle absolute imports (starting with /)
        if module_path.startswith("/"):
            return self._resolve_absolute_import(module_path)

        # Bare imports (e.g., 'lodash') are external packages
        return None

    def _resolve_relative_import(self, module_path: str, source_dir: Path) -> Path | None:
        """Resolve relative imports like ./utils or ../lib/helper.

        Args:
            module_path: The relative import path.
            source_dir: Directory to resolve from.

        Returns:
            Resolved absolute path, or None if not found.

        """
        # Compute base path
        base_path = (source_dir / module_path).resolve()

        # Check if path is within project
        try:
            base_path.relative_to(self.project_root)
        except ValueError:
            logger.debug("Import path outside project root: %s", base_path)
            return None

        # If the path already has an extension, try it directly first
        if base_path.suffix in self.EXTENSIONS:
            if base_path.exists() and base_path.is_file():
                return base_path
            # TypeScript allows importing .ts files with .js extension
            if base_path.suffix == ".js":
                ts_path = base_path.with_suffix(".ts")
                if ts_path.exists() and ts_path.is_file():
                    return ts_path

        # Try adding extensions
        resolved = self._try_extensions(base_path)
        if resolved:
            return resolved

        # Try as directory with index file
        resolved = self._try_index_file(base_path)
        if resolved:
            return resolved

        return None

    def _resolve_absolute_import(self, module_path: str) -> Path | None:
        """Resolve absolute imports starting with /.

        Args:
            module_path: The absolute import path.

        Returns:
            Resolved absolute path, or None if not found.

        """
        # Treat as relative to project root
        base_path = (self.project_root / module_path.lstrip("/")).resolve()

        # Try adding extensions
        resolved = self._try_extensions(base_path)
        if resolved:
            return resolved

        # Try as directory with index file
        resolved = self._try_index_file(base_path)
        if resolved:
            return resolved

        return None

    def _try_extensions(self, base_path: Path) -> Path | None:
        """Try adding various extensions to find the actual file.

        Args:
            base_path: The path without extension.

        Returns:
            Path if file found with an extension, None otherwise.

        """
        # If base_path already exists as file
        if base_path.exists() and base_path.is_file():
            return base_path

        # Try each extension in order
        for ext in self.EXTENSIONS:
            path_with_ext = base_path.with_suffix(ext)
            if path_with_ext.exists() and path_with_ext.is_file():
                return path_with_ext

        # Also try adding extension to paths that already have one
        # (e.g., './utils.js' might need to become './utils.js.ts' in some setups)
        # This is rare but some bundlers support it
        if base_path.suffix:
            for ext in self.EXTENSIONS:
                path_with_double_ext = Path(str(base_path) + ext)
                if path_with_double_ext.exists() and path_with_double_ext.is_file():
                    return path_with_double_ext

        return None

    def _try_index_file(self, dir_path: Path) -> Path | None:
        """Try resolving to index file in a directory.

        Args:
            dir_path: The directory path to check.

        Returns:
            Path to index file if found, None otherwise.

        """
        if not dir_path.exists() or not dir_path.is_dir():
            return None

        # Try index files with each extension
        for ext in self.EXTENSIONS:
            index_path = dir_path / f"index{ext}"
            if index_path.exists() and index_path.is_file():
                return index_path

        return None

    def _is_external_package(self, module_path: str) -> bool:
        """Check if import refers to an external package (node_modules).

        Args:
            module_path: The import module path.

        Returns:
            True if this is an external package import.

        """
        # Relative imports are not external
        if module_path.startswith("."):
            return False

        # Absolute imports (starting with /) are project-internal
        if module_path.startswith("/"):
            return False

        # Bare imports without ./ or ../ are external packages
        # This includes:
        # - 'lodash'
        # - '@company/utils'
        # - 'react'
        # - 'fs' (Node.js built-ins)
        return True


@dataclass
class HelperSearchContext:
    """Context for recursive helper search."""

    visited_files: set[Path] = field(default_factory=set)
    visited_functions: set[tuple[Path, str]] = field(default_factory=set)
    current_depth: int = 0
    max_depth: int = 2


class MultiFileHelperFinder:
    """Finds helper functions across multiple files."""

    DEFAULT_MAX_DEPTH = 2  # Target → helpers → helpers of helpers

    def __init__(self, project_root: Path, import_resolver: ImportResolver) -> None:
        """Initialize the finder.

        Args:
            project_root: Root directory of the project.
            import_resolver: ImportResolver instance for resolving imports.

        """
        self.project_root = project_root
        self.import_resolver = import_resolver

    def find_helpers(
        self,
        function: FunctionToOptimize,
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[ImportInfo],
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> dict[Path, list[HelperFunction]]:
        """Find all helper functions including cross-file dependencies.

        Args:
            function: The target function to find helpers for.
            source: Source code of the file containing the function.
            analyzer: TreeSitterAnalyzer for parsing.
            imports: List of imports in the source file.
            max_depth: Maximum recursion depth for finding helpers of helpers.

        Returns:
            Dictionary mapping file paths to lists of helper functions.

        """
        context = HelperSearchContext(max_depth=max_depth)
        context.visited_files.add(function.file_path)

        # Find all function calls within the target function
        all_functions = analyzer.find_functions(source, include_methods=True)
        target_func = None
        for func in all_functions:
            if func.name == function.function_name and func.start_line == function.starting_line:
                target_func = func
                break

        if not target_func:
            return {}

        calls = analyzer.find_function_calls(source, target_func)

        # Match calls to imports
        call_to_import = self._match_calls_to_imports(calls, imports)

        # Find helpers from imported modules
        results: dict[Path, list[HelperFunction]] = {}

        for import_info, actual_name in call_to_import.values():
            # Resolve the import to a file path
            resolved = self.import_resolver.resolve_import(import_info, function.file_path)
            if resolved is None:
                continue

            # Skip if already visited
            key = (resolved.file_path, actual_name)
            if key in context.visited_functions:
                continue
            context.visited_functions.add(key)

            # Extract the helper function from the resolved file
            helper = self._extract_helper_from_file(resolved.file_path, actual_name, analyzer)
            if helper:
                if resolved.file_path not in results:
                    results[resolved.file_path] = []
                results[resolved.file_path].append(helper)

                # Recursively find helpers of this helper (if depth allows)
                if context.current_depth < context.max_depth:
                    nested_results = self._find_helpers_recursive(
                        resolved.file_path,
                        helper,
                        HelperSearchContext(
                            visited_files=context.visited_files.copy(),
                            visited_functions=context.visited_functions.copy(),
                            current_depth=context.current_depth + 1,
                            max_depth=context.max_depth,
                        ),
                    )
                    # Merge nested results
                    for path, helpers in nested_results.items():
                        if path not in results:
                            results[path] = []
                        results[path].extend(helpers)

        return results

    def _match_calls_to_imports(self, calls: set[str], imports: list[ImportInfo]) -> dict[str, tuple[ImportInfo, str]]:
        """Match function calls to their import sources.

        Args:
            calls: Set of function call names found in the code.
            imports: List of import statements.

        Returns:
            Dictionary mapping call names to (ImportInfo, actual_function_name) tuples.

        """
        matches: dict[str, tuple[ImportInfo, str]] = {}

        for call in calls:
            # Check for namespace calls (e.g., utils.helper)
            if "." in call:
                namespace, func_name = call.split(".", 1)
                for imp in imports:
                    if imp.namespace_import == namespace:
                        matches[call] = (imp, func_name)
                        break
            else:
                # Check for direct imports
                for imp in imports:
                    # Check default import
                    if imp.default_import == call:
                        matches[call] = (imp, "default")
                        break

                    # Check named imports
                    for name, alias in imp.named_imports:
                        if (alias and alias == call) or (not alias and name == call):
                            matches[call] = (imp, name)
                            break

        return matches

    def _extract_helper_from_file(
        self, file_path: Path, function_name: str, analyzer: TreeSitterAnalyzer
    ) -> HelperFunction | None:
        """Extract a helper function from a resolved file.

        Args:
            file_path: Path to the file containing the function.
            function_name: Name of the function to extract.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            HelperFunction if found, None otherwise.

        """
        from codeflash.languages.base import HelperFunction
        from codeflash.languages.treesitter_utils import get_analyzer_for_file

        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return None

        # Get analyzer for this file type
        file_analyzer = get_analyzer_for_file(file_path)

        # Split source into lines for JSDoc extraction
        lines = source.splitlines(keepends=True)

        # Handle "default" export - look for default exported function
        if function_name == "default":
            # Find the default export
            functions = file_analyzer.find_functions(source, include_methods=True)
            # For now, return first function if looking for default
            # TODO: Implement proper default export detection
            for func in functions:
                # Extract source including JSDoc if present
                effective_start = func.doc_start_line or func.start_line
                helper_lines = lines[effective_start - 1 : func.end_line]
                helper_source = "".join(helper_lines)

                return HelperFunction(
                    name=func.name,
                    qualified_name=func.name,
                    file_path=file_path,
                    source_code=helper_source,
                    start_line=effective_start,
                    end_line=func.end_line,
                )
            return None

        # Find the function by name
        functions = file_analyzer.find_functions(source, include_methods=True)
        for func in functions:
            if func.name == function_name:
                # Extract source including JSDoc if present
                effective_start = func.doc_start_line or func.start_line
                helper_lines = lines[effective_start - 1 : func.end_line]
                helper_source = "".join(helper_lines)

                return HelperFunction(
                    name=func.name,
                    qualified_name=func.name,
                    file_path=file_path,
                    source_code=helper_source,
                    start_line=effective_start,
                    end_line=func.end_line,
                )

        logger.debug("Function %s not found in %s", function_name, file_path)
        return None

    def _find_helpers_recursive(
        self, file_path: Path, helper: HelperFunction, context: HelperSearchContext
    ) -> dict[Path, list[HelperFunction]]:
        """Recursively find helpers of a helper function.

        Args:
            file_path: Path to the file containing the helper.
            helper: The helper function to analyze.
            context: Search context with visited tracking and depth limit.

        Returns:
            Dictionary mapping file paths to lists of helper functions.

        """
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.languages.treesitter_utils import get_analyzer_for_file

        if context.current_depth >= context.max_depth:
            return {}

        if file_path in context.visited_files:
            return {}
        context.visited_files.add(file_path)

        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return {}

        # Get analyzer and imports for this file
        analyzer = get_analyzer_for_file(file_path)
        imports = analyzer.find_imports(source)

        # Create FunctionToOptimize for the helper
        func_info = FunctionToOptimize(
            function_name=helper.name,
            file_path=file_path,
            parents=[],
            starting_line=helper.start_line,
            ending_line=helper.end_line,
        )

        # Recursively find helpers
        return self.find_helpers(
            function=func_info,
            source=source,
            analyzer=analyzer,
            imports=imports,
            max_depth=context.max_depth - context.current_depth,
        )
