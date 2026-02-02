"""Find references for JavaScript/TypeScript functions.

This module provides functionality to find all references (call sites) of a function
across a JavaScript/TypeScript codebase. Similar to Jedi's find_references for Python,
this uses tree-sitter to parse and analyze code.

Key features:
- Finds all call sites of a function across multiple files
- Handles various import patterns (named, default, namespace, re-exports, aliases)
- Supports both ES modules and CommonJS
- Handles memoized functions, callbacks, and method calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

    from codeflash.languages.treesitter_utils import ImportInfo, TreeSitterAnalyzer

from codeflash.discovery.functions_to_optimize import FunctionToOptimize

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """Represents a reference (call site) to a function."""

    file_path: Path  # File containing the reference
    line: int  # 1-indexed line number
    column: int  # 0-indexed column number
    end_line: int  # 1-indexed end line
    end_column: int  # 0-indexed end column
    context: str  # The line of code containing the reference
    reference_type: str  # Type: "call", "callback", "memoized", "import", "reexport"
    import_name: str | None  # Name used to import the function (may differ from original)
    caller_function: str | None = None  # Name of the function containing this reference


@dataclass
class ExportedFunction:
    """Represents how a function is exported from its source file."""

    function_name: str  # The local function name
    export_name: str | None  # The name it's exported as (may differ)
    is_default: bool  # Whether it's a default export
    file_path: Path  # The source file


@dataclass
class ReferenceSearchContext:
    """Context for tracking visited files during reference search."""

    visited_files: set[Path] = field(default_factory=set)
    max_files: int = 1000  # Limit to prevent runaway searches


class ReferenceFinder:
    """Finds all references to a function across a JavaScript/TypeScript codebase.

    This class provides functionality similar to Jedi's find_references for Python,
    but for JavaScript/TypeScript using tree-sitter.

    Example usage:
        ```python
        from codeflash.languages.javascript.find_references import ReferenceFinder
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize

        func = FunctionToOptimize(
            function_name="myHelper",
            file_path=Path("/my/project/src/utils.ts"),
            parents=[],
            language="javascript"
        )
        finder = ReferenceFinder(project_root=Path("/my/project"))
        references = finder.find_references(func)
        for ref in references:
            print(f"{ref.file_path}:{ref.line} - {ref.context}")
        ```
    """

    # File extensions to search
    EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

    def __init__(self, project_root: Path, exclude_patterns: list[str] | None = None) -> None:
        """Initialize the ReferenceFinder.

        Args:
            project_root: Root directory of the project to search.
            exclude_patterns: Glob patterns of directories/files to exclude.
                              Defaults to ['node_modules', 'dist', 'build', '.git'].

        """
        self.project_root = project_root
        self.exclude_patterns = exclude_patterns or ["node_modules", "dist", "build", ".git", "coverage", "__pycache__"]
        self._file_cache: dict[Path, str] = {}

    def find_references(
        self, function_to_optimize: FunctionToOptimize, include_definition: bool = False, max_files: int = 1000
    ) -> list[Reference]:
        """Find all references to a function across the project.

        Args:
            function_to_optimize: The function to find references for.
            include_definition: Whether to include the function definition itself.
            max_files: Maximum number of files to search (prevents runaway searches).

        Returns:
            List of Reference objects describing each call site.

        """
        from codeflash.languages.treesitter_utils import get_analyzer_for_file

        function_name = function_to_optimize.function_name
        source_file = function_to_optimize.file_path

        references: list[Reference] = []
        context = ReferenceSearchContext(max_files=max_files)

        # Step 1: Analyze how the function is exported from its source file
        source_code = self._read_file(source_file)
        if source_code is None:
            logger.warning("Could not read source file: %s", source_file)
            return references

        analyzer = get_analyzer_for_file(source_file)
        exported = self._analyze_exports(function_to_optimize, source_file, source_code, analyzer)

        if not exported:
            logger.debug("Function %s is not exported from %s", function_name, source_file)
            # Still search in same file for internal references
            same_file_refs = self._find_references_in_file(
                source_file, source_code, function_name, None, analyzer, include_self=not include_definition
            )
            references.extend(same_file_refs)
            return references

        # Step 2: Find all files that might import from the source file
        context.visited_files.add(source_file)

        # Track files that re-export our function (we'll search for imports to these too)
        reexport_files: list[tuple[Path, str]] = []  # (file_path, export_name)

        # Step 3: Search all project files for imports and calls
        # We use a separate set for files checked for re-exports to avoid duplicate work
        checked_for_reexports: set[Path] = set()

        for file_path in self._iter_project_files():
            if file_path in context.visited_files:
                continue
            if len(context.visited_files) >= context.max_files:
                logger.warning("Reached max file limit (%d), stopping search", max_files)
                break

            file_code = self._read_file(file_path)
            if file_code is None:
                continue

            file_analyzer = get_analyzer_for_file(file_path)

            # Check if this file imports from the source file
            imports = file_analyzer.find_imports(file_code)
            import_info = self._find_matching_import(imports, source_file, file_path, exported)

            if import_info:
                # Found an import - mark as visited and search for calls
                context.visited_files.add(file_path)
                import_name, original_import = import_info
                file_refs = self._find_references_in_file(
                    file_path, file_code, function_name, import_name, file_analyzer, include_self=True
                )
                references.extend(file_refs)

            # Always check for re-exports (even without direct import match)
            # This handles the case where a file re-exports from our source file
            if file_path not in checked_for_reexports:
                checked_for_reexports.add(file_path)
                reexport_refs = self._find_reexports_direct(file_path, file_code, source_file, exported, file_analyzer)
                references.extend(reexport_refs)

                # Track re-export files for later searching
                for ref in reexport_refs:
                    reexport_files.append((file_path, ref.import_name))

        # Step 4: Follow re-export chains to find references through re-exports
        for reexport_file, reexport_name in reexport_files:
            # Create a new ExportedFunction for the re-exported function
            reexported = ExportedFunction(
                function_name=reexport_name, export_name=reexport_name, is_default=False, file_path=reexport_file
            )

            # Search for imports to the re-export file
            for file_path in self._iter_project_files():
                if file_path in context.visited_files:
                    continue
                if file_path == reexport_file:
                    continue
                if len(context.visited_files) >= context.max_files:
                    break

                file_code = self._read_file(file_path)
                if file_code is None:
                    continue

                file_analyzer = get_analyzer_for_file(file_path)
                imports = file_analyzer.find_imports(file_code)

                # Check if this file imports from the re-export file
                import_info = self._find_matching_import(imports, reexport_file, file_path, reexported)

                if import_info:
                    context.visited_files.add(file_path)
                    import_name, original_import = import_info
                    file_refs = self._find_references_in_file(
                        file_path, file_code, reexport_name, import_name, file_analyzer, include_self=True
                    )
                    # Avoid duplicates
                    existing_locs = {(r.file_path, r.line, r.column) for r in references}
                    for ref in file_refs:
                        if (ref.file_path, ref.line, ref.column) not in existing_locs:
                            references.append(ref)

        # Step 5: Include references in the same file (internal calls)
        if include_definition or not exported:
            same_file_refs = self._find_references_in_file(
                source_file, source_code, function_name, None, analyzer, include_self=True
            )
            # Filter out duplicate references
            existing_locs = {(r.file_path, r.line, r.column) for r in references}
            for ref in same_file_refs:
                if (ref.file_path, ref.line, ref.column) not in existing_locs:
                    references.append(ref)

        # Step 6: Deduplicate references (same file, line, column)
        seen: set[tuple[Path, int, int]] = set()
        unique_refs: list[Reference] = []
        for ref in references:
            key = (ref.file_path, ref.line, ref.column)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)

        return unique_refs

    def _analyze_exports(
        self, function_to_optimize: FunctionToOptimize, file_path: Path, source_code: str, analyzer: TreeSitterAnalyzer
    ) -> ExportedFunction | None:
        """Analyze how a function is exported from its file.

        For class methods, also checks if the containing class is exported.

        Args:
            function_to_optimize: The function to check.
            file_path: Path to the source file.
            source_code: Source code content.
            analyzer: TreeSitterAnalyzer instance.

        Returns:
            ExportedFunction if the function is exported, None otherwise.

        """
        function_name = function_to_optimize.function_name
        class_name = function_to_optimize.class_name
        is_exported, export_name = analyzer.is_function_exported(source_code, function_name, class_name)

        if not is_exported:
            return None

        return ExportedFunction(
            function_name=function_name,
            export_name=export_name,
            is_default=(export_name == "default"),
            file_path=file_path,
        )

    def _find_matching_import(
        self, imports: list[ImportInfo], source_file: Path, importing_file: Path, exported: ExportedFunction
    ) -> tuple[str, ImportInfo] | None:
        """Find if any import in a file imports the target function.

        Args:
            imports: List of imports in the file.
            source_file: Path to the file containing the function definition.
            importing_file: Path to the file being checked for imports.
            exported: Information about how the function is exported.

        Returns:
            Tuple of (imported_name, ImportInfo) if found, None otherwise.

        """
        from codeflash.languages.javascript.import_resolver import ImportResolver

        resolver = ImportResolver(self.project_root)

        for imp in imports:
            # Resolve the import to see if it points to our source file
            resolved = resolver.resolve_import(imp, importing_file)
            if resolved is None:
                continue

            if resolved.file_path != source_file:
                continue

            # This import is from our source file - check if it imports our function
            if exported.is_default:
                # Default export - check default import
                if imp.default_import:
                    return (imp.default_import, imp)
                # Also check namespace import
                if imp.namespace_import:
                    return (f"{imp.namespace_import}.default", imp)
            else:
                # Named export - check named imports
                export_name = exported.export_name or exported.function_name
                for name, alias in imp.named_imports:
                    if name == export_name:
                        return (alias if alias else name, imp)

                # Check namespace import
                if imp.namespace_import:
                    return (f"{imp.namespace_import}.{export_name}", imp)

                # Handle CommonJS default import used as namespace
                # e.g., const helpers = require('./helpers'); helpers.processConfig()
                # In this case, default_import acts like a namespace
                if imp.default_import and not imp.named_imports:
                    return (f"{imp.default_import}.{export_name}", imp)

        return None

    def _find_references_in_file(
        self,
        file_path: Path,
        source_code: str,
        function_name: str,
        import_name: str | None,
        analyzer: TreeSitterAnalyzer,
        include_self: bool = True,
    ) -> list[Reference]:
        """Find all references to a function within a single file.

        Args:
            file_path: Path to the file to search.
            source_code: Source code content.
            function_name: Original function name.
            import_name: Name the function is imported as (may be different).
            analyzer: TreeSitterAnalyzer instance.
            include_self: Whether to include references in the file.

        Returns:
            List of Reference objects.

        """
        references: list[Reference] = []
        source_bytes = source_code.encode("utf8")
        tree = analyzer.parse(source_bytes)
        lines = source_code.splitlines()

        # The name to search for (either imported name or original)
        search_name = import_name if import_name else function_name

        # Handle namespace imports (e.g., "utils.helper")
        if "." in search_name:
            namespace, member = search_name.split(".", 1)
            self._find_member_calls(tree.root_node, source_bytes, lines, file_path, namespace, member, references, None)
        else:
            # Find direct calls and other reference types
            self._find_identifier_references(
                tree.root_node, source_bytes, lines, file_path, search_name, function_name, references, None
            )

        return references

    def _find_identifier_references(
        self,
        node: Node,
        source_bytes: bytes,
        lines: list[str],
        file_path: Path,
        search_name: str,
        original_name: str,
        references: list[Reference],
        current_function: str | None,
    ) -> None:
        """Recursively find references to an identifier in the AST.

        Args:
            node: Current tree-sitter node.
            source_bytes: Source code as bytes.
            lines: Source code split into lines.
            file_path: Path to the file.
            search_name: Name to search for.
            original_name: Original function name.
            references: List to append references to.
            current_function: Name of the containing function (for context).

        """
        # Track current function context
        new_current_function = current_function
        if node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                new_current_function = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
        elif node.type in ("variable_declarator",):
            # Arrow function or function expression assigned to variable
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if name_node and value_node and value_node.type in ("arrow_function", "function_expression"):
                new_current_function = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

        # Check for call expressions
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and func_node.type == "identifier":
                name = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf8")
                if name == search_name:
                    ref = self._create_reference(file_path, func_node, lines, "call", search_name, current_function)
                    references.append(ref)

        # Check for identifiers used as callbacks or passed as arguments
        elif node.type == "identifier":
            name = source_bytes[node.start_byte : node.end_byte].decode("utf8")
            if name == search_name:
                parent = node.parent
                # Determine reference type based on context
                ref_type = self._determine_reference_type(node, parent, source_bytes)
                if ref_type:
                    ref = self._create_reference(file_path, node, lines, ref_type, search_name, current_function)
                    references.append(ref)

        # Recurse into children
        for child in node.children:
            self._find_identifier_references(
                child, source_bytes, lines, file_path, search_name, original_name, references, new_current_function
            )

    def _find_member_calls(
        self,
        node: Node,
        source_bytes: bytes,
        lines: list[str],
        file_path: Path,
        namespace: str,
        member: str,
        references: list[Reference],
        current_function: str | None,
    ) -> None:
        """Find calls to namespace.member (e.g., utils.helper()).

        Args:
            node: Current tree-sitter node.
            source_bytes: Source code as bytes.
            lines: Source code split into lines.
            file_path: Path to the file.
            namespace: The namespace/object name.
            member: The member/property name.
            references: List to append references to.
            current_function: Name of the containing function.

        """
        # Track current function context
        new_current_function = current_function
        if node.type in ("function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                new_current_function = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

        # Check for call expressions with member access
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and func_node.type == "member_expression":
                obj_node = func_node.child_by_field_name("object")
                prop_node = func_node.child_by_field_name("property")

                if obj_node and prop_node:
                    obj_name = source_bytes[obj_node.start_byte : obj_node.end_byte].decode("utf8")
                    prop_name = source_bytes[prop_node.start_byte : prop_node.end_byte].decode("utf8")

                    if obj_name == namespace and prop_name == member:
                        ref = self._create_reference(
                            file_path, func_node, lines, "call", f"{namespace}.{member}", current_function
                        )
                        references.append(ref)

        # Also check for member expression used as callback
        elif node.type == "member_expression":
            obj_node = node.child_by_field_name("object")
            prop_node = node.child_by_field_name("property")

            if obj_node and prop_node:
                obj_name = source_bytes[obj_node.start_byte : obj_node.end_byte].decode("utf8")
                prop_name = source_bytes[prop_node.start_byte : prop_node.end_byte].decode("utf8")

                if obj_name == namespace and prop_name == member:
                    parent = node.parent
                    if parent and parent.type != "call_expression":
                        ref_type = self._determine_reference_type(node, parent, source_bytes)
                        if ref_type:
                            ref = self._create_reference(
                                file_path, node, lines, ref_type, f"{namespace}.{member}", current_function
                            )
                            references.append(ref)

        # Recurse into children
        for child in node.children:
            self._find_member_calls(
                child, source_bytes, lines, file_path, namespace, member, references, new_current_function
            )

    def _determine_reference_type(self, node: Node, parent: Node | None, source_bytes: bytes) -> str | None:
        """Determine the type of reference based on AST context.

        Args:
            node: The identifier node.
            parent: The parent node.
            source_bytes: Source code as bytes.

        Returns:
            Reference type string or None if this isn't a valid reference.

        """
        if parent is None:
            return None

        # Skip import statements
        if parent.type in ("import_specifier", "import_clause", "named_imports"):
            return None

        # Skip function declarations (the function name itself)
        if parent.type in ("function_declaration", "method_definition"):
            name_node = parent.child_by_field_name("name")
            if name_node and name_node.id == node.id:
                return None

        # Skip variable declarations where this is being defined
        if parent.type == "variable_declarator":
            name_node = parent.child_by_field_name("name")
            if name_node and name_node.id == node.id:
                return None

        # Skip export specifiers
        if parent.type == "export_specifier":
            return None

        # Check if passed as argument (callback or memoized)
        if parent.type == "arguments":
            # Check if grandparent is a memoize call
            grandparent = parent.parent
            if grandparent and grandparent.type == "call_expression":
                func_node = grandparent.child_by_field_name("function")
                if func_node:
                    func_name = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf8")
                    if any(m in func_name.lower() for m in ["memoize", "memo", "cache"]):
                        return "memoized"
            return "callback"

        # Check if used in array (often callback patterns)
        if parent.type == "array":
            return "callback"

        # Check if passed to memoize/memoization functions (direct call check)
        if parent.type == "call_expression":
            func_node = parent.child_by_field_name("function")
            if func_node:
                func_name = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf8")
                if any(m in func_name.lower() for m in ["memoize", "memo", "cache"]):
                    return "memoized"

        # Check if used in a call expression as the function
        if parent.type == "call_expression":
            func_node = parent.child_by_field_name("function")
            if func_node and func_node.id == node.id:
                return "call"

        # Check if assigned to a property
        if parent.type in ("pair", "property"):
            return "property"

        # Check if part of member expression (method call setup)
        if parent.type == "member_expression":
            obj_node = parent.child_by_field_name("object")
            if obj_node and obj_node.id == node.id:
                # This is the object in obj.method
                return None  # We'll catch the actual call elsewhere

        # Generic reference
        return "reference"

    def _create_reference(
        self,
        file_path: Path,
        node: Node,
        lines: list[str],
        ref_type: str,
        import_name: str,
        caller_function: str | None,
    ) -> Reference:
        """Create a Reference object from a node.

        Args:
            file_path: Path to the file.
            node: The tree-sitter node.
            lines: Source code lines.
            ref_type: Type of reference.
            import_name: Name the function was imported as.
            caller_function: Name of the containing function.

        Returns:
            A Reference object.

        """
        line_num = node.start_point[0] + 1  # 1-indexed
        context = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""

        return Reference(
            file_path=file_path,
            line=line_num,
            column=node.start_point[1],
            end_line=node.end_point[0] + 1,
            end_column=node.end_point[1],
            context=context.strip(),
            reference_type=ref_type,
            import_name=import_name,
            caller_function=caller_function,
        )

    def _find_reexports(
        self,
        file_path: Path,
        source_code: str,
        exported: ExportedFunction,
        analyzer: TreeSitterAnalyzer,
        context: ReferenceSearchContext,
    ) -> list[Reference]:
        """Find re-exports of the function.

        Re-exports look like: export { helper } from './utils'

        Args:
            file_path: Path to the file being checked.
            source_code: Source code content.
            exported: Information about the original export.
            analyzer: TreeSitterAnalyzer instance.
            context: Search context.

        Returns:
            List of Reference objects for re-exports.

        """
        references: list[Reference] = []
        exports = analyzer.find_exports(source_code)
        lines = source_code.splitlines()

        for exp in exports:
            if not exp.is_reexport:
                continue

            # Check if this re-exports our function
            export_name = exported.export_name or exported.function_name
            for name, alias in exp.exported_names:
                if name == export_name:
                    # This is a re-export of our function
                    # Create a reference with the line info from the export
                    context_line = lines[exp.start_line - 1] if exp.start_line <= len(lines) else ""
                    ref = Reference(
                        file_path=file_path,
                        line=exp.start_line,
                        column=0,
                        end_line=exp.end_line,
                        end_column=0,
                        context=context_line.strip(),
                        reference_type="reexport",
                        import_name=alias if alias else name,
                        caller_function=None,
                    )
                    references.append(ref)

        return references

    def _find_reexports_direct(
        self,
        file_path: Path,
        source_code: str,
        source_file: Path,
        exported: ExportedFunction,
        analyzer: TreeSitterAnalyzer,
    ) -> list[Reference]:
        """Find re-exports that directly reference our source file.

        This method checks if a file has re-export statements that
        reference our source file.

        Args:
            file_path: Path to the file being checked.
            source_code: Source code content.
            source_file: The original source file we're looking for references to.
            exported: Information about the original export.
            analyzer: TreeSitterAnalyzer instance.

        Returns:
            List of Reference objects for re-exports.

        """
        from codeflash.languages.javascript.import_resolver import ImportResolver

        references: list[Reference] = []
        exports = analyzer.find_exports(source_code)
        lines = source_code.splitlines()
        resolver = ImportResolver(self.project_root)

        for exp in exports:
            if not exp.is_reexport or not exp.reexport_source:
                continue

            # Create a fake ImportInfo to resolve the re-export source
            from codeflash.languages.treesitter_utils import ImportInfo

            fake_import = ImportInfo(
                module_path=exp.reexport_source,
                default_import=None,
                named_imports=[],
                namespace_import=None,
                is_type_only=False,
                start_line=exp.start_line,
                end_line=exp.end_line,
            )

            resolved = resolver.resolve_import(fake_import, file_path)
            if resolved is None or resolved.file_path != source_file:
                continue

            # This file re-exports from our source file
            export_name = exported.export_name or exported.function_name
            for name, alias in exp.exported_names:
                if name == export_name:
                    context_line = lines[exp.start_line - 1] if exp.start_line <= len(lines) else ""
                    ref = Reference(
                        file_path=file_path,
                        line=exp.start_line,
                        column=0,
                        end_line=exp.end_line,
                        end_column=0,
                        context=context_line.strip(),
                        reference_type="reexport",
                        import_name=alias if alias else name,
                        caller_function=None,
                    )
                    references.append(ref)

        return references

    def _iter_project_files(self) -> list[Path]:
        """Iterate over all JavaScript/TypeScript files in the project.

        Returns:
            List of file paths to search.

        """
        files: list[Path] = []

        for ext in self.EXTENSIONS:
            for file_path in self.project_root.rglob(f"*{ext}"):
                # Check exclusion patterns
                if self._should_exclude(file_path):
                    continue
                files.append(file_path)

        return files

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from search.

        Args:
            file_path: Path to check.

        Returns:
            True if the file should be excluded.

        """
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        return False

    def _read_file(self, file_path: Path) -> str | None:
        """Read a file's contents with caching.

        Args:
            file_path: Path to the file.

        Returns:
            File contents or None if unreadable.

        """
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        try:
            content = file_path.read_text(encoding="utf-8")
            self._file_cache[file_path] = content
            return content
        except Exception as e:
            logger.debug("Could not read file %s: %s", file_path, e)
            return None


def find_references(
    function_to_optimize: FunctionToOptimize, project_root: Path | None = None, max_files: int = 1000
) -> list[Reference]:
    """Convenience function to find all references to a function.

    This is a simple wrapper around ReferenceFinder for common use cases.

    Args:
        function_to_optimize: The function to find references for.
        project_root: Root directory of the project. If None, uses source_file's parent.
        max_files: Maximum number of files to search.

    Returns:
        List of Reference objects describing each call site.

    Example:
        ```python
        from pathlib import Path
        from codeflash.languages.javascript.find_references import find_references
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize

        func = FunctionToOptimize(
            function_name="myHelper", file_path=Path("/my/project/src/utils.ts"), parents=[], language="javascript"
        )
        refs = find_references(func, project_root=Path("/my/project"))
        for ref in refs:
            print(f"{ref.file_path}:{ref.line}:{ref.column} - {ref.reference_type}")
        ```

    """
    if project_root is None:
        project_root = function_to_optimize.file_path.parent

    finder = ReferenceFinder(project_root)
    return finder.find_references(function_to_optimize, max_files=max_files)
