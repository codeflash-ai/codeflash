"""JavaScript language support implementation.

This module implements the LanguageSupport protocol for JavaScript,
using tree-sitter for code analysis and Jest for test execution.
"""

from __future__ import annotations

import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import CodeContext, FunctionFilterCriteria, HelperFunction, Language, TestInfo, TestResult
from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer, TreeSitterLanguage, get_analyzer_for_file
from codeflash.languages.registry import register_language
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeflash.languages.base import ReferenceInfo
    from codeflash.languages.javascript.treesitter import TypeDefinition

logger = logging.getLogger(__name__)


@register_language
class JavaScriptSupport:
    """JavaScript language support implementation.

    This class implements the LanguageSupport protocol for JavaScript/JSX files,
    using tree-sitter for code analysis and Jest for test execution.
    """

    # === Properties ===

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.JAVASCRIPT

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by JavaScript."""
        return (".js", ".jsx", ".mjs", ".cjs")

    @property
    def default_file_extension(self) -> str:
        """Default file extension for JavaScript."""
        return ".js"

    @property
    def test_framework(self) -> str:
        """Primary test framework for JavaScript."""
        from codeflash.languages.test_framework import get_js_test_framework_or_default

        return get_js_test_framework_or_default()

    @property
    def comment_prefix(self) -> str:
        return "//"

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        """Find all optimizable functions in a JavaScript file.

        Uses tree-sitter to parse the file and find functions.

        Args:
            file_path: Path to the JavaScript file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionToOptimize objects for discovered functions.

        """
        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return []

        try:
            analyzer = get_analyzer_for_file(file_path)
            tree_functions = analyzer.find_functions(
                source, include_methods=criteria.include_methods, include_arrow_functions=True, require_name=True
            )

            functions: list[FunctionToOptimize] = []
            for func in tree_functions:
                # Check for return statement if required
                if criteria.require_return and not analyzer.has_return_statement(func, source):
                    continue

                # Check async filter
                if not criteria.include_async and func.is_async:
                    continue

                # Skip non-exported functions (can't be imported in tests)
                # Exception: nested functions and methods are allowed if their parent is exported
                if not func.is_exported and not func.parent_function:
                    logger.debug(f"Skipping non-exported function: {func.name}")  # noqa: G004
                    continue

                # Build parents list
                parents: list[FunctionParent] = []
                if func.class_name:
                    parents.append(FunctionParent(name=func.class_name, type="ClassDef"))
                if func.parent_function:
                    parents.append(FunctionParent(name=func.parent_function, type="FunctionDef"))

                functions.append(
                    FunctionToOptimize(
                        function_name=func.name,
                        file_path=file_path,
                        parents=parents,
                        starting_line=func.start_line,
                        ending_line=func.end_line,
                        starting_col=func.start_col,
                        ending_col=func.end_col,
                        is_async=func.is_async,
                        is_method=func.is_method,
                        language=str(self.language),
                        doc_start_line=func.doc_start_line,
                    )
                )

            return functions

        except Exception as e:
            logger.warning("Failed to parse %s: %s", file_path, e)
            return []

    def discover_functions_from_source(self, source: str, file_path: Path | None = None) -> list[FunctionToOptimize]:
        """Find all functions in source code string.

        Uses tree-sitter to parse the source and find functions.

        Args:
            source: The source code to analyze.
            file_path: Optional file path for context (used for language detection).

        Returns:
            List of FunctionToOptimize objects for discovered functions.

        """
        try:
            # Use JavaScript analyzer by default, or detect from file path
            if file_path:
                analyzer = get_analyzer_for_file(file_path)
            else:
                analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

            tree_functions = analyzer.find_functions(
                source, include_methods=True, include_arrow_functions=True, require_name=True
            )

            functions: list[FunctionToOptimize] = []
            for func in tree_functions:
                # Build parents list
                parents: list[FunctionParent] = []
                if func.class_name:
                    parents.append(FunctionParent(name=func.class_name, type="ClassDef"))
                if func.parent_function:
                    parents.append(FunctionParent(name=func.parent_function, type="FunctionDef"))

                functions.append(
                    FunctionToOptimize(
                        function_name=func.name,
                        file_path=file_path or Path("unknown"),
                        parents=parents,
                        starting_line=func.start_line,
                        ending_line=func.end_line,
                        starting_col=func.start_col,
                        ending_col=func.end_col,
                        is_async=func.is_async,
                        is_method=func.is_method,
                        language=str(self.language),
                        doc_start_line=func.doc_start_line,
                    )
                )

            return functions

        except Exception as e:
            logger.warning("Failed to parse source: %s", e)
            return []

    def _get_test_patterns(self) -> list[str]:
        """Get test file patterns for this language.

        Override in subclasses to provide language-specific patterns.

        Returns:
            List of glob patterns for test files.

        """
        return ["*.test.js", "*.test.jsx", "*.spec.js", "*.spec.jsx", "__tests__/**/*.js", "__tests__/**/*.jsx"]

    def discover_tests(
        self, test_root: Path, source_functions: Sequence[FunctionToOptimize]
    ) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis.

        For JavaScript, this uses static analysis to find test files
        and match them to source functions based on imports and function calls.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.

        """
        result: dict[str, list[TestInfo]] = {}

        # Find all test files using language-specific patterns
        test_patterns = self._get_test_patterns()

        test_files: list[Path] = []
        for pattern in test_patterns:
            test_files.extend(test_root.rglob(pattern))

        for test_file in test_files:
            try:
                source = test_file.read_text()
                analyzer = get_analyzer_for_file(test_file)
                imports = analyzer.find_imports(source)

                # Build a set of imported function names
                imported_names: set[str] = set()
                for imp in imports:
                    if imp.default_import:
                        imported_names.add(imp.default_import)
                    for name, alias in imp.named_imports:
                        imported_names.add(alias or name)

                # Find test functions (describe/it/test blocks)
                test_functions = self._find_jest_tests(source, analyzer)

                # Match source functions to tests
                for func in source_functions:
                    if func.function_name in imported_names or func.function_name in source:
                        if func.qualified_name not in result:
                            result[func.qualified_name] = []
                        for test_name in test_functions:
                            result[func.qualified_name].append(
                                TestInfo(test_name=test_name, test_file=test_file, test_class=None)
                            )
            except Exception as e:
                logger.debug("Failed to analyze test file %s: %s", test_file, e)

        return result

    def _find_jest_tests(self, source: str, analyzer: TreeSitterAnalyzer) -> list[str]:
        """Find Jest test function names in source code."""
        test_names: list[str] = []
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)

        self._walk_for_jest_tests(tree.root_node, source_bytes, test_names)
        return test_names

    def _walk_for_jest_tests(self, node: Any, source_bytes: bytes, test_names: list[str]) -> None:
        """Walk tree to find Jest test/it/describe calls."""
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                func_name = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf8")
                if func_name in ("test", "it", "describe"):
                    # Get the first string argument as the test name
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        for child in args_node.children:
                            if child.type == "string":
                                test_name = source_bytes[child.start_byte : child.end_byte].decode("utf8")
                                test_names.append(test_name.strip("'\""))
                                break

        for child in node.children:
            self._walk_for_jest_tests(child, source_bytes, test_names)

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies.

        Uses tree-sitter to analyze imports and find helper functions.

        Args:
            function: The function to extract context for.
            project_root: Root of the project.
            module_root: Root of the module containing the function.

        Returns:
            CodeContext with target code and dependencies.

        """
        try:
            source = function.file_path.read_text()
        except Exception as e:
            logger.exception("Failed to read %s: %s", function.file_path, e)
            return CodeContext(target_code="", target_file=function.file_path, language=Language.JAVASCRIPT)

        # Find imports and helper functions
        analyzer = get_analyzer_for_file(function.file_path)

        # Find the FunctionNode to get doc_start_line for JSDoc inclusion
        tree_functions = analyzer.find_functions(source, include_methods=True, include_arrow_functions=True)
        target_func = None
        for func in tree_functions:
            if func.name == function.function_name and func.start_line == function.starting_line:
                target_func = func
                break

        # Extract the function source, including JSDoc if present
        lines = source.splitlines(keepends=True)
        if function.starting_line and function.ending_line:
            # Use doc_start_line if available, otherwise fall back to start_line
            effective_start = (target_func.doc_start_line if target_func else None) or function.starting_line
            target_lines = lines[effective_start - 1 : function.ending_line]
            target_code = "".join(target_lines)
        else:
            target_code = ""

        imports = analyzer.find_imports(source)

        # Find helper functions called by target (needed before class wrapping to find same-class helpers)
        helpers = self._find_helper_functions(function, source, analyzer, imports, module_root)

        # For class methods, wrap the method in its class definition
        # This is necessary because method definition syntax is only valid inside a class body
        same_class_helper_names: set[str] = set()
        if function.is_method and function.parents:
            class_name = None
            for parent in function.parents:
                if parent.type == "ClassDef":
                    class_name = parent.name
                    break

            if class_name:
                # Find same-class helper methods that need to be included inside the class wrapper
                same_class_helpers = self._find_same_class_helpers(
                    class_name, function.function_name, helpers, tree_functions, lines
                )
                same_class_helper_names = {h[0] for h in same_class_helpers}  # method names

                # Find the class definition in the source to get proper indentation, JSDoc, constructor, and fields
                class_info = self._find_class_definition(source, class_name, analyzer, function.function_name)
                if class_info:
                    class_jsdoc, class_indent, constructor_code, fields_code = class_info
                    # Build the class body with fields, constructor, target method, and same-class helpers
                    class_body_parts = []
                    if fields_code:
                        class_body_parts.append(fields_code)
                    if constructor_code:
                        class_body_parts.append(constructor_code)
                    class_body_parts.append(target_code)
                    # Add same-class helper methods inside the class body
                    for _helper_name, helper_source in same_class_helpers:
                        class_body_parts.append(helper_source)
                    class_body = "\n".join(class_body_parts)

                    # Wrap the method in a class definition with context
                    if class_jsdoc:
                        target_code = (
                            f"{class_jsdoc}\n{class_indent}class {class_name} {{\n{class_body}{class_indent}}}\n"
                        )
                    else:
                        target_code = f"{class_indent}class {class_name} {{\n{class_body}{class_indent}}}\n"
                else:
                    # Fallback: wrap with no indentation, including same-class helpers
                    helper_code = "\n".join(h[1] for h in same_class_helpers)
                    if helper_code:
                        target_code = f"class {class_name} {{\n{target_code}\n{helper_code}}}\n"
                    else:
                        target_code = f"class {class_name} {{\n{target_code}}}\n"

        # Filter out same-class helpers from the helpers list (they're already inside the class wrapper)
        if same_class_helper_names:
            helpers = [h for h in helpers if h.name not in same_class_helper_names]

        # Extract import statements as strings
        import_lines = []
        for imp in imports:
            imp_lines = lines[imp.start_line - 1 : imp.end_line]
            import_lines.append("".join(imp_lines).strip())

        # Extract type definitions for function parameters and class fields
        type_definitions_context, type_definition_names = self._extract_type_definitions_context(
            function=function, source=source, analyzer=analyzer, imports=imports, module_root=module_root
        )

        # Find module-level declarations (global variables/constants) referenced by the function
        # Exclude type definitions that are already included above to avoid duplication
        read_only_context = self._find_referenced_globals(
            target_code=target_code,
            helpers=helpers,
            source=source,
            analyzer=analyzer,
            imports=imports,
            exclude_names=type_definition_names,
        )

        # Combine type definitions with other read-only context
        if type_definitions_context:
            if read_only_context:
                read_only_context = type_definitions_context + "\n\n" + read_only_context
            else:
                read_only_context = type_definitions_context

        # Validate that the extracted code is syntactically valid
        # If not, raise an error to fail the optimization early
        if target_code and not self.validate_syntax(target_code):
            error_msg = (
                f"Extracted code for {function.function_name} is not syntactically valid JavaScript. "
                f"Cannot proceed with optimization."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return CodeContext(
            target_code=target_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context=read_only_context,
            imports=import_lines,
            language=Language.JAVASCRIPT,
        )

    def _find_class_definition(
        self, source: str, class_name: str, analyzer: TreeSitterAnalyzer, target_method_name: str | None = None
    ) -> tuple[str, str, str, str] | None:
        """Find a class definition and extract its JSDoc, indentation, constructor, and fields.

        Args:
            source: The source code to search.
            class_name: The name of the class to find.
            analyzer: TreeSitterAnalyzer for parsing.
            target_method_name: Name of the target method (to exclude from extracted context).

        Returns:
            Tuple of (jsdoc_comment, indentation, constructor_code, fields_code) or None if not found.
            Constructor and fields are included to provide context for method optimization.

        """
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)

        def find_class_node(node):
            """Recursively find a class declaration with the given name."""
            if node.type in ("class_declaration", "class"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    node_name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                    if node_name == class_name:
                        return node
            for child in node.children:
                result = find_class_node(child)
                if result:
                    return result
            return None

        class_node = find_class_node(tree.root_node)
        if not class_node:
            return None

        # Get indentation from the class line
        lines = source.splitlines(keepends=True)
        class_line_idx = class_node.start_point[0]
        if class_line_idx < len(lines):
            class_line = lines[class_line_idx]
            indent = len(class_line) - len(class_line.lstrip())
            indentation = " " * indent
        else:
            indentation = ""

        # Look for preceding JSDoc comment
        jsdoc = ""
        prev_sibling = class_node.prev_named_sibling
        if prev_sibling and prev_sibling.type == "comment":
            comment_text = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf8")
            if comment_text.strip().startswith("/**"):
                jsdoc = comment_text

        # Find class body and extract constructor and fields
        constructor_code = ""
        fields_code = ""

        body_node = class_node.child_by_field_name("body")
        if body_node:
            constructor_code, fields_code = self._extract_class_context(
                body_node, source_bytes, lines, target_method_name
            )

        return (jsdoc, indentation, constructor_code, fields_code)

    def _extract_class_context(
        self, body_node: Any, source_bytes: bytes, lines: list[str], target_method_name: str | None
    ) -> tuple[str, str]:
        """Extract constructor and field declarations from a class body.

        Args:
            body_node: Tree-sitter node for the class body.
            source_bytes: Source code as bytes.
            lines: Source code split into lines.
            target_method_name: Name of the target method to exclude.

        Returns:
            Tuple of (constructor_code, fields_code).

        """
        constructor_parts: list[str] = []
        field_parts: list[str] = []

        for child in body_node.children:
            # Skip braces and the target method
            if child.type in ("{", "}"):
                continue

            # Handle method definitions (including constructor)
            if child.type == "method_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    method_name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

                    # Extract constructor (but not the target method)
                    if method_name == "constructor":
                        # Get start line, check for preceding JSDoc
                        start_line = child.start_point[0]
                        end_line = child.end_point[0]

                        # Look for JSDoc comment before constructor
                        jsdoc_start = start_line
                        prev_sibling = child.prev_named_sibling
                        if prev_sibling and prev_sibling.type == "comment":
                            comment_text = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf8")
                            if comment_text.strip().startswith("/**"):
                                jsdoc_start = prev_sibling.start_point[0]

                        constructor_lines = lines[jsdoc_start : end_line + 1]
                        constructor_parts.append("".join(constructor_lines))

            # Handle public field definitions (class properties)
            # In JS/TS: public_field_definition, field_definition
            elif child.type in ("public_field_definition", "field_definition"):
                start_line = child.start_point[0]
                end_line = child.end_point[0]

                # Look for preceding comment
                comment_start = start_line
                prev_sibling = child.prev_named_sibling
                if prev_sibling and prev_sibling.type == "comment":
                    comment_start = prev_sibling.start_point[0]

                field_lines = lines[comment_start : end_line + 1]
                field_parts.append("".join(field_lines))

        constructor_code = "".join(constructor_parts)
        fields_code = "".join(field_parts)

        return (constructor_code, fields_code)

    def _find_same_class_helpers(
        self,
        class_name: str,
        target_method_name: str,
        helpers: list[HelperFunction],
        tree_functions: list,
        lines: list[str],
    ) -> list[tuple[str, str]]:
        """Find helper methods that belong to the same class as the target method.

        These helpers need to be included inside the class wrapper rather than
        appended outside, because they may use class-specific syntax like 'private'.

        Args:
            class_name: Name of the class containing the target method.
            target_method_name: Name of the target method (to exclude).
            helpers: List of all helper functions found.
            tree_functions: List of FunctionNode from tree-sitter analysis.
            lines: Source code split into lines.

        Returns:
            List of (method_name, source_code) tuples for same-class helpers.

        """
        same_class_helpers: list[tuple[str, str]] = []

        # Build a set of helper names for quick lookup
        helper_names = {h.name for h in helpers}

        # Names to exclude from same-class helpers (target method and constructor)
        exclude_names = {target_method_name, "constructor"}

        # Find methods in tree_functions that belong to the same class and are helpers
        for func in tree_functions:
            if func.class_name == class_name and func.name in helper_names and func.name not in exclude_names:
                # Extract source including JSDoc if present
                effective_start = func.doc_start_line or func.start_line
                helper_lines = lines[effective_start - 1 : func.end_line]
                helper_source = "".join(helper_lines)
                same_class_helpers.append((func.name, helper_source))

        return same_class_helpers

    def _find_helper_functions(
        self,
        function: FunctionToOptimize,
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[Any],
        module_root: Path,
    ) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        This method finds helpers in both the same file and imported files.

        Args:
            function: The target function to find helpers for.
            source: Source code of the file containing the function.
            analyzer: TreeSitterAnalyzer for parsing.
            imports: List of ImportInfo objects from the source file.
            module_root: Root directory of the module/project.

        Returns:
            List of HelperFunction objects from same file and imported files.

        """
        helpers: list[HelperFunction] = []

        # Get all functions in the same file
        all_functions = analyzer.find_functions(source, include_methods=True)

        # Find the target function's tree-sitter node
        target_func = None
        for func in all_functions:
            if func.name == function.function_name and func.start_line == function.starting_line:
                target_func = func
                break

        if not target_func:
            return helpers

        # Find function calls within target
        calls = analyzer.find_function_calls(source, target_func)
        calls_set = set(calls)

        # Split source into lines for JSDoc extraction
        lines = source.splitlines(keepends=True)

        # Match calls to functions in the same file
        for func in all_functions:
            if func.name in calls_set and func.name != function.function_name:
                # Extract source including JSDoc if present
                effective_start = func.doc_start_line or func.start_line
                helper_lines = lines[effective_start - 1 : func.end_line]
                helper_source = "".join(helper_lines)

                helpers.append(
                    HelperFunction(
                        name=func.name,
                        qualified_name=func.name,
                        file_path=function.file_path,
                        source_code=helper_source,
                        start_line=effective_start,  # Start from JSDoc if present
                        end_line=func.end_line,
                    )
                )

        # Find helpers in imported files
        try:
            from codeflash.languages.javascript.import_resolver import ImportResolver, MultiFileHelperFinder

            import_resolver = ImportResolver(module_root)
            helper_finder = MultiFileHelperFinder(module_root, import_resolver)

            cross_file_helpers = helper_finder.find_helpers(
                function=function,
                source=source,
                analyzer=analyzer,
                imports=imports,
                max_depth=2,  # Target → helpers → helpers of helpers
            )

            # Add cross-file helpers to the list
            for file_path, file_helpers in cross_file_helpers.items():
                if file_path != function.file_path:
                    helpers.extend(file_helpers)

        except Exception as e:
            logger.debug("Failed to find cross-file helpers: %s", e)

        return helpers

    def _find_referenced_globals(
        self,
        target_code: str,
        helpers: list[HelperFunction],
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[Any],
        exclude_names: set[str] | None = None,
    ) -> str:
        """Find module-level declarations referenced by the target function and its helpers.

        Args:
            target_code: The target function's source code.
            helpers: List of helper functions.
            source: Full source code of the file.
            analyzer: TreeSitterAnalyzer for parsing.
            imports: List of ImportInfo objects.
            exclude_names: Names to exclude from the result (e.g., type definitions).

        Returns:
            String containing all referenced global declarations.

        """
        if exclude_names is None:
            exclude_names = set()

        # Find all module-level declarations in the source file
        module_declarations = analyzer.find_module_level_declarations(source)

        if not module_declarations:
            return ""

        # Build a set of names that are imported (so we don't include them as globals)
        imported_names: set[str] = set()
        for imp in imports:
            if imp.default_import:
                imported_names.add(imp.default_import)
            if imp.namespace_import:
                imported_names.add(imp.namespace_import)
            for name, alias in imp.named_imports:
                imported_names.add(alias if alias else name)

        # Build a map of declaration name -> declaration info
        decl_map: dict[str, Any] = {}
        for decl in module_declarations:
            # Skip function declarations (they are handled as helpers)
            # Also skip if it's an import or an excluded name (type definitions)
            if decl.name not in imported_names and decl.name not in exclude_names:
                decl_map[decl.name] = decl

        if not decl_map:
            return ""

        # Find all identifiers referenced in the target code
        referenced_in_target = analyzer.find_referenced_identifiers(target_code)

        # Also find identifiers referenced in helper functions
        referenced_in_helpers: set[str] = set()
        for helper in helpers:
            helper_refs = analyzer.find_referenced_identifiers(helper.source_code)
            referenced_in_helpers.update(helper_refs)

        # Combine all referenced identifiers
        all_references = referenced_in_target | referenced_in_helpers

        # Filter to only module-level declarations that are referenced
        referenced_globals: list[Any] = []
        seen_decl_sources: set[str] = set()  # Avoid duplicates for destructuring

        for ref_name in all_references:
            if ref_name in decl_map:
                decl = decl_map[ref_name]
                # Avoid duplicate declarations (same source code)
                if decl.source_code not in seen_decl_sources:
                    referenced_globals.append(decl)
                    seen_decl_sources.add(decl.source_code)

        if not referenced_globals:
            return ""

        # Sort by line number to maintain original order
        referenced_globals.sort(key=lambda d: d.start_line)

        # Build the context string
        global_lines = [decl.source_code for decl in referenced_globals]
        return "\n".join(global_lines)

    def _extract_type_definitions_context(
        self,
        function: FunctionToOptimize,
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[Any],
        module_root: Path,
    ) -> tuple[str, set[str]]:
        """Extract type definitions used by the function for read-only context.

        Finds user-defined types referenced in:
        1. Function parameters
        2. Function return type
        3. Class fields (if the function is a class method)
        4. Types referenced within other type definitions (recursive)

        Then looks up these type definitions in:
        1. The same file
        2. Imported files

        Args:
            function: The target function to analyze.
            source: Source code of the file.
            analyzer: TreeSitterAnalyzer for parsing.
            imports: List of ImportInfo objects.
            module_root: Root directory of the module.

        Returns:
            Tuple of (type definitions string, set of found type names).

        """
        # Extract type names from function parameters and return type
        type_names = analyzer.extract_type_annotations(source, function.function_name, function.starting_line or 1)

        # If this is a class method, also extract types from class fields
        if function.is_method and function.parents:
            for parent in function.parents:
                if parent.type == "ClassDef":
                    field_types = analyzer.extract_class_field_types(source, parent.name)
                    type_names.update(field_types)

        if not type_names:
            return "", set()

        # Find type definitions in the same file
        same_file_definitions = analyzer.find_type_definitions(source)
        found_definitions: list[TypeDefinition] = []

        # Build a map of type name -> definition for same-file types
        same_file_type_map = {defn.name: defn for defn in same_file_definitions}

        # Track which types we've found (avoid duplicates)
        found_type_names: set[str] = set()

        # Recursively find types - including types referenced within type definitions
        types_to_find = set(type_names)
        processed_types: set[str] = set()
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            if not types_to_find:
                break

            new_types_to_find: set[str] = set()
            types_not_in_same_file: set[str] = set()

            for type_name in types_to_find:
                if type_name in processed_types:
                    continue
                processed_types.add(type_name)

                # Look in same file first
                if type_name in same_file_type_map and type_name not in found_type_names:
                    defn = same_file_type_map[type_name]
                    found_definitions.append(defn)
                    found_type_names.add(type_name)
                    # Extract types referenced in this type definition
                    referenced_types = self._extract_types_from_definition(defn.source_code, analyzer)
                    new_types_to_find.update(referenced_types - found_type_names - processed_types)
                elif type_name not in same_file_type_map and type_name not in found_type_names:
                    # Type not found in same file, needs to be looked up in imports
                    types_not_in_same_file.add(type_name)

            # For types not found in same file, look in imported files
            if types_not_in_same_file:
                imported_definitions = self._find_imported_type_definitions(
                    types_not_in_same_file, imports, module_root, function.file_path
                )
                for defn in imported_definitions:
                    if defn.name not in found_type_names:
                        found_definitions.append(defn)
                        found_type_names.add(defn.name)

            types_to_find = new_types_to_find

        if not found_definitions:
            return "", found_type_names

        # Sort by file path and line number for consistent ordering
        found_definitions.sort(key=lambda d: (str(d.file_path or ""), d.start_line))

        # Build the type definitions context string
        # Group by file for better organization
        type_def_parts: list[str] = []
        current_file: Path | None = None

        for defn in found_definitions:
            if defn.file_path and defn.file_path != current_file:
                current_file = defn.file_path
                # Add a comment indicating the source file
                type_def_parts.append(f"// From {current_file.name}")

            type_def_parts.append(defn.source_code)

        return "\n\n".join(type_def_parts), found_type_names

    def _extract_types_from_definition(self, type_source: str, analyzer: TreeSitterAnalyzer) -> set[str]:
        """Extract type names referenced in a type definition's source code.

        Args:
            type_source: Source code of the type definition.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            Set of type names found in the definition.

        """
        # Parse the type definition and find type identifiers
        source_bytes = type_source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        type_names: set[str] = set()

        def walk_for_types(node):
            # Look for type_identifier nodes (user-defined types)
            if node.type == "type_identifier":
                type_name = source_bytes[node.start_byte : node.end_byte].decode("utf8")
                # Skip primitive types
                if type_name not in (
                    "number",
                    "string",
                    "boolean",
                    "void",
                    "null",
                    "undefined",
                    "any",
                    "never",
                    "unknown",
                    "object",
                    "symbol",
                    "bigint",
                ):
                    type_names.add(type_name)
            for child in node.children:
                walk_for_types(child)

        walk_for_types(tree.root_node)
        return type_names

    def _find_imported_type_definitions(
        self, type_names: set[str], imports: list[Any], module_root: Path, source_file_path: Path
    ) -> list[TypeDefinition]:
        """Find type definitions in imported files.

        Args:
            type_names: Set of type names to look for.
            imports: List of ImportInfo objects from the source file.
            module_root: Root directory of the module.
            source_file_path: Path to the source file (for resolving relative imports).

        Returns:
            List of TypeDefinition objects found in imported files.

        """
        found_definitions: list[TypeDefinition] = []

        # Build a map of type names to their import info and original names
        type_import_map: dict[str, tuple[Any, str]] = {}  # local_name -> (ImportInfo, original_name)
        for imp in imports:
            # Check if any of our type names are imported from this module
            for name, alias in imp.named_imports:
                # The type could be imported with an alias
                local_name = alias if alias else name
                if local_name in type_names:
                    type_import_map[local_name] = (imp, name)  # (ImportInfo, original_name)

        if not type_import_map:
            return found_definitions

        # Resolve imports and find type definitions
        from codeflash.languages.javascript.import_resolver import ImportResolver

        try:
            import_resolver = ImportResolver(module_root)
        except Exception:
            logger.debug("Failed to create ImportResolver for type definition lookup")
            return found_definitions

        for local_name, (import_info, original_name) in type_import_map.items():
            try:
                # Resolve the import to an actual file path
                resolved_import = import_resolver.resolve_import(import_info, source_file_path)
                if not resolved_import or not resolved_import.file_path.exists():
                    continue

                resolved_path = resolved_import.file_path

                # Read the source file and find type definitions
                try:
                    imported_source = resolved_path.read_text(encoding="utf-8")
                except Exception:
                    continue

                # Get analyzer for the imported file
                imported_analyzer = get_analyzer_for_file(resolved_path)
                type_defs = imported_analyzer.find_type_definitions(imported_source)

                # Find the type we're looking for
                for defn in type_defs:
                    if defn.name == original_name:
                        # Add file path info to the definition
                        defn.file_path = resolved_path
                        found_definitions.append(defn)
                        break

            except Exception as e:
                logger.debug("Failed to resolve type definition for %s: %s", local_name, e)
                continue

        return found_definitions

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.

        """
        try:
            source = function.file_path.read_text()
            analyzer = get_analyzer_for_file(function.file_path)
            imports = analyzer.find_imports(source)
            return self._find_helper_functions(function, source, analyzer, imports, project_root)
        except Exception as e:
            logger.warning("Failed to find helpers for %s: %s", function.function_name, e)
            return []

    def find_references(
        self, function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 500
    ) -> list[ReferenceInfo]:
        """Find all references (call sites) to a function across the codebase.

        Uses tree-sitter to find all places where a JavaScript/TypeScript function
        is called, including direct calls, callbacks, memoized versions, and re-exports.

        Args:
            function: The function to find references for.
            project_root: Root of the project to search.
            tests_root: Root of tests directory (references in tests are excluded).
            max_files: Maximum number of files to search.

        Returns:
            List of ReferenceInfo objects describing each reference location.

        """
        from codeflash.languages.base import ReferenceInfo
        from codeflash.languages.javascript.find_references import ReferenceFinder

        try:
            finder = ReferenceFinder(project_root)
            refs = finder.find_references(function, max_files=max_files)

            # Convert to ReferenceInfo and filter out tests
            result: list[ReferenceInfo] = []
            for ref in refs:
                # Exclude test files if tests_root is provided
                if tests_root:
                    try:
                        ref.file_path.relative_to(tests_root)
                        continue  # Skip if in tests_root
                    except ValueError:
                        pass  # Not in tests_root, include it

                result.append(
                    ReferenceInfo(
                        file_path=ref.file_path,
                        line=ref.line,
                        column=ref.column,
                        end_line=ref.end_line,
                        end_column=ref.end_column,
                        context=ref.context,
                        reference_type=ref.reference_type,
                        import_name=ref.import_name,
                        caller_function=ref.caller_function,
                    )
                )

            return result

        except Exception as e:
            logger.warning("Failed to find references for %s: %s", function.function_name, e)
            return []

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Uses node-based replacement to extract the method body from the optimized code
        and replace only the body in the original code, preserving the original signature.

        The new_source may be:
        1. A full class definition with the optimized method inside
        2. Just the method definition itself

        Args:
            source: Original source code.
            function: FunctionToOptimize identifying the function to replace.
            new_source: New source code containing the optimized function.

        Returns:
            Modified source code with function body replaced, or original source
            if new_source is empty or invalid.

        """
        if function.starting_line is None or function.ending_line is None:
            logger.error("Function %s has no line information", function.function_name)
            return source

        # If new_source is empty or whitespace-only, return original unchanged
        if not new_source or not new_source.strip():
            logger.warning("Empty new_source provided for %s, returning original", function.function_name)
            return source

        # Get analyzer for parsing
        if function.file_path:
            analyzer = get_analyzer_for_file(function.file_path)
        else:
            analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

        # Check if new_source contains a JSDoc comment - if so, use full replacement
        # to include the updated JSDoc along with the function body
        stripped_new_source = new_source.strip()
        if stripped_new_source.startswith("/**"):
            # new_source includes JSDoc, use full replacement to apply the new JSDoc
            if not self._contains_function_declaration(new_source, function.function_name, analyzer):
                logger.warning("new_source does not contain function %s, returning original", function.function_name)
                return source
            return self._replace_function_text_based(source, function, new_source, analyzer)

        # Extract just the method body from the new source
        new_body = self._extract_function_body(new_source, function.function_name, analyzer)
        if new_body is None:
            logger.warning(
                "Could not extract body for %s from optimized code, using full replacement", function.function_name
            )
            # Verify that new_source contains actual code before falling back to text replacement
            # This prevents deletion of the original function when new_source is invalid
            if not self._contains_function_declaration(new_source, function.function_name, analyzer):
                logger.warning("new_source does not contain function %s, returning original", function.function_name)
                return source
            return self._replace_function_text_based(source, function, new_source, analyzer)

        # Find the original function and replace its body
        return self._replace_function_body(source, function, new_body, analyzer)

    def _contains_function_declaration(self, source: str, function_name: str, analyzer: TreeSitterAnalyzer) -> bool:
        """Check if source contains a function declaration with the given name.

        Args:
            source: Source code to check.
            function_name: Name of the function to look for.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            True if the source contains the function declaration.

        """
        try:
            tree_functions = analyzer.find_functions(source, include_methods=True, include_arrow_functions=True)
            if any(func.name == function_name for func in tree_functions):
                return True

            # If not found, try wrapping in a dummy class (for standalone method definitions)
            wrapped_source = f"class __DummyClass__ {{\n{source}\n}}"
            tree_functions = analyzer.find_functions(wrapped_source, include_methods=True, include_arrow_functions=True)
            return any(func.name == function_name for func in tree_functions)
        except Exception:
            return False

    def _extract_function_body(self, source: str, function_name: str, analyzer: TreeSitterAnalyzer) -> str | None:
        """Extract the body of a function from source code.

        Searches for the function by name (handles both standalone functions and class methods)
        and extracts just the body content (between { and }).

        Args:
            source: Source code containing the function.
            function_name: Name of the function to find.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            The function body content (including braces), or None if not found.

        """
        # Try to find the function in the source as-is
        result = self._find_and_extract_body(source, function_name, analyzer)
        if result is not None:
            return result

        # If not found, the source might be just a method definition without class context
        # Try wrapping it in a dummy class to parse it correctly
        wrapped_source = f"class __DummyClass__ {{\n{source}\n}}"
        return self._find_and_extract_body(wrapped_source, function_name, analyzer)

    def _find_and_extract_body(self, source: str, function_name: str, analyzer: TreeSitterAnalyzer) -> str | None:
        """Internal helper to find a function and extract its body.

        Args:
            source: Source code containing the function.
            function_name: Name of the function to find.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            The function body content (including braces), or None if not found.

        """
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)

        def find_function_node(node, target_name: str):
            """Recursively find a function/method with the given name."""
            # Check method definitions
            if node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                    if name == target_name:
                        return node

            # Check function declarations
            if node.type in ("function_declaration", "function"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                    if name == target_name:
                        return node

            # Check arrow functions assigned to variables
            if node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        if name_node and value_node and value_node.type == "arrow_function":
                            name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                            if name == target_name:
                                return value_node

            # Recurse into children
            for child in node.children:
                result = find_function_node(child, target_name)
                if result:
                    return result

            return None

        func_node = find_function_node(tree.root_node, function_name)
        if not func_node:
            return None

        # Find the body node
        body_node = func_node.child_by_field_name("body")
        if not body_node:
            # For some node types, body might be a direct child
            for child in func_node.children:
                if child.type == "statement_block":
                    body_node = child
                    break

        if not body_node:
            return None

        # Extract the body text (including braces)
        return source_bytes[body_node.start_byte : body_node.end_byte].decode("utf8")

    def _replace_function_body(
        self, source: str, function: FunctionToOptimize, new_body: str, analyzer: TreeSitterAnalyzer
    ) -> str:
        """Replace the body of a function in source code with new body content.

        Preserves the original function signature and only replaces the body.

        Args:
            source: Original source code.
            function: FunctionToOptimize identifying the function to modify.
            new_body: New body content (including braces).
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            Modified source code with function body replaced.

        """
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)

        # Find the original function node
        def find_function_at_line(node, target_name: str, target_line: int):
            """Find a function with matching name and line number."""
            if node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                    # Line numbers in tree-sitter are 0-indexed
                    if name == target_name and (node.start_point[0] + 1) == target_line:
                        return node

            if node.type in (
                "function_declaration",
                "function",
                "generator_function_declaration",
                "generator_function",
            ):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                    if name == target_name and (node.start_point[0] + 1) == target_line:
                        return node

            if node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        if name_node and value_node and value_node.type == "arrow_function":
                            name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")
                            if name == target_name and (node.start_point[0] + 1) == target_line:
                                return value_node

            for child in node.children:
                result = find_function_at_line(child, target_name, target_line)
                if result:
                    return result

            return None

        func_node = find_function_at_line(tree.root_node, function.function_name, function.starting_line)
        if not func_node:
            logger.warning("Could not find function %s at line %s", function.function_name, function.starting_line)
            return source

        # Find the body node in the original
        body_node = func_node.child_by_field_name("body")
        if not body_node:
            for child in func_node.children:
                if child.type == "statement_block":
                    body_node = child
                    break

        if not body_node:
            logger.warning("Could not find body for function %s", function.function_name)
            return source

        # Get the indentation of the original body's opening brace
        lines = source.splitlines(keepends=True)
        body_start_line = body_node.start_point[0]  # 0-indexed
        if body_start_line < len(lines):
            # Find the position of the opening brace in the line
            original_line = lines[body_start_line]
            brace_col = body_node.start_point[1]
        else:
            brace_col = 0

        # Adjust indentation of the new body to match original
        new_body_lines = new_body.splitlines(keepends=True)
        if new_body_lines:
            # Get the indentation of the new body's first line (opening brace)
            first_line = new_body_lines[0]
            new_indent = len(first_line) - len(first_line.lstrip())

            # Calculate the indentation of content lines in original (typically brace_col + 4)
            # But for the brace itself, we use the column position
            original_body_text = source_bytes[body_node.start_byte : body_node.end_byte].decode("utf8")
            original_body_lines = original_body_text.splitlines(keepends=True)
            if len(original_body_lines) > 1:
                # Get indentation of the second line (first content line)
                content_line = original_body_lines[1]
                original_content_indent = len(content_line) - len(content_line.lstrip())
            else:
                original_content_indent = brace_col + 4  # Default to 4 spaces more than brace

            # Get indentation of new body's content lines
            if len(new_body_lines) > 1:
                new_content_line = new_body_lines[1]
                new_content_indent = len(new_content_line) - len(new_content_line.lstrip())
            else:
                new_content_indent = new_indent + 4

            indent_diff = original_content_indent - new_content_indent

            # Adjust indentation
            adjusted_lines = []
            for i, line in enumerate(new_body_lines):
                if i == 0:
                    # Opening brace - keep as is (will be placed at correct position by byte replacement)
                    adjusted_lines.append(line.lstrip())
                elif line.strip():
                    if indent_diff > 0:
                        adjusted_lines.append(" " * indent_diff + line)
                    elif indent_diff < 0:
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_lines.append(line[remove_amount:])
                    else:
                        adjusted_lines.append(line)
                else:
                    adjusted_lines.append(line)

            new_body = "".join(adjusted_lines)

        # Replace the body bytes
        before = source_bytes[: body_node.start_byte]
        after = source_bytes[body_node.end_byte :]

        result = before + new_body.encode("utf8") + after
        return result.decode("utf8")

    def _replace_function_text_based(
        self, source: str, function: FunctionToOptimize, new_source: str, analyzer: TreeSitterAnalyzer
    ) -> str:
        """Fallback text-based replacement when node-based replacement fails.

        Uses line numbers to replace the entire function.

        Args:
            source: Original source code.
            function: FunctionToOptimize identifying the function to replace.
            new_source: New function source code.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            Modified source code with function replaced.

        """
        lines = source.splitlines(keepends=True)

        # Handle case where source doesn't end with newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        tree_functions = analyzer.find_functions(source, include_methods=True, include_arrow_functions=True)
        target_func = None
        for func in tree_functions:
            if func.name == function.function_name and func.start_line == function.starting_line:
                target_func = func
                break

        # Use doc_start_line if available, otherwise fall back to start_line
        effective_start = (target_func.doc_start_line if target_func else None) or function.starting_line

        # Get indentation from original function's first line
        if function.starting_line <= len(lines):
            original_first_line = lines[function.starting_line - 1]
            original_indent = len(original_first_line) - len(original_first_line.lstrip())
        else:
            original_indent = 0

        # Skip JSDoc lines to find the actual function declaration in new source
        new_lines = new_source.splitlines(keepends=True)
        func_decl_line = new_lines[0] if new_lines else ""
        for line in new_lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("/**")
                and not stripped.startswith("*")
                and not stripped.startswith("//")
            ):
                func_decl_line = line
                break

        new_indent = len(func_decl_line) - len(func_decl_line.lstrip())
        indent_diff = original_indent - new_indent

        # Adjust indentation of new function if needed
        if indent_diff != 0:
            adjusted_new_lines = []
            for line in new_lines:
                if line.strip():
                    if indent_diff > 0:
                        adjusted_new_lines.append(" " * indent_diff + line)
                    else:
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_new_lines.append(line[remove_amount:])
                else:
                    adjusted_new_lines.append(line)
            new_lines = adjusted_new_lines

        # Ensure new function ends with newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Build result
        before = lines[: effective_start - 1]
        after = lines[function.ending_line :]

        result_lines = before + new_lines + after
        return "".join(result_lines)

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format JavaScript code using prettier (if available).

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.

        """
        try:
            # Try to use prettier via npx
            result = subprocess.run(
                ["npx", "prettier", "--stdin-filepath", "file.js"],
                check=False,
                input=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug("Prettier formatting failed: %s", e)

        return source

    # === Test Execution ===

    def run_tests(
        self, test_files: Sequence[Path], cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[list[TestResult], Path]:
        """Run Jest tests and return results.

        Args:
            test_files: Paths to test files to run.
            cwd: Working directory for test execution.
            env: Environment variables.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (list of TestResults, path to JUnit XML).

        """
        # Create output directory for results
        output_dir = cwd / ".codeflash"
        output_dir.mkdir(parents=True, exist_ok=True)
        junit_xml = output_dir / "jest-results.xml"

        # Build Jest command
        test_pattern = "|".join(str(f) for f in test_files)
        cmd = [
            "npx",
            "jest",
            "--reporters=default",
            "--reporters=jest-junit",
            f"--testPathPattern={test_pattern}",
            "--runInBand",  # Sequential for deterministic timing
            "--forceExit",
        ]

        test_env = env.copy()
        test_env["JEST_JUNIT_OUTPUT_FILE"] = str(junit_xml)

        try:
            result = subprocess.run(
                cmd, check=False, cwd=cwd, env=test_env, capture_output=True, text=True, timeout=timeout
            )

            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml

        except subprocess.TimeoutExpired:
            logger.warning("Test execution timed out after %ss", timeout)
            return [], junit_xml
        except Exception as e:
            logger.exception("Test execution failed: %s", e)
            return [], junit_xml

    def parse_test_results(self, junit_xml_path: Path, stdout: str) -> list[TestResult]:
        """Parse test results from JUnit XML.

        Args:
            junit_xml_path: Path to JUnit XML results file.
            stdout: Standard output from test execution.

        Returns:
            List of TestResult objects.

        """
        results: list[TestResult] = []

        if not junit_xml_path.exists():
            return results

        try:
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()

            for testcase in root.iter("testcase"):
                name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")
                time_str = testcase.get("time", "0")

                # Convert time to nanoseconds
                try:
                    runtime_ns = int(float(time_str) * 1_000_000_000)
                except ValueError:
                    runtime_ns = None

                # Check for failure/error
                failure = testcase.find("failure")
                error = testcase.find("error")
                passed = failure is None and error is None

                error_message = None
                if failure is not None:
                    error_message = failure.get("message", failure.text)
                elif error is not None:
                    error_message = error.get("message", error.text)

                # Determine test file from classname
                # Jest typically uses the file path as classname
                test_file = Path(classname) if classname else Path("unknown")

                results.append(
                    TestResult(
                        test_name=name,
                        test_file=test_file,
                        passed=passed,
                        runtime_ns=runtime_ns,
                        error_message=error_message,
                        stdout=stdout,
                    )
                )
        except Exception as e:
            logger.warning("Failed to parse JUnit XML: %s", e)

        return results

    # === Instrumentation ===

    def instrument_for_behavior(
        self, source: str, functions: Sequence[FunctionToOptimize], output_file: Path | None = None
    ) -> str:
        """Add behavior instrumentation to capture inputs/outputs.

        For JavaScript, this wraps functions to capture their arguments
        and return values.

        Args:
            source: Source code to instrument.
            functions: Functions to add tracing to.
            output_file: Optional output file for traces.

        Returns:
            Instrumented source code.

        """
        if not functions:
            return source

        from codeflash.languages.javascript.tracer import JavaScriptTracer

        # Use first function's file path if output_file not specified
        if output_file is None:
            file_path = functions[0].file_path
            output_file = file_path.parent / ".codeflash" / "traces.db"

        tracer = JavaScriptTracer(output_file)
        return tracer.instrument_source(source, functions[0].file_path, list(functions))

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionToOptimize) -> str:
        """Add timing instrumentation to test code.

        For JavaScript/Jest, we can use Jest's built-in timing or add custom timing.

        Args:
            test_source: Test source code to instrument.
            target_function: Function being benchmarked.

        Returns:
            Instrumented test source code.

        """
        # For benchmarking, we rely on Jest's built-in timing
        # which is captured in the JUnit XML output
        # No additional instrumentation needed
        return test_source

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """Check if JavaScript source code is syntactically valid.

        Uses tree-sitter to parse and check for errors.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.

        """
        try:
            analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)
            tree = analyzer.parse(source)
            # Check if tree has errors
            return not tree.root_node.has_error
        except Exception:
            return False

    def normalize_code(self, source: str) -> str:
        """Normalize JavaScript code for deduplication.

        Removes comments and normalizes whitespace.

        Args:
            source: Source code to normalize.

        Returns:
            Normalized source code.

        """
        # Simple normalization: remove extra whitespace
        # A full implementation would use tree-sitter to strip comments
        lines = source.splitlines()
        normalized_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                normalized_lines.append(stripped)
        return "\n".join(normalized_lines)

    # === Test Editing ===

    def add_runtime_comments(
        self, test_source: str, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
    ) -> str:
        """Add runtime performance comments to JavaScript test source.

        Args:
            test_source: Test source code to annotate.
            original_runtimes: Map of invocation IDs to original runtimes (ns).
            optimized_runtimes: Map of invocation IDs to optimized runtimes (ns).

        Returns:
            Test source code with runtime comments added.

        """
        from codeflash.languages.javascript.edit_tests import add_runtime_comments

        return add_runtime_comments(test_source, original_runtimes, optimized_runtimes)

    def remove_test_functions(self, test_source: str, functions_to_remove: list[str]) -> str:
        """Remove specific test functions from JavaScript test source.

        Args:
            test_source: Test source code.
            functions_to_remove: List of function names to remove.

        Returns:
            Test source code with specified functions removed.

        """
        from codeflash.languages.javascript.edit_tests import remove_test_functions

        return remove_test_functions(test_source, functions_to_remove)

    # === Test Result Comparison ===

    def compare_test_results(
        self, original_results_path: Path, candidate_results_path: Path, project_root: Path | None = None
    ) -> tuple[bool, list]:
        """Compare test results between original and candidate code.

        Args:
            original_results_path: Path to original test results SQLite DB.
            candidate_results_path: Path to candidate test results SQLite DB.
            project_root: Project root directory where node_modules is installed.

        Returns:
            Tuple of (are_equivalent, list of TestDiff objects).

        """
        from codeflash.languages.javascript.comparator import compare_test_results

        return compare_test_results(original_results_path, candidate_results_path, project_root=project_root)

    # === Configuration ===

    def get_test_file_suffix(self) -> str:
        """Get the test file suffix for JavaScript.

        Returns:
            Jest test file suffix.

        """
        return ".test.js"

    def get_comment_prefix(self) -> str:
        """Get the comment prefix for JavaScript.

        Returns:
            JavaScript single-line comment prefix.

        """
        return "//"

    def find_test_root(self, project_root: Path) -> Path | None:
        """Find the test root directory for a JavaScript project.

        Looks for common Jest test directory patterns.

        Args:
            project_root: Root directory of the project.

        Returns:
            Path to test root, or None if not found.

        """
        # Common test directory patterns for JavaScript/Jest
        test_dirs = [
            project_root / "tests",
            project_root / "test",
            project_root / "__tests__",
            project_root / "src" / "__tests__",
            project_root / "spec",
        ]

        for test_dir in test_dirs:
            if test_dir.exists() and test_dir.is_dir():
                return test_dir

        # Check for jest.config.js to find testMatch patterns
        jest_config = project_root / "jest.config.js"
        if jest_config.exists():
            # Default to project root if jest config exists
            return project_root

        # Check for test patterns in package.json
        package_json = project_root / "package.json"
        if package_json.exists():
            return project_root

        return None

    def get_module_path(self, source_file: Path, project_root: Path, tests_root: Path | None = None) -> str:
        """Get the module path for importing a JavaScript source file from tests.

        For JavaScript, this returns a relative path from the tests directory to the source file
        (e.g., '../fibonacci' for source at /project/fibonacci.js and tests at /project/tests/).

        Args:
            source_file: Path to the source file.
            project_root: Root of the project.
            tests_root: Root directory for tests (required for JS relative path calculation).

        Returns:
            Relative path string for importing the module from tests.

        """
        import os

        from codeflash.cli_cmds.console import logger

        if tests_root is None:
            tests_root = self.find_test_root(project_root) or project_root

        try:
            # Resolve both paths to absolute to ensure consistent relative path calculation
            source_file_abs = source_file.resolve().with_suffix("")
            tests_root_abs = tests_root.resolve()

            # Find the project root using language support
            project_root_from_lang = self.find_test_root(project_root)

            # Validate that tests_root is within the same project as the source file
            if project_root_from_lang:
                try:
                    tests_root_abs.relative_to(project_root_from_lang)
                except ValueError:
                    # tests_root is outside the project - use default
                    logger.warning(
                        f"Configured tests_root {tests_root_abs} is outside project {project_root_from_lang}. "
                        f"Using default: {project_root_from_lang / 'tests'}"
                    )
                    tests_root_abs = project_root_from_lang / "tests"
                    if not tests_root_abs.exists():
                        tests_root_abs = project_root_from_lang

            # Use os.path.relpath to compute relative path from tests_root to source file
            rel_path = os.path.relpath(str(source_file_abs), str(tests_root_abs))
            logger.debug(
                f"!lsp|Module path: source={source_file_abs}, tests_root={tests_root_abs}, rel_path={rel_path}"
            )
            return rel_path
        except ValueError:
            # Fallback if paths are on different drives (Windows)
            rel_path = source_file.relative_to(project_root)
            return "../" + rel_path.with_suffix("").as_posix()

    def verify_requirements(self, project_root: Path, test_framework: str = "jest") -> tuple[bool, list[str]]:
        """Verify that all JavaScript requirements are met.

        Checks for:
        1. Node.js installation
        2. npm availability
        3. Test framework (jest/vitest) installation (with monorepo support)

        Uses find_node_modules_with_package() from init_javascript to search up the
        directory tree for node_modules containing the test framework. This supports
        monorepo setups where dependencies are hoisted to the workspace root.

        Args:
            project_root: The project root directory.
            test_framework: The test framework to check for ("jest" or "vitest").

        Returns:
            Tuple of (success, list of error messages).

        """
        errors: list[str] = []

        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], check=False, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                errors.append("Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/")
        except FileNotFoundError:
            errors.append("Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/")
        except Exception as e:
            errors.append(f"Failed to check Node.js: {e}")

        # Check npm
        try:
            result = subprocess.run(["npm", "--version"], check=False, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                errors.append("npm is not available. Please ensure npm is installed with Node.js.")
        except FileNotFoundError:
            errors.append("npm is not available. Please ensure npm is installed with Node.js.")
        except Exception as e:
            errors.append(f"Failed to check npm: {e}")

        # Check test framework is installed (with monorepo support)
        # Uses find_node_modules_with_package which searches up the directory tree
        from codeflash.cli_cmds.init_javascript import find_node_modules_with_package

        node_modules = find_node_modules_with_package(project_root, test_framework)
        if node_modules:
            logger.debug("Found %s in node_modules at %s", test_framework, node_modules / test_framework)
        else:
            # Check if local node_modules exists at all
            local_node_modules = project_root / "node_modules"
            if not local_node_modules.exists():
                errors.append(
                    f"node_modules not found in {project_root}. Please run 'npm install' to install dependencies."
                )
            else:
                errors.append(
                    f"{test_framework} is not installed. "
                    f"Please run 'npm install --save-dev {test_framework}' to install it."
                )

        return len(errors) == 0, errors

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        """Ensure codeflash npm package is installed.

        Attempts to install the npm package for test instrumentation.

        Args:
            project_root: The project root directory.

        Returns:
            True if npm package is installed, False otherwise.

        """
        from codeflash.cli_cmds.console import logger

        node_modules_pkg = project_root / "node_modules" / "codeflash"
        if node_modules_pkg.exists():
            logger.debug("codeflash already installed")
            return True

        try:
            result = subprocess.run(
                ["npm", "install", "--save-dev", "codeflash"],
                check=False,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.debug("Installed codeflash from npm registry")
                return True
            logger.warning(f"Failed to install codeflash: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error installing codeflash: {e}")

        logger.error("Could not install codeflash. Please run: npm install --save-dev codeflash")
        return False

    def create_dependency_resolver(self, project_root: Path) -> None:
        return None

    def instrument_existing_test(
        self,
        test_path: Path,
        call_positions: Sequence[Any],
        function_to_optimize: Any,
        tests_project_root: Path,
        mode: str,
    ) -> tuple[bool, str | None]:
        """Inject profiling code into an existing JavaScript test file.

        Wraps function calls with codeflash.capture() or codeflash.capturePerf()
        for behavioral verification and performance benchmarking.

        Args:
            test_path: Path to the test file.
            call_positions: List of code positions where the function is called.
            function_to_optimize: The function being optimized.
            tests_project_root: Root directory of tests.
            mode: Testing mode - "behavior" or "performance".

        Returns:
            Tuple of (success, instrumented_code).

        """
        from codeflash.languages.javascript.instrument import inject_profiling_into_existing_js_test

        return inject_profiling_into_existing_js_test(
            test_path=test_path,
            call_positions=list(call_positions),
            function_to_optimize=function_to_optimize,
            tests_project_root=tests_project_root,
            mode=mode,
        )

    def instrument_source_for_line_profiler(
        # TODO: use the context to instrument helper files also
        self,
        func_info: FunctionToOptimize,
        line_profiler_output_file: Path,
    ) -> bool:
        from codeflash.languages.javascript.line_profiler import JavaScriptLineProfiler

        source_file_path = Path(func_info.file_path)

        current_source = source_file_path.read_text("utf-8")

        # Create line profiler and instrument source
        profiler = JavaScriptLineProfiler(output_file=line_profiler_output_file)
        try:
            instrumented_source = profiler.instrument_source(
                source=current_source, file_path=source_file_path, functions=[func_info]
            )

            # Write instrumented code to source file
            source_file_path.write_text(instrumented_source, encoding="utf-8")
            logger.debug("Wrote instrumented source to %s", source_file_path)
            return True
        except Exception as e:
            logger.warning("Failed to instrument source for line profiling: %s", e)
            return False

    def parse_line_profile_results(self, line_profiler_output_file: Path) -> dict:
        from codeflash.languages.javascript.line_profiler import JavaScriptLineProfiler

        if line_profiler_output_file.exists():
            parsed_results = JavaScriptLineProfiler.parse_results(line_profiler_output_file)
            if parsed_results.get("timings"):
                # Format output string for display
                str_out = self._format_js_line_profile_output(parsed_results)
                return {"timings": parsed_results.get("timings", {}), "unit": 1e-9, "str_out": str_out}
        logger.warning("No line profiler output file found at %s", line_profiler_output_file)
        return {"timings": {}, "unit": 0, "str_out": ""}

    def _format_js_line_profile_output(self, parsed_results: dict) -> str:
        """Format JavaScript line profiler results for display."""
        if not parsed_results.get("timings"):
            return ""

        lines = ["Line Profile Results:"]
        for file_path, line_data in parsed_results.get("timings", {}).items():
            lines.append(f"\nFile: {file_path}")
            lines.append("-" * 80)
            lines.append(f"{'Line':>6}  {'Hits':>8}  {'Time (ms)':>12}  {'% Time':>8}  {'Content'}")
            lines.append("-" * 80)

            total_time_ms = sum(data.get("time_ms", 0) for data in line_data.values())
            for line_num, data in sorted(line_data.items()):
                hits = data.get("hits", 0)
                time_ms = data.get("time_ms", 0)
                pct = (time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
                content = data.get("content", "")
                # Truncate long lines for display
                if len(content) > 50:
                    content = content[:47] + "..."
                lines.append(f"{line_num:>6}  {hits:>8}  {time_ms:>12.3f}  {pct:>7.1f}%  {content}")

        return "\n".join(lines)

    # === Test Execution ===

    def run_behavioral_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        enable_coverage: bool = False,
        candidate_index: int = 0,
        test_framework: str | None = None,
    ) -> tuple[Path, Any, Path | None, Path | None]:
        """Run behavioral tests using the detected test framework.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            enable_coverage: Whether to collect coverage information.
            candidate_index: Index of the candidate being tested.
            test_framework: Test framework to use ("jest" or "vitest"). If None, uses singleton.

        Returns:
            Tuple of (result_file_path, subprocess_result, coverage_path, config_path).

        """
        from codeflash.languages.test_framework import get_js_test_framework_or_default

        framework = test_framework or get_js_test_framework_or_default()

        if framework == "vitest":
            from codeflash.languages.javascript.vitest_runner import run_vitest_behavioral_tests

            return run_vitest_behavioral_tests(
                test_paths=test_paths,
                test_env=test_env,
                cwd=cwd,
                timeout=timeout,
                project_root=project_root,
                enable_coverage=enable_coverage,
                candidate_index=candidate_index,
            )

        from codeflash.languages.javascript.test_runner import run_jest_behavioral_tests

        return run_jest_behavioral_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=timeout,
            project_root=project_root,
            enable_coverage=enable_coverage,
            candidate_index=candidate_index,
        )

    # JavaScript/TypeScript benchmarking uses high max_loops like Python (100,000)
    # The actual loop count is limited by target_duration_seconds, not max_loops
    JS_BENCHMARKING_MAX_LOOPS = 100_000

    def run_benchmarking_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        min_loops: int = 5,
        max_loops: int = 100_000,
        target_duration_seconds: float = 10.0,
        test_framework: str | None = None,
    ) -> tuple[Path, Any]:
        """Run benchmarking tests using the detected test framework.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            min_loops: Minimum number of loops for benchmarking.
            max_loops: Maximum number of loops for benchmarking.
            target_duration_seconds: Target duration for benchmarking in seconds.
            test_framework: Test framework to use ("jest" or "vitest"). If None, uses singleton.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
        from codeflash.languages.test_framework import get_js_test_framework_or_default

        framework = test_framework or get_js_test_framework_or_default()
        logger.debug("run_benchmarking_tests called with framework=%s", framework)

        # Use JS-specific high max_loops - actual loop count is limited by target_duration
        effective_max_loops = self.JS_BENCHMARKING_MAX_LOOPS

        if framework == "vitest":
            from codeflash.languages.javascript.vitest_runner import run_vitest_benchmarking_tests

            logger.debug("Dispatching to run_vitest_benchmarking_tests")
            return run_vitest_benchmarking_tests(
                test_paths=test_paths,
                test_env=test_env,
                cwd=cwd,
                timeout=timeout,
                project_root=project_root,
                min_loops=min_loops,
                max_loops=effective_max_loops,
                target_duration_ms=int(target_duration_seconds * 1000),
            )

        from codeflash.languages.javascript.test_runner import run_jest_benchmarking_tests

        return run_jest_benchmarking_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=timeout,
            project_root=project_root,
            min_loops=min_loops,
            max_loops=effective_max_loops,
            target_duration_ms=int(target_duration_seconds * 1000),
        )

    def run_line_profile_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        line_profile_output_file: Path | None = None,
        test_framework: str | None = None,
    ) -> tuple[Path, Any]:
        """Run tests for line profiling using the detected test framework.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            line_profile_output_file: Path where line profile results will be written.
            test_framework: Test framework to use ("jest" or "vitest"). If None, uses singleton.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
        from codeflash.languages.test_framework import get_js_test_framework_or_default

        framework = test_framework or get_js_test_framework_or_default()

        if framework == "vitest":
            from codeflash.languages.javascript.vitest_runner import run_vitest_line_profile_tests

            return run_vitest_line_profile_tests(
                test_paths=test_paths,
                test_env=test_env,
                cwd=cwd,
                timeout=timeout,
                project_root=project_root,
                line_profile_output_file=line_profile_output_file,
            )

        from codeflash.languages.javascript.test_runner import run_jest_line_profile_tests

        return run_jest_line_profile_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=timeout,
            project_root=project_root,
            line_profile_output_file=line_profile_output_file,
        )


@register_language
class TypeScriptSupport(JavaScriptSupport):
    """TypeScript language support implementation.

    This class extends JavaScriptSupport to provide TypeScript-specific
    language identification while sharing all JavaScript functionality.
    TypeScript and JavaScript use the same parser, test framework (Jest),
    and code instrumentation approach.
    """

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.TYPESCRIPT

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions for TypeScript files."""
        return (".ts", ".tsx", ".mts")

    def _get_test_patterns(self) -> list[str]:
        """Get test file patterns for TypeScript.

        Includes TypeScript patterns plus JavaScript patterns for mixed projects.

        Returns:
            List of glob patterns for test files.

        """
        return [
            "*.test.ts",
            "*.test.tsx",
            "*.spec.ts",
            "*.spec.tsx",
            "__tests__/**/*.ts",
            "__tests__/**/*.tsx",
            *super()._get_test_patterns(),
        ]

    def get_test_file_suffix(self) -> str:
        """Get the test file suffix for TypeScript.

        Returns:
            Jest test file suffix for TypeScript.

        """
        return ".test.ts"

    def validate_syntax(self, source: str) -> bool:
        """Check if TypeScript source code is syntactically valid.

        Uses tree-sitter TypeScript parser to parse and check for errors.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.

        """
        try:
            analyzer = TreeSitterAnalyzer(TreeSitterLanguage.TYPESCRIPT)
            tree = analyzer.parse(source)
            return not tree.root_node.has_error
        except Exception:
            return False

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format TypeScript code using prettier (if available).

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.

        """
        try:
            # Determine file extension for prettier
            stdin_filepath = str(file_path.name) if file_path else "file.ts"

            # Try to use prettier via npx
            result = subprocess.run(
                ["npx", "prettier", "--stdin-filepath", stdin_filepath],
                check=False,
                input=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug("Prettier formatting failed: %s", e)

        return source
