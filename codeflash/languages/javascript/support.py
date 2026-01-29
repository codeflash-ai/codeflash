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

from codeflash.languages.base import (
    CodeContext,
    FunctionFilterCriteria,
    FunctionInfo,
    HelperFunction,
    Language,
    ParentInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.registry import register_language
from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage, get_analyzer_for_file

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    def test_framework(self) -> str:
        """Primary test framework for JavaScript."""
        return "jest"

    @property
    def comment_prefix(self) -> str:
        return "//"

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionInfo]:
        """Find all optimizable functions in a JavaScript file.

        Uses tree-sitter to parse the file and find functions.

        Args:
            file_path: Path to the JavaScript file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionInfo objects for discovered functions.

        """
        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        try:
            analyzer = get_analyzer_for_file(file_path)
            tree_functions = analyzer.find_functions(
                source, include_methods=criteria.include_methods, include_arrow_functions=True, require_name=True
            )

            functions: list[FunctionInfo] = []
            for func in tree_functions:
                # Check for return statement if required
                if criteria.require_return and not analyzer.has_return_statement(func, source):
                    continue

                # Check async filter
                if not criteria.include_async and func.is_async:
                    continue

                # Build parents list
                parents: list[ParentInfo] = []
                if func.class_name:
                    parents.append(ParentInfo(name=func.class_name, type="ClassDef"))
                if func.parent_function:
                    parents.append(ParentInfo(name=func.parent_function, type="FunctionDef"))

                functions.append(
                    FunctionInfo(
                        name=func.name,
                        file_path=file_path,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        start_col=func.start_col,
                        end_col=func.end_col,
                        parents=tuple(parents),
                        is_async=func.is_async,
                        is_method=func.is_method,
                        language=self.language,
                        doc_start_line=func.doc_start_line,
                    )
                )

            return functions

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return []

    def discover_functions_from_source(self, source: str, file_path: Path | None = None) -> list[FunctionInfo]:
        """Find all functions in source code string.

        Uses tree-sitter to parse the source and find functions.

        Args:
            source: The source code to analyze.
            file_path: Optional file path for context (used for language detection).

        Returns:
            List of FunctionInfo objects for discovered functions.

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

            functions: list[FunctionInfo] = []
            for func in tree_functions:
                # Build parents list
                parents: list[ParentInfo] = []
                if func.class_name:
                    parents.append(ParentInfo(name=func.class_name, type="ClassDef"))
                if func.parent_function:
                    parents.append(ParentInfo(name=func.parent_function, type="FunctionDef"))

                functions.append(
                    FunctionInfo(
                        name=func.name,
                        file_path=file_path or Path("unknown"),
                        start_line=func.start_line,
                        end_line=func.end_line,
                        start_col=func.start_col,
                        end_col=func.end_col,
                        parents=tuple(parents),
                        is_async=func.is_async,
                        is_method=func.is_method,
                        language=self.language,
                        doc_start_line=func.doc_start_line,
                    )
                )

            return functions

        except Exception as e:
            logger.warning(f"Failed to parse source: {e}")
            return []

    def _get_test_patterns(self) -> list[str]:
        """Get test file patterns for this language.

        Override in subclasses to provide language-specific patterns.

        Returns:
            List of glob patterns for test files.

        """
        return ["*.test.js", "*.test.jsx", "*.spec.js", "*.spec.jsx", "__tests__/**/*.js", "__tests__/**/*.jsx"]

    def discover_tests(self, test_root: Path, source_functions: Sequence[FunctionInfo]) -> dict[str, list[TestInfo]]:
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
                    if func.name in imported_names or func.name in source:
                        if func.qualified_name not in result:
                            result[func.qualified_name] = []
                        for test_name in test_functions:
                            result[func.qualified_name].append(
                                TestInfo(test_name=test_name, test_file=test_file, test_class=None)
                            )
            except Exception as e:
                logger.debug(f"Failed to analyze test file {test_file}: {e}")

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

    def extract_code_context(self, function: FunctionInfo, project_root: Path, module_root: Path) -> CodeContext:
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
            logger.error(f"Failed to read {function.file_path}: {e}")
            return CodeContext(target_code="", target_file=function.file_path, language=Language.JAVASCRIPT)

        # Find imports and helper functions
        analyzer = get_analyzer_for_file(function.file_path)

        # Find the FunctionNode to get doc_start_line for JSDoc inclusion
        tree_functions = analyzer.find_functions(source, include_methods=True, include_arrow_functions=True)
        target_func = None
        for func in tree_functions:
            if func.name == function.name and func.start_line == function.start_line:
                target_func = func
                break

        # Extract the function source, including JSDoc if present
        lines = source.splitlines(keepends=True)
        if function.start_line and function.end_line:
            # Use doc_start_line if available, otherwise fall back to start_line
            effective_start = (target_func.doc_start_line if target_func else None) or function.start_line
            target_lines = lines[effective_start - 1 : function.end_line]
            target_code = "".join(target_lines)
        else:
            target_code = ""

        # For class methods, wrap the method in its class definition
        # This is necessary because method definition syntax is only valid inside a class body
        if function.is_method and function.parents:
            class_name = None
            for parent in function.parents:
                if parent.type == "ClassDef":
                    class_name = parent.name
                    break

            if class_name:
                # Find the class definition in the source to get proper indentation and any class JSDoc
                class_info = self._find_class_definition(source, class_name, analyzer)
                if class_info:
                    class_jsdoc, class_indent = class_info
                    # Wrap the method in a minimal class definition
                    if class_jsdoc:
                        target_code = f"{class_jsdoc}\n{class_indent}class {class_name} {{\n{target_code}{class_indent}}}\n"
                    else:
                        target_code = f"{class_indent}class {class_name} {{\n{target_code}{class_indent}}}\n"
                else:
                    # Fallback: wrap with no indentation
                    target_code = f"class {class_name} {{\n{target_code}}}\n"

        imports = analyzer.find_imports(source)

        # Find helper functions called by target
        helpers = self._find_helper_functions(function, source, analyzer, imports, module_root)

        # Extract import statements as strings
        import_lines = []
        for imp in imports:
            imp_lines = lines[imp.start_line - 1 : imp.end_line]
            import_lines.append("".join(imp_lines).strip())

        # Find module-level declarations (global variables/constants) referenced by the function
        read_only_context = self._find_referenced_globals(
            target_code=target_code, helpers=helpers, source=source, analyzer=analyzer, imports=imports
        )

        # Validate that the extracted code is syntactically valid
        # If not, raise an error to fail the optimization early
        if target_code and not self.validate_syntax(target_code):
            error_msg = (
                f"Extracted code for {function.name} is not syntactically valid JavaScript. "
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
        self, source: str, class_name: str, analyzer: TreeSitterAnalyzer
    ) -> tuple[str, str] | None:
        """Find a class definition and extract its JSDoc comment and indentation.

        Args:
            source: The source code to search.
            class_name: The name of the class to find.
            analyzer: TreeSitterAnalyzer for parsing.

        Returns:
            Tuple of (jsdoc_comment, indentation) or None if not found.

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

        return (jsdoc, indentation)

    def _find_helper_functions(
        self, function: FunctionInfo, source: str, analyzer: TreeSitterAnalyzer, imports: list[Any], module_root: Path
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
            if func.name == function.name and func.start_line == function.start_line:
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
            if func.name in calls_set and func.name != function.name:
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
            logger.debug(f"Failed to find cross-file helpers: {e}")

        return helpers

    def _find_referenced_globals(
        self,
        target_code: str,
        helpers: list[HelperFunction],
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[Any],
    ) -> str:
        """Find module-level declarations referenced by the target function and its helpers.

        Args:
            target_code: The target function's source code.
            helpers: List of helper functions.
            source: Full source code of the file.
            analyzer: TreeSitterAnalyzer for parsing.
            imports: List of ImportInfo objects.

        Returns:
            String containing all referenced global declarations.

        """
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
            # Also skip if it's an import
            if decl.name not in imported_names:
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

    def find_helper_functions(self, function: FunctionInfo, project_root: Path) -> list[HelperFunction]:
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
            logger.warning(f"Failed to find helpers for {function.name}: {e}")
            return []

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionInfo, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Uses text-based replacement with line numbers. Includes JSDoc comments
        in the replacement region if present.

        Args:
            source: Original source code.
            function: FunctionInfo identifying the function to replace.
            new_source: New function source code.

        Returns:
            Modified source code with function replaced.

        """
        if function.start_line is None or function.end_line is None:
            logger.error(f"Function {function.name} has no line information")
            return source

        lines = source.splitlines(keepends=True)

        # Handle case where source doesn't end with newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        # Find the FunctionNode to get doc_start_line for JSDoc inclusion
        if function.file_path:
            analyzer = get_analyzer_for_file(function.file_path)
        else:
            analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

        tree_functions = analyzer.find_functions(source, include_methods=True, include_arrow_functions=True)
        target_func = None
        for func in tree_functions:
            if func.name == function.name and func.start_line == function.start_line:
                target_func = func
                break

        # Use doc_start_line if available, otherwise fall back to start_line
        effective_start = (target_func.doc_start_line if target_func else None) or function.start_line

        # Get indentation from original function's first line (declaration, not JSDoc)
        if function.start_line <= len(lines):
            original_first_line = lines[function.start_line - 1]
            original_indent = len(original_first_line) - len(original_first_line.lstrip())
        else:
            original_indent = 0

        # Get indentation from new function's first line
        # Skip JSDoc lines to find the actual function declaration
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

        # Calculate indent adjustment needed
        indent_diff = original_indent - new_indent

        # Adjust indentation of new function if needed
        if indent_diff != 0:
            adjusted_new_lines = []
            for line in new_lines:
                if line.strip():  # Non-empty line
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

        # Build result using effective_start (includes JSDoc)
        before = lines[: effective_start - 1]
        after = lines[function.end_line :]

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
            logger.debug(f"Prettier formatting failed: {e}")

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
            logger.warning(f"Test execution timed out after {timeout}s")
            return [], junit_xml
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
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
            logger.warning(f"Failed to parse JUnit XML: {e}")

        return results

    # === Instrumentation ===

    def instrument_for_behavior(
        self, source: str, functions: Sequence[FunctionInfo], output_file: Path | None = None
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

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionInfo) -> str:
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

    def get_module_path(
        self, source_file: Path, project_root: Path, tests_root: Path | None = None
    ) -> str:
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

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        """Ensure codeflash npm package is installed.

        Attempts to install the npm package for test instrumentation.
        Falls back to copying files if npm install fails.

        Args:
            project_root: The project root directory.

        Returns:
            True if npm package is installed, False if falling back to file copy.

        """
        import subprocess

        from codeflash.cli_cmds.console import logger

        # Check if package is already installed
        node_modules_pkg = project_root / "node_modules" / "codeflash"
        if node_modules_pkg.exists():
            logger.debug("codeflash already installed")
            return True

        # Try to install from local package first (for development)
        local_package_path = Path(__file__).parent.parent.parent.parent / "packages" / "cli"
        if local_package_path.exists():
            try:
                result = subprocess.run(
                    ["npm", "install", "--save-dev", str(local_package_path)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    logger.debug("Installed codeflash from local package")
                    return True
                logger.warning(f"Failed to install local package: {result.stderr}")
            except Exception as e:
                logger.warning(f"Error installing local package: {e}")

        # Could try npm registry here in the future:
        # subprocess.run(["npm", "install", "--save-dev", "codeflash"], ...)

        return False

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
        func_info: FunctionInfo,
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
            logger.debug(f"Wrote instrumented source to {source_file_path}")  # noqa: G004
            return True  # noqa: TRY300
        except Exception as e:
            logger.warning(f"Failed to instrument source for line profiling: {e}")  # noqa: G004
            return False

    def parse_line_profile_results(self, line_profiler_output_file: Path) -> dict:
        from codeflash.languages.javascript.line_profiler import JavaScriptLineProfiler

        if line_profiler_output_file.exists():
            parsed_results = JavaScriptLineProfiler.parse_results(line_profiler_output_file)
            if parsed_results.get("timings"):
                # Format output string for display
                str_out = self._format_js_line_profile_output(parsed_results)
                return {"timings": parsed_results.get("timings", {}), "unit": 1e-9, "str_out": str_out}
        logger.warning(f"No line profiler output file found at {line_profiler_output_file}")  # noqa: G004
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
    ) -> tuple[Path, Any, Path | None, Path | None]:
        """Run Jest behavioral tests.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            enable_coverage: Whether to collect coverage information.
            candidate_index: Index of the candidate being tested.

        Returns:
            Tuple of (result_file_path, subprocess_result, coverage_path, config_path).

        """
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
    ) -> tuple[Path, Any]:
        """Run Jest benchmarking tests.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            min_loops: Minimum number of loops for benchmarking.
            max_loops: Maximum number of loops for benchmarking.
            target_duration_seconds: Target duration for benchmarking in seconds.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
        from codeflash.languages.javascript.test_runner import run_jest_benchmarking_tests

        return run_jest_benchmarking_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=timeout,
            project_root=project_root,
            min_loops=min_loops,
            max_loops=max_loops,
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
    ) -> tuple[Path, Any]:
        """Run Jest tests for line profiling.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            line_profile_output_file: Path where line profile results will be written.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
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
        ] + super()._get_test_patterns()

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
            if file_path:
                stdin_filepath = str(file_path.name)
            else:
                stdin_filepath = "file.ts"

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
            logger.debug(f"Prettier formatting failed: {e}")

        return source
