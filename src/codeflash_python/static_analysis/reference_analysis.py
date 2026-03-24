from __future__ import annotations

import ast
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jedi

from codeflash_python.code_utils.config_consts import MAX_CONTEXT_LEN_REVIEW

if TYPE_CHECKING:
    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


@dataclass
class FunctionCallLocation:
    """Represents a location where the target function is called."""

    calling_function: str
    line: int
    column: int


@dataclass
class FunctionDefinitionInfo:
    """Contains information about a function definition."""

    name: str
    node: ast.FunctionDef
    source_code: str
    start_line: int
    end_line: int
    is_method: bool
    class_name: str | None = None


class FunctionCallFinder(ast.NodeVisitor):
    """AST visitor that finds all function definitions that call a specific qualified function.

    Args:
        target_function_name: The qualified name of the function to find (e.g., "module.function" or "function")
        target_filepath: The filepath where the target function is defined

    """

    def __init__(self, target_function_name: str, target_filepath: str, source_lines: list[str]) -> None:
        self.target_function_name = target_function_name
        self.target_filepath = target_filepath
        self.source_lines = source_lines  # Store original source lines for extraction

        # Parse the target function name into parts
        self.target_parts = target_function_name.split(".")
        self.target_base_name = self.target_parts[-1]

        # Track current context
        self.current_function_stack: list[tuple[str, ast.FunctionDef]] = []
        self.current_class_stack: list[str] = []

        # Track imports to resolve qualified names
        self.imports: dict[str, str] = {}  # Maps imported names to their full paths

        # Results
        self.function_calls: list[FunctionCallLocation] = []
        self.calling_functions: set[str] = set()
        self.function_definitions: dict[str, FunctionDefinitionInfo] = {}

        # Track if we found calls in the current function
        self.found_call_in_current_function = False
        self.functions_with_nested_calls: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Track regular imports."""
        for alias in node.names:
            if alias.asname:
                # import module as alias
                self.imports[alias.asname] = alias.name
            else:
                # import module
                self.imports[alias.name.split(".")[-1]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            for alias in node.names:
                if alias.name == "*":
                    # from module import *
                    self.imports["*"] = node.module
                elif alias.asname:
                    # from module import name as alias
                    self.imports[alias.asname] = f"{node.module}.{alias.name}"
                else:
                    # from module import name
                    self.imports[alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track when entering a class definition."""
        self.current_class_stack.append(node.name)
        self.generic_visit(node)
        self.current_class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track when entering a function definition."""
        self.visit_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track when entering an async function definition."""
        self.visit_function_def(node)  # type: ignore[arg-type]

    def visit_function_def(self, node: ast.FunctionDef) -> None:
        """Track when entering a function definition."""
        func_name = node.name

        # Build the full qualified name including class if applicable
        full_name = f"{'.'.join(self.current_class_stack)}.{func_name}" if self.current_class_stack else func_name

        self.current_function_stack.append((full_name, node))
        self.found_call_in_current_function = False

        # Visit the function body
        self.generic_visit(node)

        # Process the function after visiting its body
        if self.found_call_in_current_function and full_name not in self.function_definitions:
            # Extract function source code
            source_code = self.extract_source_code(node)

            self.function_definitions[full_name] = FunctionDefinitionInfo(
                name=full_name,
                node=node,
                source_code=source_code,
                start_line=node.lineno,
                end_line=node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
                is_method=bool(self.current_class_stack),
                class_name=self.current_class_stack[-1] if self.current_class_stack else None,
            )

        # Handle nested functions - mark parent as containing nested calls
        if self.found_call_in_current_function and len(self.current_function_stack) > 1:
            parent_name = self.current_function_stack[-2][0]
            self.functions_with_nested_calls.add(parent_name)

            # Also store the parent function if not already stored
            if parent_name not in self.function_definitions:
                parent_node = self.current_function_stack[-2][1]
                parent_source = self.extract_source_code(parent_node)

                # Check if parent is a method (excluding current level)
                parent_class_context = self.current_class_stack if len(self.current_function_stack) == 2 else []

                self.function_definitions[parent_name] = FunctionDefinitionInfo(
                    name=parent_name,
                    node=parent_node,
                    source_code=parent_source,
                    start_line=parent_node.lineno,
                    end_line=parent_node.end_lineno if hasattr(parent_node, "end_lineno") else parent_node.lineno,
                    is_method=bool(parent_class_context),
                    class_name=parent_class_context[-1] if parent_class_context else None,
                )

        self.current_function_stack.pop()

        # Reset flag for parent function
        if self.current_function_stack:
            parent_name = self.current_function_stack[-1][0]
            self.found_call_in_current_function = parent_name in self.calling_functions

    def visit_Call(self, node: ast.Call) -> None:
        """Check if this call matches our target function."""
        if not self.current_function_stack:
            # Not inside a function, skip
            self.generic_visit(node)
            return

        if self.is_target_function_call(node):
            current_func_name = self.current_function_stack[-1][0]

            call_location = FunctionCallLocation(
                calling_function=current_func_name, line=node.lineno, column=node.col_offset
            )

            self.function_calls.append(call_location)
            self.calling_functions.add(current_func_name)
            self.found_call_in_current_function = True

        self.generic_visit(node)

    def is_target_function_call(self, node: ast.Call) -> bool:
        """Determine if this call node is calling our target function."""
        call_name = self.get_call_name(node.func)
        if not call_name:
            return False

        # Check if it matches directly
        if call_name == self.target_function_name:
            return True

        # Check if it's just the base name matching
        if call_name == self.target_base_name:
            # Could be imported with a different name, check imports
            if call_name in self.imports:
                imported_path = self.imports[call_name]
                if imported_path == self.target_function_name or imported_path.endswith(
                    f".{self.target_function_name}"
                ):
                    return True
            # Could also be a direct call if we're in the same file
            return True

        # Check for qualified calls with imports
        call_parts = call_name.split(".")
        if call_parts[0] in self.imports:
            # Resolve the full path using imports
            base_import = self.imports[call_parts[0]]
            full_path = f"{base_import}.{'.'.join(call_parts[1:])}" if len(call_parts) > 1 else base_import

            if full_path == self.target_function_name or full_path.endswith(f".{self.target_function_name}"):
                return True

        return False

    def get_call_name(self, func_node) -> str | None:
        """Extract the name being called from a function node."""
        # Fast path short-circuit for ast.Name nodes
        if isinstance(func_node, ast.Name):
            return func_node.id

        # Fast attribute chain extraction (speed: append, loop, join, NO reversed)
        if isinstance(func_node, ast.Attribute):
            parts = []
            current = func_node
            # Unwind attribute chain as tight as possible (checked at each loop iteration)
            while True:
                parts.append(current.attr)
                val = current.value
                if isinstance(val, ast.Attribute):
                    current = val
                    continue
                if isinstance(val, ast.Name):
                    parts.append(val.id)
                    # Join in-place backwards via slice instead of reversed for slight speedup
                    return ".".join(parts[::-1])
                break
        return None

    def extract_source_code(self, node: ast.FunctionDef) -> str:
        """Extract source code for a function node using original source lines."""
        if not self.source_lines or not hasattr(node, "lineno"):
            # Fallback to ast.unparse if available (Python 3.9+)
            try:
                return ast.unparse(node)
            except AttributeError:
                return f"# Source code extraction not available for {node.name}"

        # Get the lines for this function
        start_line = node.lineno - 1  # Convert to 0-based index
        end_line = node.end_lineno if hasattr(node, "end_lineno") else len(self.source_lines)

        # Extract the function lines
        func_lines = self.source_lines[start_line:end_line]

        # Find the minimum indentation (excluding empty lines)
        min_indent = float("inf")
        for line in func_lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # If this is a method (inside a class), preserve one level of indentation
        if self.current_class_stack:
            # Keep 4 spaces of indentation for methods
            dedent_amount = max(0, min_indent - 4)
            result_lines = []
            for line in func_lines:
                if line.strip():  # Only dedent non-empty lines
                    result_lines.append(line[dedent_amount:] if len(line) > dedent_amount else line)
                else:
                    result_lines.append(line)
        else:
            # For top-level functions, remove all leading indentation
            result_lines = []
            for line in func_lines:
                if line.strip():  # Only dedent non-empty lines
                    result_lines.append(line[min_indent:] if len(line) > min_indent else line)
                else:
                    result_lines.append(line)

        return "".join(result_lines).rstrip()

    def get_results(self) -> dict[str, str]:
        """Get the results of the analysis.

        Returns:
            A dictionary mapping qualified function names to their source code definitions.

        """
        return {info.name: info.source_code for info in self.function_definitions.values()}


def find_function_calls(source_code: str, target_function_name: str, target_filepath: str) -> dict[str, str]:
    """Find all function definitions that call a specific target function.

    Args:
        source_code: The Python source code to analyze
        target_function_name: The qualified name of the function to find (e.g., "module.function")
        target_filepath: The filepath where the target function is defined

    Returns:
        A dictionary mapping qualified function names to their source code definitions.
        Example: {"function_a": "def function_a():    ...", "MyClass.method_one": "def method_one(self):    ..."}

    """
    # Parse the source code
    tree = ast.parse(source_code)

    # Split source into lines for source extraction
    source_lines = source_code.splitlines(keepends=True)

    # Create and run the visitor
    visitor = FunctionCallFinder(target_function_name, target_filepath, source_lines)
    visitor.visit(tree)

    return visitor.get_results()


def find_references(
    function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 500
) -> list:
    """Find all references (call sites) to a function across the codebase."""
    from codeflash_python.context.types import ReferenceInfo

    try:
        source = function.file_path.read_text()
        script = jedi.Script(code=source, path=function.file_path)
        names = script.get_names(all_scopes=True, definitions=True)

        function_pos = None
        for name in names:
            if name.type == "function" and name.name == function.function_name:
                if function.class_name:
                    parent = name.parent()
                    if parent and parent.name == function.class_name and parent.type == "class":
                        function_pos = (name.line, name.column)
                        break
                else:
                    function_pos = (name.line, name.column)
                    break

        if function_pos is None:
            return []

        script = jedi.Script(code=source, path=function.file_path, project=jedi.Project(path=project_root))
        references = script.get_references(line=function_pos[0], column=function_pos[1])

        result: list[ReferenceInfo] = []
        seen_locations: set[tuple[Path, int, int]] = set()

        for ref in references:
            if not ref.module_path:
                continue
            ref_path = Path(ref.module_path)
            if ref_path == function.file_path and ref.line == function_pos[0]:
                continue
            if tests_root:
                try:
                    ref_path.relative_to(tests_root)
                    continue
                except ValueError:
                    pass
            loc_key = (ref_path, ref.line, ref.column)
            if loc_key in seen_locations:
                continue
            seen_locations.add(loc_key)
            try:
                ref_source = ref_path.read_text()
                lines = ref_source.splitlines()
                context = lines[ref.line - 1] if ref.line <= len(lines) else ""
            except Exception:
                context = ""
            caller_function = None
            try:
                parent = ref.parent()
                if parent and parent.type == "function":
                    caller_function = parent.name
            except Exception:
                pass
            result.append(
                ReferenceInfo(
                    file_path=ref_path,
                    line=ref.line,
                    column=ref.column,
                    end_line=ref.line,
                    end_column=ref.column + len(function.function_name),
                    context=context.strip(),
                    reference_type="call",
                    import_name=function.function_name,
                    caller_function=caller_function,
                )
            )
        return result
    except Exception as e:
        logger.warning("Failed to find references for %s: %s", function.function_name, e)
        return []


def extract_calling_function_source(source_code: str, function_name: str, ref_line: int) -> str | None:
    """Extract the source code of a calling function in Python."""
    try:
        import ast as _ast

        lines = source_code.splitlines()
        tree = _ast.parse(source_code)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == function_name:
                end_line = node.end_lineno or node.lineno
                if node.lineno <= ref_line <= end_line:
                    return "\n".join(lines[node.lineno - 1 : end_line])
    except Exception:
        return None
    return None


def get_opt_review_metrics(
    source_code: str, file_path: Path, qualified_name: str, project_root: Path, tests_root: Path, language: str
) -> str:
    """Get function reference metrics for optimization review.

    Uses static analysis to find references in Python code.

    Args:
        source_code: Source code of the file containing the function.
        file_path: Path to the file.
        qualified_name: Qualified name of the function (e.g., "module.ClassName.method").
        project_root: Root of the project.
        tests_root: Root of the tests directory.
        language: The programming language.

    Returns:
        Markdown-formatted string with code blocks showing calling functions.

    """
    from codeflash_core.models import FunctionParent, FunctionToOptimize

    start_time = time.perf_counter()

    try:
        # Parse qualified name to get function name and class name
        qualified_name_split = qualified_name.rsplit(".", maxsplit=1)
        if len(qualified_name_split) == 1:
            function_name, class_name = qualified_name_split[0], None
        else:
            function_name, class_name = qualified_name_split[1], qualified_name_split[0]

        # Create a FunctionToOptimize for the function
        # We don't have full line info here, so we'll use defaults
        parents: list[FunctionParent] = []
        if class_name:
            parents = [FunctionParent(name=class_name, type="ClassDef")]

        func_info = FunctionToOptimize(
            function_name=function_name,
            file_path=file_path,
            parents=parents,
            starting_line=1,
            ending_line=1,
            language=str(language),
        )

        # Find references
        references = find_references(func_info, project_root, tests_root, max_files=500)

        if not references:
            return ""

        # Format references as markdown code blocks
        calling_fns_details = format_references_as_markdown(references, file_path, project_root, language)

    except Exception as e:
        logger.debug("Error getting function references: %s", e)
        calling_fns_details = ""

    end_time = time.perf_counter()
    logger.debug("Got function references in %.2f seconds", end_time - start_time)
    return calling_fns_details


def format_references_as_markdown(references: list, file_path: Path, project_root: Path, language: str) -> str:
    """Format references as markdown code blocks with calling function code.

    Args:
        references: List of ReferenceInfo objects.
        file_path: Path to the source file (to exclude).
        project_root: Root of the project.
        language: The programming language.

    Returns:
        Markdown-formatted string.

    """
    # Group references by file
    refs_by_file: dict[Path, list] = {}
    for ref in references:
        # Exclude the source file's definition/import references
        if ref.file_path == file_path and ref.reference_type in ("import", "reexport"):
            continue

        if ref.file_path not in refs_by_file:
            refs_by_file[ref.file_path] = []
        refs_by_file[ref.file_path].append(ref)

    fn_call_context = ""
    context_len = 0

    for ref_file, file_refs in refs_by_file.items():
        if context_len > MAX_CONTEXT_LEN_REVIEW:
            break

        try:
            path_relative = ref_file.relative_to(project_root)
        except ValueError:
            continue

        lang_hint = "python"

        # Read the file to extract calling function context
        try:
            file_content = ref_file.read_text(encoding="utf-8")
            lines = file_content.splitlines()
        except Exception:
            continue

        # Get unique caller functions from this file
        callers_seen: set[str] = set()
        caller_contexts: list[str] = []

        for ref in file_refs:
            caller = ref.caller_function or "<module>"
            if caller in callers_seen:
                continue
            callers_seen.add(caller)

            # Extract context around the reference
            if ref.caller_function:
                # Try to extract the full calling function
                func_code = extract_calling_function_source(file_content, ref.caller_function, ref.line)
                if func_code:
                    caller_contexts.append(func_code)
                    context_len += len(func_code)
            else:
                # Module-level call - show a few lines of context
                start_line = max(0, ref.line - 3)
                end_line = min(len(lines), ref.line + 2)
                context_code = "\n".join(lines[start_line:end_line])
                caller_contexts.append(context_code)
                context_len += len(context_code)

        if caller_contexts:
            fn_call_context += f"```{lang_hint}:{path_relative.as_posix()}\n"
            fn_call_context += "\n".join(caller_contexts)
            fn_call_context += "\n```\n"

    return fn_call_context
