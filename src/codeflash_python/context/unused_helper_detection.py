"""Detection and reversion of unused helper functions in optimized code."""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.models.models import CodeString, CodeStringsMarkdown
from codeflash_python.static_analysis.code_replacer import replace_function_definitions_in_module

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext, FunctionSource
    from codeflash_core.models import FunctionToOptimize


logger = logging.getLogger("codeflash_python")


def revert_unused_helper_functions(
    project_root: Path, unused_helpers: list[FunctionSource], original_helper_code: dict[Path, str]
) -> None:
    """Revert unused helper functions back to their original definitions.

    Args:
        project_root: project_root
        unused_helpers: List of unused helper functions to revert
        original_helper_code: Dictionary mapping file paths to their original code

    """
    if not unused_helpers:
        return

    logger.debug("Reverting %s unused helper function(s) to original definitions", len(unused_helpers))

    # Resolve all path keys for consistent comparison (Windows 8.3 short names may differ from Jedi-resolved paths)
    resolved_original_helper_code = {p.resolve(): code for p, code in original_helper_code.items()}

    # Group unused helpers by file path
    unused_helpers_by_file = defaultdict(list)
    for helper in unused_helpers:
        unused_helpers_by_file[helper.file_path.resolve()].append(helper)

    # For each file, revert the unused helper functions to their original definitions
    for file_path, helpers_in_file in unused_helpers_by_file.items():
        if file_path in resolved_original_helper_code:
            try:
                # Get original code for this file
                original_code = resolved_original_helper_code[file_path]

                # Use the code replacer to selectively revert only the unused helper functions
                helper_names = [helper.qualified_name for helper in helpers_in_file]
                reverted_code = replace_function_definitions_in_module(
                    function_names=helper_names,
                    optimized_code=CodeStringsMarkdown(
                        code_strings=[
                            CodeString(code=original_code, file_path=Path(file_path).relative_to(project_root))
                        ]
                    ),  # Use original code as the "optimized" code to revert
                    module_abspath=file_path,
                    preexisting_objects=set(),  # Empty set since we're reverting
                    project_root_path=project_root,
                    should_add_global_assignments=False,  # since we revert helpers functions after applying the optimization, we know that the file already has global assignments added, otherwise they would be added twice.
                )

                if reverted_code:
                    logger.debug("Reverted unused helpers in %s: %s", file_path, ", ".join(helper_names))

            except Exception as e:
                logger.exception("Error reverting unused helpers in %s: %s", file_path, e)


def analyze_imports_in_optimized_code(
    optimized_ast: ast.AST, code_context: CodeOptimizationContext
) -> dict[str, set[str]]:
    """Analyze import statements in optimized code to map imported names to qualified helper names.

    Args:
        optimized_ast: The AST of the optimized code
        code_context: The code optimization context containing helper functions

    Returns:
        Dictionary mapping imported names to sets of possible qualified helper names

    """
    imported_names_map = defaultdict(set)

    # Precompute a two-level dict: module_name -> func_name -> [helpers]
    helpers_by_file_and_func = defaultdict(dict)
    helpers_by_file = defaultdict(list)  # preserved for "import module"
    for helper in code_context.helper_functions:
        jedi_type = helper.definition_type
        if jedi_type != "class":  # Include when definition_type is None (non-Python)
            func_name = helper.only_function_name
            module_name = helper.file_path.stem
            # Cache function lookup for this (module, func)
            helpers_by_file_and_func[module_name].setdefault(func_name, []).append(helper)
            helpers_by_file[module_name].append(helper)

    # Collect only import nodes to avoid per-node isinstance checks across the whole AST
    class _ImportCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.nodes: list[ast.AST] = []

        def visit_Import(self, node: ast.Import) -> None:
            self.nodes.append(node)
            # No need to recurse further for import nodes

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            self.nodes.append(node)
            # No need to recurse further for import-from nodes

    collector = _ImportCollector()
    collector.visit(optimized_ast)

    for node in collector.nodes:
        if isinstance(node, ast.ImportFrom):
            # Handle "from module import function" statements
            module_name = node.module
            if module_name:
                file_entry = helpers_by_file_and_func.get(module_name)
                if file_entry:
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        original_name = alias.name
                        helpers = file_entry.get(original_name)
                        if helpers:
                            imported_set = imported_names_map[imported_name]
                            for helper in helpers:
                                imported_set.add(helper.qualified_name)
                                imported_set.add(helper.fully_qualified_name)

        elif isinstance(node, ast.Import):
            # Handle "import module" statements
            for alias in node.names:
                imported_name = alias.asname if alias.asname else alias.name
                module_name = alias.name
                helpers = helpers_by_file.get(module_name)
                if helpers:
                    imported_set = imported_names_map[f"{imported_name}.{{func}}"]
                    for helper in helpers:
                        # For "import module" statements, functions would be called as module.function
                        full_call = f"{imported_name}.{helper.only_function_name}"
                        full_call_set = imported_names_map[full_call]
                        full_call_set.add(helper.qualified_name)
                        full_call_set.add(helper.fully_qualified_name)

    return dict(imported_names_map)


def find_target_node(
    root: ast.AST, function_to_optimize: FunctionToOptimize
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parents = function_to_optimize.parents
    node = root
    for parent in parents:
        # Fast loop: directly look for the matching ClassDef in node.body
        body = getattr(node, "body", None)
        if not body:
            return None
        for child in body:
            if isinstance(child, ast.ClassDef) and child.name == parent.name:
                node = child
                break
        else:
            return None

    # Now node is either the root or the target parent class; look for function
    body = getattr(node, "body", None)
    if not body:
        return None
    target_name = function_to_optimize.function_name
    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == target_name:
            return child
    return None


def detect_unused_helper_functions(
    function_to_optimize: FunctionToOptimize,
    code_context: CodeOptimizationContext,
    optimized_code: str | CodeStringsMarkdown,
) -> list[FunctionSource]:
    """Detect helper functions that are no longer called by the optimized entrypoint function.

    Args:
        function_to_optimize: The function to optimize
        code_context: The code optimization context containing helper functions
        optimized_code: The optimized code to analyze

    Returns:
        List of FunctionSource objects representing unused helper functions

    """
    if isinstance(optimized_code, CodeStringsMarkdown) and len(optimized_code.code_strings) > 0:
        return list(
            chain.from_iterable(
                detect_unused_helper_functions(function_to_optimize, code_context, code.code)
                for code in optimized_code.code_strings
            )
        )

    try:
        # Parse the optimized code to analyze function calls and imports
        optimized_ast = ast.parse(optimized_code)  # type: ignore[call-overload]

        # Find the optimized entrypoint function
        entrypoint_function_ast = find_target_node(optimized_ast, function_to_optimize)

        if not entrypoint_function_ast:
            logger.debug("Could not find entrypoint function %s in optimized code", function_to_optimize.function_name)
            return []

        # First, analyze imports to build a mapping of imported names to their original qualified names
        imported_names_map = analyze_imports_in_optimized_code(optimized_ast, code_context)

        # Extract all function calls and attribute references in the entrypoint function
        called_function_names = {function_to_optimize.function_name}
        for node in ast.walk(entrypoint_function_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Regular function call: function_name()
                    called_name = node.func.id
                    called_function_names.add(called_name)
                    # Also add the qualified name if this is an imported function
                    mapped_names = imported_names_map.get(called_name)
                    if mapped_names:
                        called_function_names.update(mapped_names)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method() or self.method() or module.function()
                    if isinstance(node.func.value, ast.Name):
                        attr_name = node.func.attr
                        value_id = node.func.value.id
                        if value_id == "self":
                            # self.method_name() -> add both method_name and ClassName.method_name
                            called_function_names.add(attr_name)
                            # For class methods, also add the qualified name
                            if hasattr(function_to_optimize, "parents") and function_to_optimize.parents:
                                class_name = function_to_optimize.parents[0].name
                                called_function_names.add(f"{class_name}.{attr_name}")
                        else:
                            called_function_names.add(attr_name)
                            full_call = f"{value_id}.{attr_name}"
                            called_function_names.add(full_call)
                            # Check if this is a module.function call that maps to a helper
                            mapped_names = imported_names_map.get(full_call)
                            if mapped_names:
                                called_function_names.update(mapped_names)
                    # Handle nested attribute access like obj.attr.method()
                    else:
                        called_function_names.add(node.func.attr)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                # Attribute reference without call: e.g. self._parse1 = self._parse_literal
                # This covers methods used as callbacks, stored in variables, passed as arguments, etc.
                attr_name = node.attr
                value_id = node.value.id
                if value_id == "self":
                    called_function_names.add(attr_name)
                    if hasattr(function_to_optimize, "parents") and function_to_optimize.parents:
                        class_name = function_to_optimize.parents[0].name
                        called_function_names.add(f"{class_name}.{attr_name}")
                else:
                    called_function_names.add(attr_name)
                    full_ref = f"{value_id}.{attr_name}"
                    called_function_names.add(full_ref)
                    mapped_names = imported_names_map.get(full_ref)
                    if mapped_names:
                        called_function_names.update(mapped_names)

        logger.debug("Functions called in optimized entrypoint: %s", called_function_names)
        logger.debug("Imported names mapping: %s", imported_names_map)

        # Find helper functions that are no longer called
        unused_helpers = []
        entrypoint_file_path = function_to_optimize.file_path
        for helper_function in code_context.helper_functions:
            jedi_type = helper_function.definition_type
            if jedi_type != "class":  # Include when definition_type is None (non-Python)
                # Check if the helper function is called using multiple name variants
                helper_qualified_name = helper_function.qualified_name
                helper_simple_name = helper_function.only_function_name
                helper_fully_qualified_name = helper_function.fully_qualified_name

                # Check membership efficiently - exit early on first match
                if (
                    helper_qualified_name in called_function_names
                    or helper_simple_name in called_function_names
                    or helper_fully_qualified_name in called_function_names
                ):
                    is_called = True
                # For cross-file helpers, also consider module-based calls
                elif helper_function.file_path != entrypoint_file_path:
                    # Add potential module.function combinations
                    module_name = helper_function.file_path.stem
                    module_call = f"{module_name}.{helper_simple_name}"
                    is_called = module_call in called_function_names
                else:
                    is_called = False

                if not is_called:
                    unused_helpers.append(helper_function)
                    logger.debug("Helper function %s is not called in optimized code", helper_qualified_name)
                else:
                    logger.debug("Helper function %s is still called in optimized code", helper_qualified_name)

    except Exception as e:
        logger.debug("Error detecting unused helper functions: %s", e)
        return []
    else:
        return unused_helpers
