"""AST-based import analysis for filtering test files by target function imports."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.models.models import TestsInFile

logger = logging.getLogger("codeflash_python")


class ImportAnalyzer(ast.NodeVisitor):
    """AST-based analyzer to check if any qualified names from function_names_to_find are imported or used in a test file."""

    def __init__(self, function_names_to_find: set[str]) -> None:
        self.function_names_to_find = function_names_to_find
        self.found_any_target_function: bool = False
        self.found_qualified_name = None
        self.imported_modules: set[str] = set()
        self.has_dynamic_imports: bool = False
        self.wildcard_modules: set[str] = set()
        # Track aliases: alias_name -> original_name
        self.alias_mapping: dict[str, str] = {}
        # Track instances: variable_name -> class_name
        self.instance_mapping: dict[str, str] = {}

        # Precompute function_names for prefix search
        # For prefix match, store mapping from prefix-root to candidates for O(1) matching
        self.exact_names = function_names_to_find
        self.prefix_roots: dict[str, list[str]] = {}
        # Precompute sets for faster lookup during visit_Attribute()
        self.dot_names: set[str] = set()
        self.dot_methods: dict[str, set[str]] = {}
        self.class_method_to_target: dict[tuple[str, str], str] = {}

        # Optimize prefix-roots and dot_methods construction
        add_dot_methods = self.dot_methods.setdefault
        add_prefix_roots = self.prefix_roots.setdefault
        dot_names_add = self.dot_names.add
        class_method_to_target = self.class_method_to_target
        for name in function_names_to_find:
            if "." in name:
                root, method = name.rsplit(".", 1)
                dot_names_add(name)
                add_dot_methods(method, set()).add(root)
                class_method_to_target[(root, method)] = name
                root_prefix = name.split(".", 1)[0]
                add_prefix_roots(root_prefix, []).append(name)

    def visit_Import(self, node: ast.Import) -> None:
        """Handle 'import module' statements."""
        if self.found_any_target_function:
            return

        for alias in node.names:
            module_name = alias.asname if alias.asname else alias.name
            self.imported_modules.add(module_name)

            # Check for dynamic import modules
            if alias.name == "importlib":
                self.has_dynamic_imports = True

            # Check if module itself is a target qualified name
            if module_name in self.function_names_to_find:
                self.found_any_target_function = True
                self.found_qualified_name = module_name
                return
            # Check if any target qualified name starts with this module
            for target_func in self.function_names_to_find:
                if target_func.startswith(f"{module_name}."):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments, especially class instantiations."""
        if self.found_any_target_function:
            return

        # Check if the assignment is a class instantiation
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            class_name = value.func.id
            if class_name in self.imported_modules:
                # Map the variable to the actual class name (handling aliases)
                original_class = self.alias_mapping.get(class_name, class_name)
                # Use list comprehension for direct assignment to instance_mapping, reducing loop overhead
                targets = node.targets
                instance_mapping = self.instance_mapping
                # since ast.Name nodes are heavily used, avoid local lookup for isinstance
                # and reuse locals for faster attribute access
                for target in targets:
                    if isinstance(target, ast.Name):
                        instance_mapping[target.id] = original_class

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle 'from module import name' statements."""
        if self.found_any_target_function:
            return

        mod = node.module
        if not mod:
            return

        fnames = self.exact_names
        proots = self.prefix_roots

        for alias in node.names:
            aname = alias.name
            if aname == "*":
                self.wildcard_modules.add(mod)
                continue

            imported_name = alias.asname if alias.asname else aname
            self.imported_modules.add(imported_name)

            if alias.asname:
                self.alias_mapping[imported_name] = aname

            # Fast check for dynamic import
            if mod == "importlib" and aname == "import_module":
                self.has_dynamic_imports = True

            qname = f"{mod}.{aname}"

            # Fast exact match check
            if aname in fnames:
                self.found_any_target_function = True
                self.found_qualified_name = aname
                return
            if qname in fnames:
                self.found_any_target_function = True
                self.found_qualified_name = qname
                return

            # Check if any target function is a method of the imported class/module
            # Be conservative except when an alias is used (which requires exact method matching)
            for target_func in fnames:
                if "." in target_func:
                    class_name, _method_name = target_func.split(".", 1)
                    if aname == class_name and not alias.asname:
                        self.found_any_target_function = True
                        self.found_qualified_name = target_func
                        return
                        # If an alias is used, track it for later method access detection
                        # The actual method usage will be detected in visit_Attribute

            prefix = qname + "."
            # Only bother if one of the targets startswith the prefix-root
            candidates = proots.get(qname, ())
            for target_func in candidates:
                if target_func.startswith(prefix):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Handle attribute access like module.function_name."""
        if self.found_any_target_function:
            return

        # Check if this is accessing a target function through an imported module

        node_value = node.value
        node_attr = node.attr

        # Check if this is accessing a target function through an imported module

        # Accessing a target function through an imported module (fast path for imported modules)
        val_id = getattr(node_value, "id", None)
        if val_id is not None and val_id in self.imported_modules:
            if node_attr in self.function_names_to_find:
                self.found_any_target_function = True
                self.found_qualified_name = node_attr
                return
            # Methods via imported modules using precomputed _dot_methods and _class_method_to_target
            roots_possible = self.dot_methods.get(node_attr)
            if roots_possible:
                imported_name = val_id
                original_name = self.alias_mapping.get(imported_name, imported_name)
                if original_name in roots_possible:
                    self.found_any_target_function = True
                    self.found_qualified_name = self.class_method_to_target[(original_name, node_attr)]
                    return
                # Also check if the imported name itself (without resolving alias) matches
                # This handles cases where the class itself is the target
                if imported_name in roots_possible:
                    self.found_any_target_function = True
                    self.found_qualified_name = self.class_method_to_target.get(
                        (imported_name, node_attr), f"{imported_name}.{node_attr}"
                    )
                    return

        # Methods on instance variables (tighten type check and lookup, important for larger ASTs)
        if val_id is not None and val_id in self.instance_mapping:
            class_name = self.instance_mapping[val_id]
            roots_possible = self.dot_methods.get(node_attr)
            if roots_possible and class_name in roots_possible:
                self.found_any_target_function = True
                self.found_qualified_name = self.class_method_to_target[(class_name, node_attr)]
                return

        # Check for dynamic import match
        if self.has_dynamic_imports and node_attr in self.function_names_to_find:
            self.found_any_target_function = True
            self.found_qualified_name = node_attr
            return

        # Replace self.generic_visit with base class impl directly: removes an attribute lookup
        if not self.found_any_target_function:
            ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls, particularly __import__."""
        if self.found_any_target_function:
            return

        # Check if this is a __import__ call
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self.has_dynamic_imports = True
            # When __import__ is used, any target function could potentially be imported
            # Be conservative and assume it might import target functions

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Handle direct name usage like target_function()."""
        if self.found_any_target_function:
            return

        # Check for __import__ usage
        if node.id == "__import__":
            self.has_dynamic_imports = True

        # Check if this is a direct usage of a target function name
        # This catches cases like: result = target_function()
        if node.id in self.function_names_to_find:
            self.found_any_target_function = True
            self.found_qualified_name = node.id
            return

        # Check if this name could come from a wildcard import
        for wildcard_module in self.wildcard_modules:
            for target_func in self.function_names_to_find:
                # Check if target_func is from this wildcard module and name matches
                if target_func.startswith(f"{wildcard_module}.") and target_func.endswith(f".{node.id}"):
                    self.found_any_target_function = True
                    self.found_qualified_name = target_func
                    return

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Override generic_visit to stop traversal if a target function is found."""
        if self.found_any_target_function:
            return
        # Direct base call improves run speed (avoids extra method resolution)
        self.fast_generic_visit(node)

    def fast_generic_visit(self, node: ast.AST) -> None:
        """Faster generic_visit: Inline traversal, avoiding method resolution overhead.

        Short-circuits (returns) if found_any_target_function is True.
        """
        # This logic is derived from ast.NodeVisitor.generic_visit, but with optimizations.
        if self.found_any_target_function:
            return

        # Local bindings for improved lookup speed (10-15% faster for inner loop)
        visit_cache = type(self).__dict__
        node_fields = node._fields

        # Use manual stack for iterative traversal, replacing recursion
        stack = [(node_fields, node)]
        append = stack.append
        pop = stack.pop

        while stack:
            fields, curr_node = pop()
            for field in fields:
                value = getattr(curr_node, field, None)
                if isinstance(value, list):
                    for item in value:
                        if self.found_any_target_function:
                            return
                        if isinstance(item, ast.AST):
                            # Method resolution: fast dict lookup first, then getattr fallback
                            meth = visit_cache.get("visit_" + item.__class__.__name__)
                            if meth is not None:
                                meth(self, item)
                            else:
                                append((item._fields, item))
                    continue
                if isinstance(value, ast.AST):
                    if self.found_any_target_function:
                        return
                    meth = visit_cache.get("visit_" + value.__class__.__name__)
                    if meth is not None:
                        meth(self, value)
                    else:
                        append((value._fields, value))


def analyze_imports_in_test_file(test_file_path: Path | str, target_functions: set[str]) -> bool:
    """Analyze a test file to see if it imports any of the target functions."""
    try:
        with Path(test_file_path).open("r", encoding="utf-8") as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=str(test_file_path))
        analyzer = ImportAnalyzer(target_functions)
        analyzer.visit(tree)
    except (SyntaxError, FileNotFoundError) as e:
        logger.debug("Failed to analyze imports in %s: %s", test_file_path, e)
        return True

    if analyzer.found_any_target_function:
        # logger.debug(f"Test file {test_file_path} imports target function: {analyzer.found_qualified_name}")
        return True

    # Be conservative with dynamic imports - if __import__ is used and a target function
    # is referenced, we should process the file
    if analyzer.has_dynamic_imports:
        # Check if any target function name appears as a string literal or direct usage
        for target_func in target_functions:
            if target_func in source_code:
                # logger.debug(f"Test file {test_file_path} has dynamic imports and references {target_func}")
                return True

    # logger.debug(f"Test file {test_file_path} does not import any target functions.")
    return False


def filter_test_files_by_imports(
    file_to_test_map: dict[Path, list[TestsInFile]], target_functions: set[str]
) -> dict[Path, list[TestsInFile]]:
    """Filter test files based on import analysis to reduce Jedi processing.

    Args:
        file_to_test_map: Original mapping of test files to test functions
        target_functions: Set of function names we're optimizing

    Returns:
        Filtered mapping of test files to test functions

    """
    if not target_functions:
        return file_to_test_map

    # logger.debug(f"Target functions for import filtering: {target_functions}")

    filtered_map = {}
    for test_file, test_functions in file_to_test_map.items():
        should_process = analyze_imports_in_test_file(test_file, target_functions)
        if should_process:
            filtered_map[test_file] = test_functions

    logger.debug(
        "analyzed %s test files for imports, filtered down to %s relevant files",
        len(file_to_test_map),
        len(filtered_map),
    )
    return filtered_map
