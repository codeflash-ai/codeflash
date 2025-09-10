"""Test file processor for parallel import and Jedi analysis."""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.discovery.parallel_models import (
    ImportAnalysisResult,
    ImportAnalysisTask,
    JediAnalysisResult,
    JediAnalysisTask,
)
from codeflash.models.models import (
    CodePosition,
    FunctionCalledInTest,
    TestsInFile,
    TestType,
)


# Regex patterns from the original code
PYTEST_PARAMETERIZED_TEST_NAME_REGEX = re.compile(r"[\[\]]")
UNITTEST_PARAMETERIZED_TEST_NAME_REGEX = re.compile(r"^test_\w+_\d+(?:_\w+)*")
UNITTEST_STRIP_NUMBERED_SUFFIX_REGEX = re.compile(r"_\d+(?:_\w+)*$")
FUNCTION_NAME_REGEX = re.compile(r"([^.]+)\.([a-zA-Z0-9_]+)$")


class ImportAnalyzer(ast.NodeVisitor):
    """AST-based analyzer to check if any qualified names from function_names_to_find are imported or used in a test file.
    
    This is a copy of the ImportAnalyzer from discover_unit_tests.py to enable parallel processing.
    """

    def __init__(self, function_names_to_find: Set[str]) -> None:
        self.function_names_to_find = function_names_to_find
        self.found_any_target_function: bool = False
        self.found_qualified_name = None
        self.imported_modules: Set[str] = set()
        self.has_dynamic_imports: bool = False
        self.wildcard_modules: Set[str] = set()
        # Track aliases: alias_name -> original_name
        self.alias_mapping: Dict[str, str] = {}

        # Precompute function_names for prefix search
        # For prefix match, store mapping from prefix-root to candidates for O(1) matching
        self._exact_names = function_names_to_find
        self._prefix_roots: Dict[str, List[str]] = {}
        for name in function_names_to_find:
            if "." in name:
                root = name.split(".", 1)[0]
                self._prefix_roots.setdefault(root, []).append(name)

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

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle 'from module import name' statements."""
        if self.found_any_target_function:
            return

        mod = node.module
        if not mod:
            return

        fnames = self._exact_names
        proots = self._prefix_roots

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
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.imported_modules
            and node.attr in self.function_names_to_find
        ):
            self.found_any_target_function = True
            self.found_qualified_name = node.attr
            return

        if isinstance(node.value, ast.Name) and node.value.id in self.imported_modules:
            for target_func in self.function_names_to_find:
                if "." in target_func:
                    class_name, method_name = target_func.rsplit(".", 1)
                    if node.attr == method_name:
                        imported_name = node.value.id
                        original_name = self.alias_mapping.get(imported_name, imported_name)
                        if original_name == class_name:
                            self.found_any_target_function = True
                            self.found_qualified_name = target_func
                            return

        # Check if this is accessing a target function through a dynamically imported module
        # Only if we've detected dynamic imports are being used
        if self.has_dynamic_imports and node.attr in self.function_names_to_find:
            self.found_any_target_function = True
            self.found_qualified_name = node.attr
            return

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Handle direct name usage like target_function()."""
        if self.found_any_target_function:
            return

        # Check for __import__ usage
        if node.id == "__import__":
            self.has_dynamic_imports = True

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
        super().generic_visit(node)


class TestFileProcessor:
    """Handles processing of individual test files in worker processes."""
    
    @staticmethod
    def process_import_analysis(task: ImportAnalysisTask) -> ImportAnalysisResult:
        """Analyze imports in a test file to check for target functions.
        
        Args:
            task: Import analysis task containing file path and target functions
            
        Returns:
            ImportAnalysisResult with analysis results
        """
        try:
            with task.test_file_path.open("r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=str(task.test_file_path))
            analyzer = ImportAnalyzer(task.target_functions)
            analyzer.visit(tree)
            
            has_target_imports = analyzer.found_any_target_function
            
            if has_target_imports:
                logger.debug(f"Test file {task.test_file_path} imports target function: {analyzer.found_qualified_name}")
            else:
                logger.debug(f"Test file {task.test_file_path} does not import any target functions.")
                
            return ImportAnalysisResult(
                test_file_path=task.test_file_path,
                has_target_imports=has_target_imports
            )
            
        except (SyntaxError, FileNotFoundError) as e:
            logger.debug(f"Failed to analyze imports in {task.test_file_path}: {e}")
            # Return True on error to be safe and include the file for processing
            return ImportAnalysisResult(
                test_file_path=task.test_file_path,
                has_target_imports=True,
                error=str(e)
            )
        except Exception as e:
            logger.warning(f"Unexpected error analyzing imports in {task.test_file_path}: {e}")
            return ImportAnalysisResult(
                test_file_path=task.test_file_path,
                has_target_imports=True,
                error=str(e)
            )
    
    @staticmethod
    def process_jedi_analysis(task: JediAnalysisTask) -> JediAnalysisResult:
        """Use Jedi to analyze test file and find function references.
        
        Args:
            task: Jedi analysis task containing test file and configuration
            
        Returns:
            JediAnalysisResult with function mappings and counts
        """
        try:
            import jedi
            
            # Initialize Jedi project
            jedi_project = jedi.Project(path=task.project_root)
            script = jedi.Script(path=task.test_file, project=jedi_project)
            
            function_to_test_map = defaultdict(set)
            num_discovered_tests = 0
            num_discovered_replay_tests = 0
            
            # Get all names with references and definitions
            all_names = script.get_names(all_scopes=True, references=True, definitions=True)
            
            # Filter and create lookup dictionaries
            top_level_functions = {}
            top_level_classes = {}
            all_defs = []

            for name in all_names:
                if name.type == "function":
                    top_level_functions[name.name] = name
                    all_defs.append(name)
                elif name.type == "class":
                    top_level_classes[name.name] = name
                    
            # Process test functions based on framework
            test_functions = TestFileProcessor._extract_test_functions(
                task.test_functions, task.test_framework, top_level_functions, top_level_classes, all_defs
            )
            
            # Analyze function references
            test_function_names_set = {func.function_name for func in test_functions}
            relevant_names = []

            names_with_full_name = [name for name in all_names if name.full_name is not None]

            for name in names_with_full_name:
                match = FUNCTION_NAME_REGEX.search(name.full_name)
                if match and match.group(1) in test_function_names_set:
                    relevant_names.append((name, match.group(1)))

            # Process each relevant name to find function definitions
            test_functions_by_name = defaultdict(list)
            for func in test_functions:
                test_functions_by_name[func.function_name].append(func)

            for name, scope in relevant_names:
                try:
                    definition = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    logger.debug(f"Error getting definition for {name}: {e}")
                    continue
                    
                try:
                    if not definition or definition[0].type != "function":
                        continue
                        
                    definition_obj = definition[0]
                    definition_path = str(definition_obj.module_path)

                    project_root_str = str(task.project_root)
                    if (
                        definition_path.startswith(project_root_str + os.sep)
                        and definition_obj.module_name != name.module_name
                        and definition_obj.full_name is not None
                    ):
                        # Compute qualified name
                        module_prefix = definition_obj.module_name + "."
                        full_name_without_module_prefix = definition_obj.full_name.replace(module_prefix, "", 1)
                        qualified_name_with_modules_from_root = f"{module_name_from_file_path(definition_obj.module_path, task.project_root)}.{full_name_without_module_prefix}"

                        for test_func in test_functions_by_name[scope]:
                            if test_func.parameters is not None:
                                if task.test_framework == "pytest":
                                    scope_test_function = f"{test_func.function_name}[{test_func.parameters}]"
                                else:  # unittest
                                    scope_test_function = f"{test_func.function_name}_{test_func.parameters}"
                            else:
                                scope_test_function = test_func.function_name

                            function_to_test_map[qualified_name_with_modules_from_root].add(
                                FunctionCalledInTest(
                                    tests_in_file=TestsInFile(
                                        test_file=task.test_file,
                                        test_class=test_func.test_class,
                                        test_function=scope_test_function,
                                        test_type=test_func.test_type,
                                    ),
                                    position=CodePosition(line_no=name.line, col_no=name.column),
                                )
                            )
                            
                            if test_func.test_type == TestType.REPLAY_TEST:
                                num_discovered_replay_tests += 1

                            num_discovered_tests += 1
                            
                except Exception as e:
                    logger.debug(f"Error processing definition for {name}: {e}")
                    continue
                    
            return JediAnalysisResult(
                test_file=task.test_file,
                function_mappings=dict(function_to_test_map),
                test_count=num_discovered_tests,
                replay_test_count=num_discovered_replay_tests
            )
            
        except Exception as e:
            logger.warning(f"Failed to process Jedi analysis for {task.test_file}: {e}")
            return JediAnalysisResult(
                test_file=task.test_file,
                function_mappings={},
                test_count=0,
                replay_test_count=0,
                error=str(e)
            )
    
    @staticmethod
    def _extract_test_functions(test_functions_list, test_framework, top_level_functions, top_level_classes, all_defs):
        """Extract test functions based on framework type."""
        from dataclasses import dataclass
        from typing import Optional
        
        @dataclass(frozen=True)
        class TestFunction:
            function_name: str
            test_class: Optional[str]
            parameters: Optional[str]
            test_type: TestType
        
        test_functions = set()
        
        if test_framework == "pytest":
            for function in test_functions_list:
                if "[" in function.test_function:
                    function_name = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(function.test_function)[0]
                    parameters = PYTEST_PARAMETERIZED_TEST_NAME_REGEX.split(function.test_function)[1]
                    if function_name in top_level_functions:
                        test_functions.add(
                            TestFunction(function_name, function.test_class, parameters, function.test_type)
                        )
                elif function.test_function in top_level_functions:
                    test_functions.add(
                        TestFunction(function.test_function, function.test_class, None, function.test_type)
                    )
                elif UNITTEST_PARAMETERIZED_TEST_NAME_REGEX.match(function.test_function):
                    base_name = UNITTEST_STRIP_NUMBERED_SUFFIX_REGEX.sub("", function.test_function)
                    if base_name in top_level_functions:
                        test_functions.add(
                            TestFunction(
                                function_name=base_name,
                                test_class=function.test_class,
                                parameters=function.test_function,
                                test_type=function.test_type,
                            )
                        )

        elif test_framework == "unittest":
            functions_to_search = [elem.test_function for elem in test_functions_list]
            test_suites = {elem.test_class for elem in test_functions_list}

            matching_names = test_suites & top_level_classes.keys()
            for matched_name in matching_names:
                for def_name in all_defs:
                    if (
                        def_name.type == "function"
                        and def_name.full_name is not None
                        and f".{matched_name}." in def_name.full_name
                    ):
                        for function in functions_to_search:
                            (is_parameterized, new_function, parameters) = TestFileProcessor._discover_parameters_unittest(function)

                            if is_parameterized and new_function == def_name.name:
                                test_functions.add(
                                    TestFunction(
                                        function_name=def_name.name,
                                        test_class=matched_name,
                                        parameters=parameters,
                                        test_type=test_functions_list[0].test_type,
                                    )
                                )
                            elif function == def_name.name:
                                test_functions.add(
                                    TestFunction(
                                        function_name=def_name.name,
                                        test_class=matched_name,
                                        parameters=None,
                                        test_type=test_functions_list[0].test_type,
                                    )
                                )
        
        return test_functions
    
    @staticmethod
    def _discover_parameters_unittest(function_name: str):
        """Discover parameters for unittest functions."""
        function_name_parts = function_name.split("_")
        if len(function_name_parts) > 1 and function_name_parts[-1].isdigit():
            return True, "_".join(function_name_parts[:-1]), function_name_parts[-1]
        return False, function_name, None