from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.models.models import TestingMode
from codeflash_core.models import FunctionParent, FunctionToOptimize
from codeflash_python.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash_python.code_utils.formatter import sort_imports
from codeflash_python.verification.device_sync import detect_frameworks_from_code
from codeflash_python.verification.wrapper_generation import create_wrapper_function

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeflash.models.models import CodePosition


logger = logging.getLogger("codeflash_python")


@dataclass(frozen=True)
class FunctionCallNodeArguments:
    args: list[ast.expr]
    keywords: list[ast.keyword]


def get_call_arguments(call_node: ast.Call) -> FunctionCallNodeArguments:
    return FunctionCallNodeArguments(call_node.args, call_node.keywords)


def node_in_call_position(node: ast.AST, call_positions: list[CodePosition]) -> bool:
    # Profile: The most meaningful speedup here is to reduce attribute lookup and to localize call_positions if not empty.
    # Small optimizations for tight loop:
    if isinstance(node, ast.Call):
        node_lineno = getattr(node, "lineno", None)
        node_col_offset = getattr(node, "col_offset", None)
        node_end_lineno = getattr(node, "end_lineno", None)
        node_end_col_offset = getattr(node, "end_col_offset", None)
        if node_lineno is not None and node_col_offset is not None and node_end_lineno is not None:
            # Faster loop: reduce attribute lookups, use local variables for conditionals.
            for pos in call_positions:
                pos_line = pos.line_no
                if pos_line is not None and node_lineno <= pos_line <= node_end_lineno:
                    if pos_line == node_lineno and node_col_offset <= pos.col_no:
                        return True
                    if (
                        pos_line == node_end_lineno
                        and node_end_col_offset is not None
                        and node_end_col_offset >= pos.col_no
                    ):
                        return True
                    if node_lineno < pos_line < node_end_lineno:
                        return True
    return False


def is_argument_name(name: str, arguments_node: ast.arguments) -> bool:
    return any(
        element.arg == name
        for attribute_name in dir(arguments_node)
        if isinstance(attribute := getattr(arguments_node, attribute_name), list)
        for element in attribute
        if isinstance(element, ast.arg)
    )


class InjectPerfOnly(ast.NodeTransformer):
    def __init__(
        self,
        function: FunctionToOptimize,
        module_path: str,
        call_positions: list[CodePosition],
        mode: TestingMode = TestingMode.BEHAVIOR,
    ) -> None:
        self.mode: TestingMode = mode
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.module_path = module_path
        self.call_positions = call_positions
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name

    def find_and_update_line_node(
        self, test_node: ast.stmt, node_name: str, index: str, test_class_name: str | None = None
    ) -> Iterable[ast.stmt] | None:
        # Major optimization: since ast.walk is *very* expensive for big trees and only checks for ast.Call,
        # it's much more efficient to visit nodes manually. We'll only descend into expressions/statements.

        # Helper for manual walk
        def iter_ast_calls(node):
            # Generator to yield each ast.Call in test_node, preserves node identity
            stack = [node]
            while stack:
                n = stack.pop()
                if isinstance(n, ast.Call):
                    yield n
                # Instead of using ast.walk (which calls iter_child_nodes under the hood in Python, which copy lists and stack-frames for EVERY node),
                # do a specialized BFS with only the necessary attributes
                for _field, value in ast.iter_fields(n):
                    if isinstance(value, list):
                        for item in reversed(value):
                            if isinstance(item, ast.AST):
                                stack.append(item)
                    elif isinstance(value, ast.AST):
                        stack.append(value)

        # This change improves from O(N) stack-frames per child-node to a single stack, less python call overhead
        return_statement = [test_node]
        call_node = None

        # Minor optimization: Convert mode, function_name, test_class_name, qualified_name, etc to locals
        fn_obj = self.function_object
        module_path = self.module_path
        mode = self.mode
        qualified_name = fn_obj.qualified_name

        # Use locals for all 'current' values, only look up class/function/constant AST object once.
        codeflash_loop_index = ast.Name(id="codeflash_loop_index", ctx=ast.Load())
        codeflash_cur = ast.Name(id="codeflash_cur", ctx=ast.Load())
        codeflash_con = ast.Name(id="codeflash_con", ctx=ast.Load())

        for node in iter_ast_calls(test_node):
            if not node_in_call_position(node, self.call_positions):
                continue

            call_node = node
            all_args = get_call_arguments(call_node)
            # Two possible call types: Name and Attribute
            node_func = node.func

            if isinstance(node_func, ast.Name):
                function_name = node_func.id

                # Check if this is the function we want to instrument
                if function_name != fn_obj.function_name:
                    continue

                if fn_obj.is_async:
                    return [test_node]

                # Build once, reuse objects.
                inspect_name = ast.Name(id="inspect", ctx=ast.Load())
                bind_call = ast.Assign(
                    targets=[ast.Name(id="_call__bound__arguments", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(value=inspect_name, attr="signature", ctx=ast.Load()),
                                args=[ast.Name(id=function_name, ctx=ast.Load())],
                                keywords=[],
                            ),
                            attr="bind",
                            ctx=ast.Load(),
                        ),
                        args=all_args.args,
                        keywords=all_args.keywords,
                    ),
                    lineno=test_node.lineno,
                    col_offset=test_node.col_offset,
                )

                apply_defaults = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                            attr="apply_defaults",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    lineno=test_node.lineno + 1,
                    col_offset=test_node.col_offset,
                )

                node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                base_args = [
                    ast.Name(id=function_name, ctx=ast.Load()),
                    ast.Constant(value=module_path),
                    ast.Constant(value=test_class_name or None),
                    ast.Constant(value=node_name),
                    ast.Constant(value=qualified_name),
                    ast.Constant(value=index),
                    codeflash_loop_index,
                ]
                # Extend with BEHAVIOR extras if needed
                if mode == TestingMode.BEHAVIOR:
                    base_args += [codeflash_cur, codeflash_con]
                # Extend with call args (performance) or starred bound args (behavior)
                if mode == TestingMode.PERFORMANCE:
                    base_args += call_node.args
                else:
                    base_args.append(
                        ast.Starred(
                            value=ast.Attribute(
                                value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                                attr="args",
                                ctx=ast.Load(),
                            ),
                            ctx=ast.Load(),
                        )
                    )
                node.args = base_args
                # Prepare keywords
                if mode == TestingMode.BEHAVIOR:
                    node.keywords = [
                        ast.keyword(
                            value=ast.Attribute(
                                value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                                attr="kwargs",
                                ctx=ast.Load(),
                            )
                        )
                    ]
                else:
                    node.keywords = call_node.keywords

                return_statement = (
                    [bind_call, apply_defaults, test_node] if mode == TestingMode.BEHAVIOR else [test_node]
                )
                break
            if isinstance(node_func, ast.Attribute):
                function_to_test = node_func.attr
                if function_to_test == fn_obj.function_name:
                    if fn_obj.is_async:
                        return [test_node]

                    # Create the signature binding statements

                    # Unparse only once
                    function_name_expr = ast.parse(ast.unparse(node_func), mode="eval").body

                    inspect_name = ast.Name(id="inspect", ctx=ast.Load())
                    bind_call = ast.Assign(
                        targets=[ast.Name(id="_call__bound__arguments", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Call(
                                    func=ast.Attribute(value=inspect_name, attr="signature", ctx=ast.Load()),
                                    args=[function_name_expr],
                                    keywords=[],
                                ),
                                attr="bind",
                                ctx=ast.Load(),
                            ),
                            args=all_args.args,
                            keywords=all_args.keywords,
                        ),
                        lineno=test_node.lineno,
                        col_offset=test_node.col_offset,
                    )

                    apply_defaults = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                                attr="apply_defaults",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                        lineno=test_node.lineno + 1,
                        col_offset=test_node.col_offset,
                    )

                    node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                    base_args = [
                        function_name_expr,
                        ast.Constant(value=module_path),
                        ast.Constant(value=test_class_name or None),
                        ast.Constant(value=node_name),
                        ast.Constant(value=qualified_name),
                        ast.Constant(value=index),
                        codeflash_loop_index,
                    ]
                    if mode == TestingMode.BEHAVIOR:
                        base_args += [codeflash_cur, codeflash_con]
                    if mode == TestingMode.PERFORMANCE:
                        base_args += call_node.args
                    else:
                        base_args.append(
                            ast.Starred(
                                value=ast.Attribute(
                                    value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                                    attr="args",
                                    ctx=ast.Load(),
                                ),
                                ctx=ast.Load(),
                            )
                        )
                    node.args = base_args
                    if mode == TestingMode.BEHAVIOR:
                        node.keywords = [
                            ast.keyword(
                                value=ast.Attribute(
                                    value=ast.Name(id="_call__bound__arguments", ctx=ast.Load()),
                                    attr="kwargs",
                                    ctx=ast.Load(),
                                )
                            )
                        ]
                    else:
                        node.keywords = call_node.keywords

                    # Return the signature binding statements along with the test_node
                    return_statement = (
                        [bind_call, apply_defaults, test_node] if mode == TestingMode.BEHAVIOR else [test_node]
                    )
                    break

        if call_node is None:
            return None
        return return_statement

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # TODO: Ensure that this class inherits from unittest.TestCase. Don't modify non unittest.TestCase classes.
        for inner_node in ast.walk(node):
            if isinstance(inner_node, ast.FunctionDef):
                self.visit_FunctionDef(inner_node, node.name)

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, test_class_name: str | None = None) -> ast.FunctionDef:
        if node.name.startswith("test_"):
            did_update = False
            i = len(node.body) - 1
            while i >= 0:
                line_node = node.body[i]
                if isinstance(line_node, (ast.With, ast.For, ast.While, ast.If)):
                    j = len(line_node.body) - 1
                    while j >= 0:
                        compound_line_node: ast.stmt = line_node.body[j]
                        internal_node: ast.AST
                        for internal_node in ast.walk(compound_line_node):
                            if isinstance(internal_node, (ast.stmt, ast.Assign)):
                                updated_node = self.find_and_update_line_node(
                                    internal_node, node.name, str(i) + "_" + str(j), test_class_name
                                )
                                if updated_node is not None:
                                    line_node.body[j : j + 1] = updated_node
                                    did_update = True
                                    break
                        j -= 1
                else:
                    updated_node = self.find_and_update_line_node(line_node, node.name, str(i), test_class_name)
                    if updated_node is not None:
                        node.body[i : i + 1] = updated_node
                        did_update = True
                i -= 1
            if did_update:
                node.body = [
                    ast.Assign(
                        targets=[ast.Name(id="codeflash_loop_index", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="int", ctx=ast.Load()),
                            args=[
                                ast.Subscript(
                                    value=ast.Attribute(
                                        value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load()
                                    ),
                                    slice=ast.Constant(value="CODEFLASH_LOOP_INDEX"),
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        lineno=node.lineno + 2,
                        col_offset=node.col_offset,
                    ),
                    *(
                        [
                            ast.Assign(
                                targets=[ast.Name(id="codeflash_iteration", ctx=ast.Store())],
                                value=ast.Subscript(
                                    value=ast.Attribute(
                                        value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load()
                                    ),
                                    slice=ast.Constant(value="CODEFLASH_TEST_ITERATION"),
                                    ctx=ast.Load(),
                                ),
                                lineno=node.lineno + 1,
                                col_offset=node.col_offset,
                            ),
                            ast.Assign(
                                targets=[ast.Name(id="codeflash_con", ctx=ast.Store())],
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="sqlite3", ctx=ast.Load()), attr="connect", ctx=ast.Load()
                                    ),
                                    args=[
                                        ast.JoinedStr(
                                            values=[
                                                ast.Constant(
                                                    value=f"{get_run_tmp_file(Path('test_return_values_')).as_posix()}"
                                                ),
                                                ast.FormattedValue(
                                                    value=ast.Name(id="codeflash_iteration", ctx=ast.Load()),
                                                    conversion=-1,
                                                ),
                                                ast.Constant(value=".sqlite"),
                                            ]
                                        )
                                    ],
                                    keywords=[],
                                ),
                                lineno=node.lineno + 3,
                                col_offset=node.col_offset,
                            ),
                            ast.Assign(
                                targets=[ast.Name(id="codeflash_cur", ctx=ast.Store())],
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="codeflash_con", ctx=ast.Load()),
                                        attr="cursor",
                                        ctx=ast.Load(),
                                    ),
                                    args=[],
                                    keywords=[],
                                ),
                                lineno=node.lineno + 4,
                                col_offset=node.col_offset,
                            ),
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="codeflash_cur", ctx=ast.Load()),
                                        attr="execute",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.Constant(
                                            value="CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT,"
                                            " test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT,"
                                            " loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)"
                                        )
                                    ],
                                    keywords=[],
                                ),
                                lineno=node.lineno + 5,
                                col_offset=node.col_offset,
                            ),
                        ]
                        if self.mode == TestingMode.BEHAVIOR
                        else []
                    ),
                    *node.body,
                    *(
                        [
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="codeflash_con", ctx=ast.Load()), attr="close", ctx=ast.Load()
                                    ),
                                    args=[],
                                    keywords=[],
                                )
                            )
                        ]
                        if self.mode == TestingMode.BEHAVIOR
                        else []
                    ),
                ]
        return node


class AsyncCallInstrumenter(ast.NodeTransformer):
    def __init__(
        self,
        function: FunctionToOptimize,
        module_path: str,
        call_positions: list[CodePosition],
        mode: TestingMode = TestingMode.BEHAVIOR,
    ) -> None:
        self.mode = mode
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.module_path = module_path
        self.call_positions = call_positions
        self.did_instrument = False
        # Track function call count per test function
        self.async_call_counter: dict[str, int] = {}
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        result = self.generic_visit(node)
        assert isinstance(result, ast.ClassDef)
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        if not node.name.startswith("test_"):
            return node

        result = self.process_test_function(node)
        assert isinstance(result, ast.AsyncFunctionDef)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Only process test functions
        if not node.name.startswith("test_"):
            return node

        result = self.process_test_function(node)
        assert isinstance(result, ast.FunctionDef)
        return result

    def process_test_function(
        self, node: ast.AsyncFunctionDef | ast.FunctionDef
    ) -> ast.AsyncFunctionDef | ast.FunctionDef:
        # Initialize counter for this test function
        if node.name not in self.async_call_counter:
            self.async_call_counter[node.name] = 0

        new_body = []

        # Optimize ast.walk calls inside instrument_statement, by scanning only relevant nodes
        for _i, stmt in enumerate(node.body):
            transformed_stmt, added_env_assignment = self.optimized_instrument_statement(stmt)

            if added_env_assignment:
                current_call_index = self.async_call_counter[node.name]
                self.async_call_counter[node.name] += 1

                env_assignment = ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load()
                            ),
                            slice=ast.Constant(value="CODEFLASH_CURRENT_LINE_ID"),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=f"{current_call_index}"),
                    lineno=stmt.lineno if hasattr(stmt, "lineno") else 1,
                )
                new_body.append(env_assignment)
                self.did_instrument = True

            new_body.append(transformed_stmt)

        node.body = new_body
        return node

    def instrument_statement(self, stmt: ast.stmt, _node_name: str) -> tuple[ast.stmt, bool]:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and self.is_target_call(node.value)
                and self.call_in_positions(node.value)
            ):
                # Check if this call is in one of our target positions
                return stmt, True  # Return original statement but signal we added env var

        return stmt, False

    def is_target_call(self, call_node: ast.Call) -> bool:
        """Check if this call node is calling our target async function."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == self.function_object.function_name
        if isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == self.function_object.function_name
        return False

    def call_in_positions(self, call_node: ast.Call) -> bool:
        if not hasattr(call_node, "lineno") or not hasattr(call_node, "col_offset"):
            return False

        return node_in_call_position(call_node, self.call_positions)

    # Optimized version: only walk child nodes for Await
    def optimized_instrument_statement(self, stmt: ast.stmt) -> tuple[ast.stmt, bool]:
        # Stack-based DFS, manual for relevant Await nodes
        stack = [stmt]
        while stack:
            node = stack.pop()
            # Favor direct ast.Await detection
            if isinstance(node, ast.Await):
                val = node.value
                if isinstance(val, ast.Call) and self.is_target_call(val) and self.call_in_positions(val):
                    return stmt, True
            # Use _fields instead of ast.walk for less allocations
            for fname in getattr(node, "_fields", ()):
                child = getattr(node, fname, None)
                if isinstance(child, list):
                    stack.extend(child)
                elif isinstance(child, ast.AST):
                    stack.append(child)
        return stmt, False


class FunctionImportedAsVisitor(ast.NodeVisitor):
    """Checks if a function has been imported as an alias. We only care about the alias then.

    from numpy import array as np_array
    np_array is what we want
    """

    def __init__(self, function: FunctionToOptimize) -> None:
        assert len(function.parents) <= 1, "Only support functions with one or less parent"
        self.imported_as = function
        self.function = function
        if function.parents:
            self.to_match = function.parents[0].name
        else:
            self.to_match = function.function_name

    # TODO: Validate if the function imported is actually from the right module
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == self.to_match and hasattr(alias, "asname") and alias.asname is not None:
                if self.function.parents:
                    self.imported_as = FunctionToOptimize(
                        function_name=self.function.function_name,
                        parents=[FunctionParent(alias.asname, "ClassDef")],
                        file_path=self.function.file_path,
                        starting_line=self.function.starting_line,
                        ending_line=self.function.ending_line,
                        is_async=self.function.is_async,
                        language=self.function.language,
                    )
                else:
                    self.imported_as = FunctionToOptimize(
                        function_name=alias.asname,
                        parents=[],
                        file_path=self.function.file_path,
                        starting_line=self.function.starting_line,
                        ending_line=self.function.ending_line,
                        is_async=self.function.is_async,
                        language=self.function.language,
                    )


def inject_async_profiling_into_existing_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    mode: TestingMode = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    """Inject profiling for async function calls by setting environment variables before each call."""
    with test_path.open(encoding="utf8") as f:
        test_code = f.read()

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logger.exception("Syntax error in code in file - %s", test_path)
        return False, None
    # TODO: Pass the full name of function here, otherwise we can run into namespace clashes
    test_module_path = module_name_from_file_path(test_path, tests_project_root)
    import_visitor = FunctionImportedAsVisitor(function_to_optimize)
    import_visitor.visit(tree)
    func = import_visitor.imported_as

    async_instrumenter = AsyncCallInstrumenter(func, test_module_path, call_positions, mode=mode)
    tree = async_instrumenter.visit(tree)

    if not async_instrumenter.did_instrument:
        return False, None

    # Add necessary imports
    new_imports = [ast.Import(names=[ast.alias(name="os")])]

    tree.body = [*new_imports, *tree.body]
    return True, sort_imports(ast.unparse(tree), float_to_top=True)


def inject_profiling_into_existing_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    mode: TestingMode = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    tests_project_root = tests_project_root.resolve()
    if function_to_optimize.is_async:
        return inject_async_profiling_into_existing_test(
            test_path, call_positions, function_to_optimize, tests_project_root, mode
        )

    with test_path.open(encoding="utf8") as f:
        test_code = f.read()

    used_frameworks = detect_frameworks_from_code(test_code)
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logger.exception("Syntax error in code in file - %s", test_path)
        return False, None

    test_module_path = module_name_from_file_path(test_path, tests_project_root)
    import_visitor = FunctionImportedAsVisitor(function_to_optimize)
    import_visitor.visit(tree)
    func = import_visitor.imported_as

    tree = InjectPerfOnly(func, test_module_path, call_positions, mode=mode).visit(tree)
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
    ]
    if mode == TestingMode.BEHAVIOR:
        new_imports.extend(
            [
                ast.Import(names=[ast.alias(name="inspect")]),
                ast.Import(names=[ast.alias(name="sqlite3")]),
                ast.Import(names=[ast.alias(name="dill", asname="pickle")]),
            ]
        )
    # Add framework imports for GPU sync code (needed when framework is only imported via submodule)
    for framework_name, framework_alias in used_frameworks.items():
        if framework_alias == framework_name:
            # Only add import if we're using the framework name directly (not an alias)
            # This handles cases like "from torch.nn import Module" where torch needs to be imported
            new_imports.append(ast.Import(names=[ast.alias(name=framework_name)]))
        else:
            # If there's an alias, use it (e.g., "import torch as th")
            new_imports.append(ast.Import(names=[ast.alias(name=framework_name, asname=framework_alias)]))
    additional_functions = [create_wrapper_function(mode, used_frameworks)]

    tree.body = [*new_imports, *additional_functions, *tree.body]
    return True, sort_imports(ast.unparse(tree), float_to_top=True)
