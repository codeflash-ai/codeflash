from __future__ import annotations

import ast
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import isort
import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash.code_utils.formatter import sort_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent, TestingMode, VerificationType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeflash.models.models import CodePosition


def node_in_call_position(node: ast.AST, call_positions: list[CodePosition]) -> bool:
    if isinstance(node, ast.Call) and hasattr(node, "lineno") and hasattr(node, "col_offset"):
        for pos in call_positions:
            if (
                pos.line_no is not None
                and node.end_lineno is not None
                and node.lineno <= pos.line_no <= node.end_lineno
            ):
                if pos.line_no == node.lineno and node.col_offset <= pos.col_no:
                    return True
                if (
                    pos.line_no == node.end_lineno
                    and node.end_col_offset is not None
                    and node.end_col_offset >= pos.col_no
                ):
                    return True
                if node.lineno < pos.line_no < node.end_lineno:
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
        test_framework: str,
        call_positions: list[CodePosition],
        mode: TestingMode = TestingMode.BEHAVIOR,
    ) -> None:
        self.mode: TestingMode = mode
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.module_path = module_path
        self.test_framework = test_framework
        self.call_positions = call_positions
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name

    def find_and_update_line_node(
        self, test_node: ast.stmt, node_name: str, index: str, test_class_name: str | None = None
    ) -> Iterable[ast.stmt] | None:
        call_node = None
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call) and node_in_call_position(node, self.call_positions):
                call_node = node
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id

                    if self.function_object.is_async:
                        return [test_node]

                    node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                    node.args = [
                        ast.Name(id=function_name, ctx=ast.Load()),
                        ast.Constant(value=self.module_path),
                        ast.Constant(value=test_class_name or None),
                        ast.Constant(value=node_name),
                        ast.Constant(value=self.function_object.qualified_name),
                        ast.Constant(value=index),
                        ast.Name(id="codeflash_loop_index", ctx=ast.Load()),
                        *(
                            [ast.Name(id="codeflash_cur", ctx=ast.Load()), ast.Name(id="codeflash_con", ctx=ast.Load())]
                            if self.mode == TestingMode.BEHAVIOR
                            else []
                        ),
                        *call_node.args,
                    ]
                    node.keywords = call_node.keywords
                    break
                if isinstance(node.func, ast.Attribute):
                    function_to_test = node.func.attr
                    if function_to_test == self.function_object.function_name:
                        if self.function_object.is_async:
                            return [test_node]

                        function_name = ast.unparse(node.func)
                        node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                        node.args = [
                            ast.Name(id=function_name, ctx=ast.Load()),
                            ast.Constant(value=self.module_path),
                            ast.Constant(value=test_class_name or None),
                            ast.Constant(value=node_name),
                            ast.Constant(value=self.function_object.qualified_name),
                            ast.Constant(value=index),
                            ast.Name(id="codeflash_loop_index", ctx=ast.Load()),
                            *(
                                [
                                    ast.Name(id="codeflash_cur", ctx=ast.Load()),
                                    ast.Name(id="codeflash_con", ctx=ast.Load()),
                                ]
                                if self.mode == TestingMode.BEHAVIOR
                                else []
                            ),
                            *call_node.args,
                        ]
                        node.keywords = call_node.keywords
                        break

        if call_node is None:
            return None
        return [test_node]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # TODO: Ensure that this class inherits from unittest.TestCase. Don't modify non unittest.TestCase classes.
        for inner_node in ast.walk(node):
            if isinstance(inner_node, ast.FunctionDef):
                self.visit_FunctionDef(inner_node, node.name)

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, test_class_name: str | None = None) -> ast.FunctionDef:
        if node.name.startswith("test_"):
            did_update = False
            if self.test_framework == "unittest" and platform.system() != "Windows":
                # Only add timeout decorator on non-Windows platforms
                # Windows doesn't support SIGALRM signal required by timeout_decorator

                node.decorator_list.append(
                    ast.Call(
                        func=ast.Name(id="timeout_decorator.timeout", ctx=ast.Load()),
                        args=[ast.Constant(value=15)],
                        keywords=[],
                    )
                )
            i = len(node.body) - 1
            while i >= 0:
                line_node = node.body[i]
                # TODO: Validate if the functional call actually did not raise any exceptions

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
        test_framework: str,
        call_positions: list[CodePosition],
        mode: TestingMode = TestingMode.BEHAVIOR,
    ) -> None:
        self.mode = mode
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.module_path = module_path
        self.test_framework = test_framework
        self.call_positions = call_positions
        self.did_instrument = False
        # Track function call count per test function
        self.async_call_counter: dict[str, int] = {}
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Add timeout decorator for unittest test classes if needed
        if self.test_framework == "unittest":
            timeout_decorator = ast.Call(
                func=ast.Name(id="timeout_decorator.timeout", ctx=ast.Load()),
                args=[ast.Constant(value=15)],
                keywords=[],
            )
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name.startswith("test_")
                    and not any(
                        isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Name)
                        and d.func.id == "timeout_decorator.timeout"
                        for d in item.decorator_list
                    )
                ):
                    item.decorator_list.append(timeout_decorator)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        if not node.name.startswith("test_"):
            return node

        return self._process_test_function(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Only process test functions
        if not node.name.startswith("test_"):
            return node

        return self._process_test_function(node)

    def _process_test_function(
        self, node: ast.AsyncFunctionDef | ast.FunctionDef
    ) -> ast.AsyncFunctionDef | ast.FunctionDef:
        # Optimize the search for decorator presence
        if self.test_framework == "unittest":
            found_timeout = False
            for d in node.decorator_list:
                # Avoid isinstance(d.func, ast.Name) if d is not ast.Call
                if isinstance(d, ast.Call):
                    f = d.func
                    # Avoid attribute lookup if f is not ast.Name
                    if isinstance(f, ast.Name) and f.id == "timeout_decorator.timeout":
                        found_timeout = True
                        break
            if not found_timeout:
                timeout_decorator = ast.Call(
                    func=ast.Name(id="timeout_decorator.timeout", ctx=ast.Load()),
                    args=[ast.Constant(value=15)],
                    keywords=[],
                )
                node.decorator_list.append(timeout_decorator)

        # Initialize counter for this test function
        if node.name not in self.async_call_counter:
            self.async_call_counter[node.name] = 0

        new_body = []

        # Optimize ast.walk calls inside _instrument_statement, by scanning only relevant nodes
        for _i, stmt in enumerate(node.body):
            transformed_stmt, added_env_assignment = self._optimized_instrument_statement(stmt)

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

    def _instrument_statement(self, stmt: ast.stmt, _node_name: str) -> tuple[ast.stmt, bool]:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and self._is_target_call(node.value)
                and self._call_in_positions(node.value)
            ):
                # Check if this call is in one of our target positions
                return stmt, True  # Return original statement but signal we added env var

        return stmt, False

    def _is_target_call(self, call_node: ast.Call) -> bool:
        """Check if this call node is calling our target async function."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == self.function_object.function_name
        if isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == self.function_object.function_name
        return False

    def _call_in_positions(self, call_node: ast.Call) -> bool:
        if not hasattr(call_node, "lineno") or not hasattr(call_node, "col_offset"):
            return False

        return node_in_call_position(call_node, self.call_positions)

    # Optimized version: only walk child nodes for Await
    def _optimized_instrument_statement(self, stmt: ast.stmt) -> tuple[ast.stmt, bool]:
        # Stack-based DFS, manual for relevant Await nodes
        stack = [stmt]
        while stack:
            node = stack.pop()
            # Favor direct ast.Await detection
            if isinstance(node, ast.Await):
                val = node.value
                if isinstance(val, ast.Call) and self._is_target_call(val) and self._call_in_positions(val):
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
                    )
                else:
                    self.imported_as = FunctionToOptimize(
                        function_name=alias.asname,
                        parents=[],
                        file_path=self.function.file_path,
                        starting_line=self.function.starting_line,
                        ending_line=self.function.ending_line,
                        is_async=self.function.is_async,
                    )


def instrument_source_module_with_async_decorators(
    source_path: Path, function_to_optimize: FunctionToOptimize, mode: TestingMode = TestingMode.BEHAVIOR
) -> tuple[bool, str | None]:
    if not function_to_optimize.is_async:
        return False, None

    try:
        with source_path.open(encoding="utf8") as f:
            source_code = f.read()

        modified_code, decorator_added = add_async_decorator_to_function(source_code, function_to_optimize, mode)

        if decorator_added:
            return True, modified_code

    except Exception:
        return False, None
    else:
        return False, None


def inject_async_profiling_into_existing_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    test_framework: str,
    mode: TestingMode = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    """Inject profiling for async function calls by setting environment variables before each call."""
    with test_path.open(encoding="utf8") as f:
        test_code = f.read()

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logger.exception(f"Syntax error in code in file - {test_path}")
        return False, None
    # TODO: Pass the full name of function here, otherwise we can run into namespace clashes
    test_module_path = module_name_from_file_path(test_path, tests_project_root)
    import_visitor = FunctionImportedAsVisitor(function_to_optimize)
    import_visitor.visit(tree)
    func = import_visitor.imported_as

    async_instrumenter = AsyncCallInstrumenter(func, test_module_path, test_framework, call_positions, mode=mode)
    tree = async_instrumenter.visit(tree)

    if not async_instrumenter.did_instrument:
        return False, None

    # Add necessary imports
    new_imports = [ast.Import(names=[ast.alias(name="os")])]
    if test_framework == "unittest":
        new_imports.append(ast.Import(names=[ast.alias(name="timeout_decorator")]))

    tree.body = [*new_imports, *tree.body]
    return True, isort.code(ast.unparse(tree), float_to_top=True)


def inject_profiling_into_existing_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    test_framework: str,
    mode: TestingMode = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    if function_to_optimize.is_async:
        return inject_async_profiling_into_existing_test(
            test_path, call_positions, function_to_optimize, tests_project_root, test_framework, mode
        )

    with test_path.open(encoding="utf8") as f:
        test_code = f.read()
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logger.exception(f"Syntax error in code in file - {test_path}")
        return False, None

    test_module_path = module_name_from_file_path(test_path, tests_project_root)
    import_visitor = FunctionImportedAsVisitor(function_to_optimize)
    import_visitor.visit(tree)
    func = import_visitor.imported_as

    tree = InjectPerfOnly(func, test_module_path, test_framework, call_positions, mode=mode).visit(tree)
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
    ]
    if mode == TestingMode.BEHAVIOR:
        new_imports.extend(
            [ast.Import(names=[ast.alias(name="sqlite3")]), ast.Import(names=[ast.alias(name="dill", asname="pickle")])]
        )
    if test_framework == "unittest" and platform.system() != "Windows":
        new_imports.append(ast.Import(names=[ast.alias(name="timeout_decorator")]))
    additional_functions = [create_wrapper_function(mode)]

    tree.body = [*new_imports, *additional_functions, *tree.body]
    return True, isort.code(ast.unparse(tree), float_to_top=True)


def create_wrapper_function(mode: TestingMode = TestingMode.BEHAVIOR) -> ast.FunctionDef:
    lineno = 1
    wrapper_body: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="test_id", ctx=ast.Store())],
            value=ast.JoinedStr(
                values=[
                    ast.FormattedValue(value=ast.Name(id="codeflash_test_module_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="codeflash_test_class_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="codeflash_test_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="codeflash_line_id", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="codeflash_loop_index", ctx=ast.Load()), conversion=-1),
                ]
            ),
            lineno=lineno + 1,
        ),
        ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id="hasattr", ctx=ast.Load()),
                    args=[ast.Name(id="codeflash_wrap", ctx=ast.Load()), ast.Constant(value="index")],
                    keywords=[],
                ),
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id="codeflash_wrap", ctx=ast.Load()), attr="index", ctx=ast.Store()
                        )
                    ],
                    value=ast.Dict(keys=[], values=[]),
                    lineno=lineno + 3,
                )
            ],
            orelse=[],
            lineno=lineno + 2,
        ),
        ast.If(
            test=ast.Compare(
                left=ast.Name(id="test_id", ctx=ast.Load()),
                ops=[ast.In()],
                comparators=[
                    ast.Attribute(value=ast.Name(id="codeflash_wrap", ctx=ast.Load()), attr="index", ctx=ast.Load())
                ],
            ),
            body=[
                ast.AugAssign(
                    target=ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id="codeflash_wrap", ctx=ast.Load()), attr="index", ctx=ast.Load()
                        ),
                        slice=ast.Name(id="test_id", ctx=ast.Load()),
                        ctx=ast.Store(),
                    ),
                    op=ast.Add(),
                    value=ast.Constant(value=1),
                    lineno=lineno + 5,
                )
            ],
            orelse=[
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id="codeflash_wrap", ctx=ast.Load()), attr="index", ctx=ast.Load()
                            ),
                            slice=ast.Name(id="test_id", ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=0),
                    lineno=lineno + 6,
                )
            ],
            lineno=lineno + 4,
        ),
        ast.Assign(
            targets=[ast.Name(id="codeflash_test_index", ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Attribute(value=ast.Name(id="codeflash_wrap", ctx=ast.Load()), attr="index", ctx=ast.Load()),
                slice=ast.Name(id="test_id", ctx=ast.Load()),
                ctx=ast.Load(),
            ),
            lineno=lineno + 7,
        ),
        ast.Assign(
            targets=[ast.Name(id="invocation_id", ctx=ast.Store())],
            value=ast.JoinedStr(
                values=[
                    ast.FormattedValue(value=ast.Name(id="codeflash_line_id", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value="_"),
                    ast.FormattedValue(value=ast.Name(id="codeflash_test_index", ctx=ast.Load()), conversion=-1),
                ]
            ),
            lineno=lineno + 8,
        ),
        *(
            [
                ast.Assign(
                    targets=[ast.Name(id="test_stdout_tag", ctx=ast.Store())],
                    value=ast.JoinedStr(
                        values=[
                            ast.FormattedValue(
                                value=ast.Name(id="codeflash_test_module_name", ctx=ast.Load()), conversion=-1
                            ),
                            ast.Constant(value=":"),
                            ast.FormattedValue(
                                value=ast.IfExp(
                                    test=ast.Name(id="codeflash_test_class_name", ctx=ast.Load()),
                                    body=ast.BinOp(
                                        left=ast.Name(id="codeflash_test_class_name", ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Constant(value="."),
                                    ),
                                    orelse=ast.Constant(value=""),
                                ),
                                conversion=-1,
                            ),
                            ast.FormattedValue(value=ast.Name(id="codeflash_test_name", ctx=ast.Load()), conversion=-1),
                            ast.Constant(value=":"),
                            ast.FormattedValue(
                                value=ast.Name(id="codeflash_function_name", ctx=ast.Load()), conversion=-1
                            ),
                            ast.Constant(value=":"),
                            ast.FormattedValue(
                                value=ast.Name(id="codeflash_loop_index", ctx=ast.Load()), conversion=-1
                            ),
                            ast.Constant(value=":"),
                            ast.FormattedValue(value=ast.Name(id="invocation_id", ctx=ast.Load()), conversion=-1),
                        ]
                    ),
                    lineno=lineno + 9,
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=[
                            ast.JoinedStr(
                                values=[
                                    ast.Constant(value="!$######"),
                                    ast.FormattedValue(
                                        value=ast.Name(id="test_stdout_tag", ctx=ast.Load()), conversion=-1
                                    ),
                                    ast.Constant(value="######$!"),
                                ]
                            )
                        ],
                        keywords=[],
                    )
                ),
            ]
        ),
        ast.Assign(
            targets=[ast.Name(id="exception", ctx=ast.Store())], value=ast.Constant(value=None), lineno=lineno + 10
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="gc", ctx=ast.Load()), attr="disable", ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
            lineno=lineno + 9,
        ),
        ast.Try(
            body=[
                ast.Assign(
                    targets=[ast.Name(id="counter", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="time", ctx=ast.Load()), attr="perf_counter_ns", ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[],
                    ),
                    lineno=lineno + 11,
                ),
                ast.Assign(
                    targets=[ast.Name(id="return_value", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="codeflash_wrapped", ctx=ast.Load()),
                        args=[ast.Starred(value=ast.Name(id="args", ctx=ast.Load()), ctx=ast.Load())],
                        keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))],
                    ),
                    lineno=lineno + 12,
                ),
                ast.Assign(
                    targets=[ast.Name(id="codeflash_duration", ctx=ast.Store())],
                    value=ast.BinOp(
                        left=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="time", ctx=ast.Load()), attr="perf_counter_ns", ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[],
                        ),
                        op=ast.Sub(),
                        right=ast.Name(id="counter", ctx=ast.Load()),
                    ),
                    lineno=lineno + 13,
                ),
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="Exception", ctx=ast.Load()),
                    name="e",
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id="codeflash_duration", ctx=ast.Store())],
                            value=ast.BinOp(
                                left=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="time", ctx=ast.Load()),
                                        attr="perf_counter_ns",
                                        ctx=ast.Load(),
                                    ),
                                    args=[],
                                    keywords=[],
                                ),
                                op=ast.Sub(),
                                right=ast.Name(id="counter", ctx=ast.Load()),
                            ),
                            lineno=lineno + 15,
                        ),
                        ast.Assign(
                            targets=[ast.Name(id="exception", ctx=ast.Store())],
                            value=ast.Name(id="e", ctx=ast.Load()),
                            lineno=lineno + 13,
                        ),
                    ],
                    lineno=lineno + 14,
                )
            ],
            orelse=[],
            finalbody=[],
            lineno=lineno + 11,
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="gc", ctx=ast.Load()), attr="enable", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[
                    ast.JoinedStr(
                        values=[
                            ast.Constant(value="!######"),
                            ast.FormattedValue(value=ast.Name(id="test_stdout_tag", ctx=ast.Load()), conversion=-1),
                            *(
                                [
                                    ast.Constant(value=":"),
                                    ast.FormattedValue(
                                        value=ast.Name(id="codeflash_duration", ctx=ast.Load()), conversion=-1
                                    ),
                                ]
                                if mode == TestingMode.PERFORMANCE
                                else []
                            ),
                            ast.Constant(value="######!"),
                        ]
                    )
                ],
                keywords=[],
            )
        ),
        *(
            [
                ast.Assign(
                    targets=[ast.Name(id="pickled_return_value", ctx=ast.Store())],
                    value=ast.IfExp(
                        test=ast.Name(id="exception", ctx=ast.Load()),
                        body=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="pickle", ctx=ast.Load()), attr="dumps", ctx=ast.Load()
                            ),
                            args=[ast.Name(id="exception", ctx=ast.Load())],
                            keywords=[],
                        ),
                        orelse=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="pickle", ctx=ast.Load()), attr="dumps", ctx=ast.Load()
                            ),
                            args=[ast.Name(id="return_value", ctx=ast.Load())],
                            keywords=[],
                        ),
                    ),
                    lineno=lineno + 18,
                )
            ]
            if mode == TestingMode.BEHAVIOR
            else []
        ),
        *(
            [
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="codeflash_cur", ctx=ast.Load()), attr="execute", ctx=ast.Load()
                        ),
                        args=[
                            ast.Constant(value="INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"),
                            ast.Tuple(
                                elts=[
                                    ast.Name(id="codeflash_test_module_name", ctx=ast.Load()),
                                    ast.Name(id="codeflash_test_class_name", ctx=ast.Load()),
                                    ast.Name(id="codeflash_test_name", ctx=ast.Load()),
                                    ast.Name(id="codeflash_function_name", ctx=ast.Load()),
                                    ast.Name(id="codeflash_loop_index", ctx=ast.Load()),
                                    ast.Name(id="invocation_id", ctx=ast.Load()),
                                    ast.Name(id="codeflash_duration", ctx=ast.Load()),
                                    ast.Name(id="pickled_return_value", ctx=ast.Load()),
                                    ast.Constant(value=VerificationType.FUNCTION_CALL.value),
                                ],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    lineno=lineno + 20,
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="codeflash_con", ctx=ast.Load()), attr="commit", ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[],
                    ),
                    lineno=lineno + 21,
                ),
            ]
            if mode == TestingMode.BEHAVIOR
            else []
        ),
        ast.If(
            test=ast.Name(id="exception", ctx=ast.Load()),
            body=[ast.Raise(exc=ast.Name(id="exception", ctx=ast.Load()), cause=None, lineno=lineno + 22)],
            orelse=[],
            lineno=lineno + 22,
        ),
        ast.Return(value=ast.Name(id="return_value", ctx=ast.Load()), lineno=lineno + 19),
    ]
    return ast.FunctionDef(
        name="codeflash_wrap",
        args=ast.arguments(
            args=[
                ast.arg(arg="codeflash_wrapped", annotation=None),
                ast.arg(arg="codeflash_test_module_name", annotation=None),
                ast.arg(arg="codeflash_test_class_name", annotation=None),
                ast.arg(arg="codeflash_test_name", annotation=None),
                ast.arg(arg="codeflash_function_name", annotation=None),
                ast.arg(arg="codeflash_line_id", annotation=None),
                ast.arg(arg="codeflash_loop_index", annotation=None),
                *([ast.arg(arg="codeflash_cur", annotation=None)] if mode == TestingMode.BEHAVIOR else []),
                *([ast.arg(arg="codeflash_con", annotation=None)] if mode == TestingMode.BEHAVIOR else []),
            ],
            vararg=ast.arg(arg="args"),
            kwarg=ast.arg(arg="kwargs"),
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=wrapper_body,
        lineno=lineno,
        decorator_list=[],
        returns=None,
    )


class AsyncDecoratorAdder(cst.CSTTransformer):
    """Transformer that adds async decorator to async function definitions."""

    def __init__(self, function: FunctionToOptimize, mode: TestingMode = TestingMode.BEHAVIOR) -> None:
        """Initialize the transformer.

        Args:
        ----
            function: The FunctionToOptimize object representing the target async function.
            mode: The testing mode to determine which decorator to apply.

        """
        super().__init__()
        self.function = function
        self.mode = mode
        self.qualified_name_parts = function.qualified_name.split(".")
        self.context_stack = []
        self.added_decorator = False

        # Choose decorator based on mode
        self.decorator_name = (
            "codeflash_behavior_async" if mode == TestingMode.BEHAVIOR else "codeflash_performance_async"
        )

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Track when we enter a class
        self.context_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:  # noqa: ARG002
        # Pop the context when we leave a class
        self.context_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Track when we enter a function
        self.context_stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Check if this is an async function and matches our target
        if original_node.asynchronous is not None and self.context_stack == self.qualified_name_parts:
            # Check if the decorator is already present
            has_decorator = any(
                self._is_target_decorator(decorator.decorator) for decorator in original_node.decorators
            )

            # Only add the decorator if it's not already there
            if not has_decorator:
                new_decorator = cst.Decorator(decorator=cst.Name(value=self.decorator_name))

                # Add our new decorator to the existing decorators
                updated_decorators = [new_decorator, *list(updated_node.decorators)]
                updated_node = updated_node.with_changes(decorators=tuple(updated_decorators))
                self.added_decorator = True

        # Pop the context when we leave a function
        self.context_stack.pop()
        return updated_node

    def _is_target_decorator(self, decorator_node: cst.Name | cst.Attribute | cst.Call) -> bool:
        """Check if a decorator matches our target decorator name."""
        if isinstance(decorator_node, cst.Name):
            return decorator_node.value in {
                "codeflash_trace_async",
                "codeflash_behavior_async",
                "codeflash_performance_async",
            }
        if isinstance(decorator_node, cst.Call) and isinstance(decorator_node.func, cst.Name):
            return decorator_node.func.value in {
                "codeflash_trace_async",
                "codeflash_behavior_async",
                "codeflash_performance_async",
            }
        return False


class AsyncDecoratorImportAdder(cst.CSTTransformer):
    """Transformer that adds the import for async decorators."""

    def __init__(self, mode: TestingMode = TestingMode.BEHAVIOR) -> None:
        self.mode = mode
        self.has_import = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # Check if the async decorator import is already present
        if (
            isinstance(node.module, cst.Attribute)
            and isinstance(node.module.value, cst.Attribute)
            and isinstance(node.module.value.value, cst.Name)
            and node.module.value.value.value == "codeflash"
            and node.module.value.attr.value == "code_utils"
            and node.module.attr.value == "codeflash_wrap_decorator"
            and not isinstance(node.names, cst.ImportStar)
        ):
            decorator_name = (
                "codeflash_behavior_async" if self.mode == TestingMode.BEHAVIOR else "codeflash_performance_async"
            )
            for import_alias in node.names:
                if import_alias.name.value == decorator_name:
                    self.has_import = True

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: ARG002
        # If the import is already there, don't add it again
        if self.has_import:
            return updated_node

        # Choose import based on mode
        decorator_name = (
            "codeflash_behavior_async" if self.mode == TestingMode.BEHAVIOR else "codeflash_performance_async"
        )

        # Parse the import statement into a CST node
        import_node = cst.parse_statement(f"from codeflash.code_utils.codeflash_wrap_decorator import {decorator_name}")

        # Add the import to the module's body
        return updated_node.with_changes(body=[import_node, *list(updated_node.body)])


def add_async_decorator_to_function(
    source_code: str, function: FunctionToOptimize, mode: TestingMode = TestingMode.BEHAVIOR
) -> tuple[str, bool]:
    """Add async decorator to an async function definition.

    Args:
    ----
        source_code: The source code to modify.
        function: The FunctionToOptimize object representing the target async function.
        mode: The testing mode to determine which decorator to apply.

    Returns:
    -------
        Tuple of (modified_source_code, was_decorator_added).

    """
    if not function.is_async:
        return source_code, False

    try:
        module = cst.parse_module(source_code)

        # Add the decorator to the function
        decorator_transformer = AsyncDecoratorAdder(function, mode)
        module = module.visit(decorator_transformer)

        # Add the import if decorator was added
        if decorator_transformer.added_decorator:
            import_transformer = AsyncDecoratorImportAdder(mode)
            module = module.visit(import_transformer)

        return sort_imports(code=module.code, float_to_top=True), decorator_transformer.added_decorator
    except Exception as e:
        logger.exception(f"Error adding async decorator to function {function.qualified_name}: {e}")
        return source_code, False


def create_instrumented_source_module_path(source_path: Path, temp_dir: Path) -> Path:
    instrumented_filename = f"instrumented_{source_path.name}"
    return temp_dir / instrumented_filename
