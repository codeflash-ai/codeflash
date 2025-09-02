from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import isort

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
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


class AsyncIOGatherRemover(ast.NodeTransformer):
    def _contains_asyncio_gather(self, node: ast.AST) -> bool:
        """Check if a node contains asyncio.gather calls."""
        stack = [node]
        while stack:
            child_node = stack.pop()
            # Check for `asyncio.gather`
            if isinstance(child_node, ast.Call):
                func = child_node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "asyncio"
                    and func.attr == "gather"
                ):
                    return True

                # Check for direct `gather`
                if isinstance(func, ast.Name) and func.id == "gather":
                    return True

            stack.extend(ast.iter_child_nodes(child_node))
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
        if node.name.startswith("test_") and self._contains_asyncio_gather(node):
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef | None:
        if node.name.startswith("test_") and self._contains_asyncio_gather(node):
            return None
        return self.generic_visit(node)


class InjectPerfOnly(ast.NodeTransformer):
    def __init__(
        self,
        function: FunctionToOptimize,
        module_path: str,
        test_framework: str,
        call_positions: list[CodePosition],
        mode: TestingMode = TestingMode.BEHAVIOR,
        *,
        is_async: bool = False,
    ) -> None:
        self.mode: TestingMode = mode
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.module_path = module_path
        self.test_framework = test_framework
        self.call_positions = call_positions
        self.is_async = is_async
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name

    def find_and_update_line_node(
        self, test_node: ast.stmt, node_name: str, index: str, test_class_name: str | None = None
    ) -> Iterable[ast.stmt] | None:
        call_node = None
        await_node = None

        for node in ast.walk(test_node):
            if isinstance(node, ast.Call) and node_in_call_position(node, self.call_positions):
                call_node = node
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
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

            # Check for awaited function calls
            elif (
                isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and node_in_call_position(node.value, self.call_positions)
            ):
                call_node = node.value
                await_node = node
                if isinstance(call_node.func, ast.Name):
                    function_name = call_node.func.id
                    call_node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                    call_node.args = [
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
                    call_node.keywords = call_node.keywords
                    # Keep the await wrapper around the modified call
                    await_node.value = call_node
                    break
                if isinstance(call_node.func, ast.Attribute):
                    function_to_test = call_node.func.attr
                    if function_to_test == self.function_object.function_name:
                        function_name = ast.unparse(call_node.func)
                        call_node.func = ast.Name(id="codeflash_wrap", ctx=ast.Load())
                        call_node.args = [
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
                        call_node.keywords = call_node.keywords
                        # Keep the await wrapper around the modified call
                        await_node.value = call_node
                        break

        if call_node is None:
            return None
        return [test_node]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # TODO: Ensure that this class inherits from unittest.TestCase. Don't modify non unittest.TestCase classes.
        for inner_node in ast.walk(node):
            if isinstance(inner_node, ast.FunctionDef):
                self.visit_FunctionDef(inner_node, node.name)
            elif isinstance(inner_node, ast.AsyncFunctionDef):
                self.visit_AsyncFunctionDef(inner_node, node.name)

        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef, test_class_name: str | None = None
    ) -> ast.AsyncFunctionDef:
        sync_node = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            lineno=node.lineno,
            col_offset=node.col_offset if hasattr(node, "col_offset") else 0,
        )
        processed_sync = self.visit_FunctionDef(sync_node, test_class_name)
        return ast.AsyncFunctionDef(
            name=processed_sync.name,
            args=processed_sync.args,
            body=processed_sync.body,
            decorator_list=processed_sync.decorator_list,
            returns=processed_sync.returns,
            lineno=processed_sync.lineno,
            col_offset=processed_sync.col_offset if hasattr(processed_sync, "col_offset") else 0,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef, test_class_name: str | None = None) -> ast.FunctionDef:
        if node.name.startswith("test_"):
            did_update = False
            if self.test_framework == "unittest":
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
                                                ast.Constant(value=f"{get_run_tmp_file(Path('test_return_values_'))}"),
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


def inject_profiling_into_existing_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    test_framework: str,
    mode: TestingMode = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
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

    is_async = function_to_optimize.is_async
    logger.debug(f"Using async status from discovery phase for {function_to_optimize.function_name}: {is_async}")

    if is_async:
        asyncio_gather_remover = AsyncIOGatherRemover()
        tree = asyncio_gather_remover.visit(tree)

    tree = InjectPerfOnly(func, test_module_path, test_framework, call_positions, mode=mode, is_async=is_async).visit(
        tree
    )
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
    ]
    if mode == TestingMode.BEHAVIOR:
        new_imports.extend(
            [ast.Import(names=[ast.alias(name="sqlite3")]), ast.Import(names=[ast.alias(name="dill", asname="pickle")])]
        )
    if test_framework == "unittest":
        new_imports.append(ast.Import(names=[ast.alias(name="timeout_decorator")]))
    if is_async:
        new_imports.append(ast.Import(names=[ast.alias(name="inspect")]))
    tree.body = [*new_imports, create_wrapper_function(mode, is_async=is_async), *tree.body]
    return True, isort.code(ast.unparse(tree), float_to_top=True)


def create_wrapper_function(
    mode: TestingMode = TestingMode.BEHAVIOR, *, is_async: bool = False
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    lineno = 1
    wrapper_body: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="test_id", ctx=ast.Store())],
            value=ast.JoinedStr(
                values=[
                    ast.FormattedValue(value=ast.Name(id="test_module_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="test_class_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="test_name", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="line_id", ctx=ast.Load()), conversion=-1),
                    ast.Constant(value=":"),
                    ast.FormattedValue(value=ast.Name(id="loop_index", ctx=ast.Load()), conversion=-1),
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
                    ast.FormattedValue(value=ast.Name(id="line_id", ctx=ast.Load()), conversion=-1),
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
                            ast.FormattedValue(value=ast.Name(id="test_module_name", ctx=ast.Load()), conversion=-1),
                            ast.Constant(value=":"),
                            ast.FormattedValue(
                                value=ast.IfExp(
                                    test=ast.Name(id="test_class_name", ctx=ast.Load()),
                                    body=ast.BinOp(
                                        left=ast.Name(id="test_class_name", ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Constant(value="."),
                                    ),
                                    orelse=ast.Constant(value=""),
                                ),
                                conversion=-1,
                            ),
                            ast.FormattedValue(value=ast.Name(id="test_name", ctx=ast.Load()), conversion=-1),
                            ast.Constant(value=":"),
                            ast.FormattedValue(value=ast.Name(id="function_name", ctx=ast.Load()), conversion=-1),
                            ast.Constant(value=":"),
                            ast.FormattedValue(value=ast.Name(id="loop_index", ctx=ast.Load()), conversion=-1),
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
                # For async wrappers
                # Call the wrapped function first, then check if result is awaitable before awaiting.
                # This handles mixed scenarios where async tests might call both sync and async functions.
                *(
                    [
                        ast.Assign(
                            targets=[ast.Name(id="ret", ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id="wrapped", ctx=ast.Load()),
                                args=[ast.Starred(value=ast.Name(id="args", ctx=ast.Load()), ctx=ast.Load())],
                                keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))],
                            ),
                            lineno=lineno + 12,
                        ),
                        ast.If(
                            test=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="inspect", ctx=ast.Load()), attr="isawaitable", ctx=ast.Load()
                                ),
                                args=[ast.Name(id="ret", ctx=ast.Load())],
                                keywords=[],
                            ),
                            body=[
                                ast.Assign(
                                    targets=[ast.Name(id="counter", ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Name(id="time", ctx=ast.Load()),
                                            attr="perf_counter_ns",
                                            ctx=ast.Load(),
                                        ),
                                        args=[],
                                        keywords=[],
                                    ),
                                    lineno=lineno + 14,
                                ),
                                ast.Assign(
                                    targets=[ast.Name(id="return_value", ctx=ast.Store())],
                                    value=ast.Await(value=ast.Name(id="ret", ctx=ast.Load())),
                                    lineno=lineno + 15,
                                ),
                            ],
                            orelse=[
                                ast.Assign(
                                    targets=[ast.Name(id="return_value", ctx=ast.Store())],
                                    value=ast.Name(id="ret", ctx=ast.Load()),
                                    lineno=lineno + 16,
                                )
                            ],
                            lineno=lineno + 13,
                        ),
                    ]
                    if is_async
                    else [
                        ast.Assign(
                            targets=[ast.Name(id="return_value", ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id="wrapped", ctx=ast.Load()),
                                args=[ast.Starred(value=ast.Name(id="args", ctx=ast.Load()), ctx=ast.Load())],
                                keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))],
                            ),
                            lineno=lineno + 12,
                        )
                    ]
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
                                    ast.Name(id="test_module_name", ctx=ast.Load()),
                                    ast.Name(id="test_class_name", ctx=ast.Load()),
                                    ast.Name(id="test_name", ctx=ast.Load()),
                                    ast.Name(id="function_name", ctx=ast.Load()),
                                    ast.Name(id="loop_index", ctx=ast.Load()),
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
    func_def = ast.FunctionDef(
        name="codeflash_wrap",
        args=ast.arguments(
            args=[
                ast.arg(arg="wrapped", annotation=None),
                ast.arg(arg="test_module_name", annotation=None),
                ast.arg(arg="test_class_name", annotation=None),
                ast.arg(arg="test_name", annotation=None),
                ast.arg(arg="function_name", annotation=None),
                ast.arg(arg="line_id", annotation=None),
                ast.arg(arg="loop_index", annotation=None),
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
    if is_async:
        return ast.AsyncFunctionDef(
            name="codeflash_wrap",
            args=func_def.args,
            body=func_def.body,
            lineno=func_def.lineno,
            decorator_list=func_def.decorator_list,
            returns=func_def.returns,
        )
    return func_def
