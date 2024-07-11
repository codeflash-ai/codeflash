from __future__ import annotations

import ast
import logging
from typing import Iterable

import isort

from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path


class ReplaceCallNodeWithName(ast.NodeTransformer):
    def __init__(self, only_function_name: str, new_variable_name: str = "codeflash_return_value") -> None:
        self.only_function_name = only_function_name
        self.new_variable_name = new_variable_name

    def visit_Call(self, node: ast.Call) -> ast.Name | ast.Call:
        if isinstance(node, ast.Call):
            function_name: str = ""
            if hasattr(node.func, "id"):
                function_name = node.func.id

            if hasattr(node.func, "attr"):
                function_name = node.func.attr

            if hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                function_name = node.func.value.id + "." + function_name

            if function_name == self.only_function_name:
                return ast.Name(id=self.new_variable_name, ctx=ast.Load())

        self.generic_visit(node)
        return node


class InjectPerfOnly(ast.NodeTransformer):
    def __init__(
        self, function_name: str, module_path: str, test_framework: str
    ) -> None:
        self.only_function_name = function_name
        self.module_path = module_path
        self.test_framework = test_framework

    def update_line_node(
        self,
            test_node: ast.stmt,
        node_name: str,
        index: str,
        test_class_name: str | None = None,
    ) -> Iterable[ast.stmt]:
        call_node = None
        function_name: str = ""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                if hasattr(node.func, "id"):
                    function_name = node.func.id

                if hasattr(node.func, "attr"):
                    function_name = node.func.attr

                if hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                    function_name = node.func.value.id + "." + function_name

                if function_name == self.only_function_name:
                    call_node = node
                    break

        if call_node is None:
            return [test_node]

        updated_nodes = [
            ast.Assign(
                targets=[ast.Name(id="codeflash_return_value", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                    args=[
                        ast.Name(id=function_name, ctx=ast.Load()),
                        ast.Constant(value=self.module_path),
                        ast.Constant(value=test_class_name or None),
                        ast.Constant(value=node_name),
                        ast.Constant(value=self.only_function_name),
                        ast.Constant(value=index),
                        ast.Name(id="codeflash_cur", ctx=ast.Load()),
                        ast.Name(id="codeflash_con", ctx=ast.Load()),
                        *call_node.args,
                        *call_node.keywords,
                    ],
                    keywords=[],
                ),
                lineno=test_node.lineno,
                col_offset=test_node.col_offset,
            ),
        ]
        subbed_node = ReplaceCallNodeWithName(self.only_function_name).visit(test_node)

        # TODO: Not just run the tests and ensure that the tests pass but also test the return value and
        #  compare that for equality amongst the original and the optimized version. This will ensure that the
        #  optimizations are correct in a more robust way.

        updated_nodes.append(subbed_node)
        return updated_nodes

    def is_target_function_line(self, line_node: ast.AST) -> bool:
        node: ast.AST
        for node in ast.walk(line_node):
            if isinstance(node, ast.Call):
                function_name: str = ""
                if hasattr(node.func, "id"):
                    function_name = node.func.id

                if hasattr(node.func, "attr"):
                    function_name = node.func.attr

                if hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                    function_name = node.func.value.id + "." + function_name

                if function_name == self.only_function_name:
                    return True

        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # TODO: Ensure that this class inherits from unittest.TestCase. Don't modify non unittest.TestCase classes.
        for inner_node in ast.walk(node):
            if isinstance(inner_node, ast.FunctionDef):
                self.visit_FunctionDef(inner_node, node.name)

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, test_class_name: str | None = None) -> ast.FunctionDef:
        if node.name.startswith("test_"):
            node.body = (
                [
                    ast.Assign(
                        targets=[ast.Name(id="codeflash_iteration", ctx=ast.Store())],
                        value=ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id="os", ctx=ast.Load()),
                                attr="environ",
                                ctx=ast.Load(),
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
                                value=ast.Name(id="sqlite3", ctx=ast.Load()),
                                attr="connect",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.JoinedStr(
                                    values=[
                                        ast.Constant(
                                            value=f"{get_run_tmp_file('test_return_values_')}",
                                        ),
                                        ast.FormattedValue(
                                            value=ast.Name(
                                                id="codeflash_iteration",
                                                ctx=ast.Load(),
                                            ),
                                            conversion=-1,
                                        ),
                                        ast.Constant(value=".sqlite"),
                                    ],
                                ),
                            ],
                            keywords=[],
                        ),
                        lineno=node.lineno + 2,
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
                        lineno=node.lineno + 3,
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
                                    " iteration_id TEXT, runtime INTEGER, return_value BLOB)",
                                ),
                            ],
                            keywords=[],
                        ),
                        lineno=node.lineno + 4,
                        col_offset=node.col_offset,
                    ),
                ]
                + node.body
                + [
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="codeflash_con", ctx=ast.Load()),
                                attr="close",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                    ),
                ]
            )
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
                            if isinstance(internal_node, ast.stmt) and self.is_target_function_line(internal_node):
                                line_node.body[j : j + 1] = self.update_line_node(
                                    internal_node,
                                    node.name,
                                    str(i) + "_" + str(j),
                                    test_class_name,
                                )
                                break
                        j -= 1
                elif self.is_target_function_line(line_node):
                    node.body[i : i + 1] = self.update_line_node(
                        line_node,
                        node.name,
                        str(i),
                        test_class_name,
                    )
                i -= 1
        return node


class FunctionImportedAsVisitor(ast.NodeVisitor):
    """This checks if a function has been imported as an alias. We only care about the alias then.
    from numpy import array as np_array
    np_array is what we want
    """

    def __init__(self, qualified_name: str) -> None:
        self.split_qualified_name = qualified_name.split(".")
        assert len(self.split_qualified_name) <= 2, "Only support functions in the format module.function"
        self.imported_as = qualified_name
        self.to_match = self.split_qualified_name[0]
        try:
            self.optional_method_name: str | None = self.split_qualified_name[1]
        except IndexError:
            self.optional_method_name = None

    # TODO: Validate if the function imported is actually from the right module
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == self.to_match and hasattr(alias, "asname") and alias.asname is not None:
                self.imported_as = alias.asname + (
                    "." + self.optional_method_name if self.optional_method_name else ""
                )


def inject_profiling_into_existing_test(
    test_path: str,
    function_name: str,
    root_path: str,
    test_framework: str,
) -> tuple[bool, str | None]:
    with open(test_path, encoding="utf8") as f:
        test_code = f.read()
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logging.exception("Syntax error in code")
        return False, None
    # TODO: Pass the full name of function here, otherwise we can run into namespace clashes
    module_path = module_name_from_file_path(test_path, root_path)
    import_visitor = FunctionImportedAsVisitor(function_name)
    import_visitor.visit(tree)
    function_name = import_visitor.imported_as

    tree = InjectPerfOnly(function_name, module_path, test_framework).visit(tree)
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
        ast.Import(names=[ast.alias(name="sqlite3")]),
        ast.Import(names=[ast.alias(name="dill", asname="pickle")]),
        ast.Import(names=[ast.alias(name="timeout_decorator")]),
    ]
    tree.body = [*new_imports, create_wrapper_function(), *tree.body]
    return True, isort.code(ast.unparse(tree), float_to_top=True)


def create_wrapper_function() -> ast.FunctionDef:
    lineno = 1
    return ast.FunctionDef(
        name="codeflash_wrap",
        args=ast.arguments(
            args=[
                ast.arg(arg="wrapped", annotation=None),
                ast.arg(arg="test_module_name", annotation=None),
                ast.arg(arg="test_class_name", annotation=None),
                ast.arg(arg="test_name", annotation=None),
                ast.arg(arg="function_name", annotation=None),
                ast.arg(arg="line_id", annotation=None),
                ast.arg(arg="codeflash_cur", annotation=None),
                ast.arg(arg="codeflash_con", annotation=None),
            ],
            vararg=ast.arg(arg="args"),
            kwarg=ast.arg(arg="kwargs"),
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="test_id", ctx=ast.Store())],
                value=ast.JoinedStr(
                    values=[
                        ast.FormattedValue(
                            value=ast.Name(id="test_module_name", ctx=ast.Load()),
                            conversion=-1,
                        ),
                        ast.Constant(value=":"),
                        ast.FormattedValue(
                            value=ast.Name(id="test_class_name", ctx=ast.Load()),
                            conversion=-1,
                        ),
                        ast.Constant(value=":"),
                        ast.FormattedValue(
                            value=ast.Name(id="test_name", ctx=ast.Load()),
                            conversion=-1,
                        ),
                        ast.Constant(value=":"),
                        ast.FormattedValue(
                            value=ast.Name(id="line_id", ctx=ast.Load()),
                            conversion=-1,
                        ),
                    ],
                ),
                lineno=lineno + 1,
            ),
            ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Call(
                        func=ast.Name(id="hasattr", ctx=ast.Load()),
                        args=[
                            ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                            ast.Constant(value="index"),
                        ],
                        keywords=[],
                    ),
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                                attr="index",
                                ctx=ast.Store(),
                            ),
                        ],
                        value=ast.Dict(keys=[], values=[]),
                        lineno=lineno + 3,
                    ),
                ],
                orelse=[],
                lineno=lineno + 2,
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="test_id", ctx=ast.Load()),
                    ops=[ast.In()],
                    comparators=[
                        ast.Attribute(
                            value=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                            attr="index",
                            ctx=ast.Load(),
                        ),
                    ],
                ),
                body=[
                    ast.AugAssign(
                        target=ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                                attr="index",
                                ctx=ast.Load(),
                            ),
                            slice=ast.Name(id="test_id", ctx=ast.Load()),
                            ctx=ast.Store(),
                        ),
                        op=ast.Add(),
                        value=ast.Constant(value=1),
                        lineno=lineno + 5,
                    ),
                ],
                orelse=[
                    ast.Assign(
                        targets=[
                            ast.Subscript(
                                value=ast.Attribute(
                                    value=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                                    attr="index",
                                    ctx=ast.Load(),
                                ),
                                slice=ast.Name(id="test_id", ctx=ast.Load()),
                                ctx=ast.Store(),
                            ),
                        ],
                        value=ast.Constant(value=0),
                        lineno=lineno + 6,
                    ),
                ],
                lineno=lineno + 4,
            ),
            ast.Assign(
                targets=[
                    ast.Name(id="codeflash_test_index", ctx=ast.Store()),
                ],
                value=ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id="codeflash_wrap", ctx=ast.Load()),
                        attr="index",
                        ctx=ast.Load(),
                    ),
                    slice=ast.Name(id="test_id", ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                lineno=lineno + 7,
            ),
            ast.Assign(
                targets=[ast.Name(id="invocation_id", ctx=ast.Store())],
                value=ast.JoinedStr(
                    values=[
                        ast.FormattedValue(
                            value=ast.Name(id="line_id", ctx=ast.Load()),
                            conversion=-1,
                        ),
                        ast.Constant(value="_"),
                        ast.FormattedValue(
                            value=ast.Name(id="codeflash_test_index", ctx=ast.Load()),
                            conversion=-1,
                        ),
                    ],
                ),
                lineno=lineno + 8,
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[
                        ast.JoinedStr(
                            values=[
                                ast.Constant(value="!######"),
                                ast.FormattedValue(
                                    value=ast.Name(id="test_module_name", ctx=ast.Load()),
                                    conversion=-1,
                                ),
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
                                ast.FormattedValue(
                                    value=ast.Name(id="test_name", ctx=ast.Load()),
                                    conversion=-1,
                                ),
                                ast.Constant(value=":"),
                                ast.FormattedValue(
                                    value=ast.Name(id="function_name", ctx=ast.Load()),
                                    conversion=-1,
                                ),
                                ast.Constant(value=":"),
                                ast.FormattedValue(
                                    value=ast.Name(id="invocation_id", ctx=ast.Load()),
                                    conversion=-1,
                                ),
                                ast.Constant(value="######!"),
                            ],
                        ),
                    ],
                    keywords=[],
                ),
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="gc", ctx=ast.Load()),
                        attr="disable",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
                lineno=lineno + 9,
            ),
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
                lineno=lineno + 10,
            ),
            ast.Assign(
                targets=[ast.Name(id="return_value", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="wrapped", ctx=ast.Load()),
                    args=[ast.Starred(value=ast.Name(id="args", ctx=ast.Load()), ctx=ast.Load())],
                    keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))],
                ),
                lineno=lineno + 11,
            ),
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
                lineno=lineno + 12,
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="gc", ctx=ast.Load()),
                        attr="enable",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
                lineno=lineno + 13,
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="codeflash_cur", ctx=ast.Load()),
                        attr="execute",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value="INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?)"),
                        ast.Tuple(
                            elts=[
                                ast.Name(id="test_module_name", ctx=ast.Load()),
                                ast.Name(id="test_class_name", ctx=ast.Load()),
                                ast.Name(id="test_name", ctx=ast.Load()),
                                ast.Name(id="function_name", ctx=ast.Load()),
                                ast.Name(id="invocation_id", ctx=ast.Load()),
                                ast.Name(id="codeflash_duration", ctx=ast.Load()),
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="pickle", ctx=ast.Load()),
                                        attr="dumps",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Name(id="return_value", ctx=ast.Load())],
                                    keywords=[],
                                ),
                            ],
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                ),
                lineno=lineno + 14,
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="codeflash_con", ctx=ast.Load()),
                        attr="commit",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
                lineno=lineno + 15,
            ),
            ast.Return(value=ast.Name(id="return_value", ctx=ast.Load()), lineno=lineno + 16),
        ],
        lineno=lineno,
        decorator_list=[],
        returns=None,
        type_params=[],
    )
