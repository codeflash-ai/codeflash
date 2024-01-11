import ast
import sys
from _ast import ClassDef
from typing import Any, Optional

from codeflash.code_utils.code_utils import module_name_from_file_path, get_run_tmp_file

if sys.version_info < (3, 9, 0):
    from astunparse import unparse as ast_unparse


class ReplaceCallNodeWithName(ast.NodeTransformer):
    def __init__(self, only_function_name, new_variable_name="return_value"):
        self.only_function_name = only_function_name
        self.new_variable_name = new_variable_name

    def visit_Call(self, node: ast.Call):
        if isinstance(node, ast.Call) and (
            (hasattr(node.func, "id") and node.func.id == self.only_function_name)
            or (hasattr(node.func, "attr") and node.func.attr == self.only_function_name)
        ):
            return ast.Name(id=self.new_variable_name, ctx=ast.Load())
        self.generic_visit(node)
        return node


class InjectPerfOnly(ast.NodeTransformer):
    def __init__(self, function_name, module_path):
        self.only_function_name = function_name
        self.module_path = module_path

    def update_line_node(
        self, test_node, node_name, index: str, test_class_name: Optional[str] = None
    ):
        call_node = None
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call) and (
                (hasattr(node.func, "id") and node.func.id == self.only_function_name)
                or (hasattr(node.func, "attr") and node.func.attr == self.only_function_name)
            ):
                call_node = node
        if call_node is None:
            return [test_node]
        updated_nodes = [
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
                lineno=test_node.lineno,
                col_offset=test_node.col_offset,
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
                lineno=test_node.lineno + 1,
                col_offset=test_node.col_offset,
            ),
            # TODO : This assign has a namespace clash with the rest of the function body
            #  if it has a variable called return_value
            ast.Assign(
                targets=[ast.Name(id="return_value", ctx=ast.Store())],
                value=call_node,
                lineno=test_node.lineno + 2,
                col_offset=test_node.col_offset,
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
                lineno=test_node.lineno + 3,
                col_offset=test_node.col_offset,
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
                lineno=test_node.lineno + 4,
                col_offset=test_node.col_offset,
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
                                ast.Constant(value=self.module_path),
                                ast.Constant(value=test_class_name or None),
                                ast.Constant(value=node_name),
                                ast.Constant(value=self.only_function_name),
                                ast.Constant(value=index),
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
                lineno=test_node.lineno + 5,
                col_offset=test_node.col_offset,
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
                lineno=test_node.lineno + 6,
                col_offset=test_node.col_offset,
            ),
            # TODO: Remove this print statement as it has been supplanted by the sqlite logging
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[
                        ast.JoinedStr(
                            values=[
                                ast.Constant(value="#####"),
                                ast.Constant(
                                    value=f"{self.module_path}:{test_class_name or ''}{'.' if test_class_name is not None else ''}{node_name}:{self.only_function_name}:{index}"
                                ),
                                ast.Constant(value="#####"),
                                ast.FormattedValue(
                                    value=ast.Name(id="codeflash_duration", ctx=ast.Load()),
                                    conversion=-1,
                                ),
                                ast.Constant(value="^^^^^"),
                            ]
                        )
                    ],
                    keywords=[],
                ),
                lineno=test_node.lineno + 7,
                col_offset=test_node.col_offset,
            ),
        ]
        subbed_node = ReplaceCallNodeWithName(self.only_function_name).visit(test_node)

        # TODO: Not just run the tests and ensure that the tests pass but also test the return value and compare that
        #  for equality amongst the original and the optimized version. This will ensure that the optimizations are correct
        #  in a more robust way.

        updated_nodes.append(subbed_node)
        return updated_nodes

    def is_target_function_line(self, line_node):
        for node in ast.walk(line_node):
            if isinstance(node, ast.Call) and (
                (hasattr(node.func, "id") and node.func.id == self.only_function_name)
                or (hasattr(node.func, "attr") and node.func.attr == self.only_function_name)
            ):
                return True
        return False

    def visit_ClassDef(self, node: ClassDef) -> Any:
        # TODO: Ensure that this class inherits from unittest.TestCase. Don't modify non unittest.TestCase classes
        for inner_node in ast.walk(node):
            if isinstance(inner_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner_node = self.visit_FunctionDef(inner_node, node.name)

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, test_class_name: Optional[str] = None):
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
                                            value=f"{get_run_tmp_file('test_return_values_')}"
                                        ),
                                        ast.FormattedValue(
                                            value=ast.Name(
                                                id="codeflash_iteration", ctx=ast.Load()
                                            ),
                                            conversion=-1,
                                        ),
                                        ast.Constant(value=".sqlite"),
                                    ]
                                )
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
                                    " iteration_id TEXT, runtime INTEGER, return_value BLOB)"
                                )
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
                        )
                    )
                ]
            )
            i = len(node.body) - 1
            while i >= 0:
                line_node = node.body[i]
                # TODO: Validate if the functional call actually did not raise any exceptions

                if isinstance(line_node, (ast.With, ast.For)):
                    j = len(line_node.body) - 1
                    while j >= 0:
                        compound_line_node = line_node.body[j]
                        for internal_node in ast.walk(compound_line_node):
                            if self.is_target_function_line(internal_node):
                                line_node.body[j : j + 1] = self.update_line_node(
                                    internal_node, node.name, str(i) + "_" + str(j), test_class_name
                                )
                                break
                        j -= 1
                else:
                    if self.is_target_function_line(line_node):
                        node.body[i : i + 1] = self.update_line_node(
                            line_node, node.name, str(i), test_class_name
                        )
                i -= 1
        return node


class FunctionImportedAsVisitor(ast.NodeVisitor):
    """This checks if a function has been imported as an alias. We only care about the alias then.
    from numpy import array as np_array
    np_array is what we want"""

    def __init__(self, original_function_name):
        self.original_function_name = original_function_name
        self.imported_as_function_name = original_function_name

    # TODO: Validate if the function imported is actually from the right module
    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            if alias.name == self.original_function_name:
                if hasattr(alias, "asname") and not alias.asname is None:
                    self.imported_as_function_name = alias.asname


def inject_profiling_into_existing_test(test_path, function_name, root_path):
    with open(test_path, "r") as f:
        test_code = f.read()
    tree = ast.parse(test_code)
    # TODO: Pass the full name of function here, otherwise we can run into namespace clashes
    import_visitor = FunctionImportedAsVisitor(function_name)
    import_visitor.visit(tree)
    function_name = import_visitor.imported_as_function_name
    module_path = module_name_from_file_path(test_path, root_path)
    tree = InjectPerfOnly(function_name, module_path).visit(tree)
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
        ast.Import(names=[ast.alias(name="sqlite3")]),
        ast.Import(names=[ast.alias(name="pickle")]),
    ]
    tree.body = new_imports + tree.body

    if sys.version_info < (3, 9, 0):
        return ast_unparse(tree)
    else:
        return ast.unparse(tree)
