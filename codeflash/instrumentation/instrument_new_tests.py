import ast
from typing import Optional

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class InjectPerfAndLogging(ast.NodeTransformer):
    def __init__(
        self,
        function: FunctionToOptimize,
        auxiliary_function_names,
        test_module_path: str,
        test_framework="pytest",
        test_timeout: int = 15,
    ):
        self.function_object = function
        self.class_name = None
        self.only_function_name = function.function_name
        self.test_framework = test_framework
        self.individual_test_timeout = test_timeout
        self.test_module_path = test_module_path
        if len(function.parents) == 1 and function.parents[0].type == "ClassDef":
            self.class_name = function.top_level_parent_name
        self.auxiliary_function_names = (
            auxiliary_function_names  # Other functional dependencies that were injected
        )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if any([name.name == self.only_function_name for name in node.names]):
            return None  # Remove the import of the function the test generation code
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        # If the original class exists during testing, then remove it all together
        if node.name == self.class_name:
            return None  # Remove the re-definition of the class and its dependencies from the test generation code
        for inner_node in ast.walk(node):
            if isinstance(inner_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner_node = self.visit_FunctionDef(inner_node, node.name)
        return node

    def visit_Assert(self, node: ast.Assert):
        # TODO : This does not work yet
        # Remove the assert statements from the test generation code
        for test_node in ast.walk(node):
            if self.is_target_function_node(test_node):
                return node
        return None

    def is_target_function_node(self, node):
        return isinstance(node, ast.Call) and (
            (hasattr(node.func, "id") and node.func.id == self.only_function_name)
            or (hasattr(node.func, "attr") and node.func.attr == self.only_function_name)
        )

    def update_line_node(self, test_node, node_name, index: str, class_name: Optional[str] = None):
        return [
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
            ast.Assign(
                targets=[ast.Name(id="return_value", ctx=ast.Store())],
                value=test_node,
                lineno=test_node.lineno + 2,
                col_offset=test_node.col_offset,
            ),
            ast.Assign(
                targets=[ast.Name(id="duration", ctx=ast.Store())],
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
                ast.Call(
                    func=ast.Name(id="_log__test__values", ctx=ast.Load()),
                    args=[
                        ast.Name(id="return_value", ctx=ast.Load()),
                        ast.Name(id="duration", ctx=ast.Load()),
                        ast.Constant(
                            value=f"{self.test_module_path}:{class_name or ''}{'.' if class_name is not None else ''}{node_name}:{self.only_function_name}:{index}"
                        ),
                    ],
                    keywords=[],
                ),
                lineno=test_node.lineno + 5,
                col_offset=test_node.col_offset,
            ),
        ]

    def visit_FunctionDef(self, node: ast.FunctionDef, class_name: Optional[str] = None):
        if node.name == self.only_function_name or node.name in self.auxiliary_function_names:
            return None  # Remove the re-definition of the function and its dependencies from the test generation code
        elif node.name.startswith("test_"):
            i = len(node.body) - 1
            while i >= 0:
                line_node = node.body[i]
                did_delete = False
                # TODO: Validate if the functional call actually did not raise any exceptions
                if isinstance(line_node, ast.Assert):
                    line_node = self.visit_Assert(line_node)
                    if line_node is None:
                        del node.body[i]
                        did_delete = True
                        i -= 1
                        continue

                if isinstance(line_node, ast.With):
                    j = len(line_node.body) - 1
                    while j >= 0:
                        with_line_node = line_node.body[j]
                        for with_node in ast.walk(with_line_node):
                            if self.is_target_function_node(with_node):
                                line_node.body[j : j + 1] = self.update_line_node(
                                    with_node, node.name, str(i) + "_" + str(j), class_name
                                )
                                did_delete = True
                                break
                        j -= 1
                else:
                    for test_node in ast.walk(line_node):
                        if self.is_target_function_node(test_node):
                            node.body[i : i + 1] = self.update_line_node(
                                test_node, node.name, str(i), class_name
                            )
                            did_delete = True
                            break
                # Remove any spare unittest asserts here
                if (
                    isinstance(line_node, ast.Expr)
                    and isinstance(line_node.value, ast.Call)
                    and isinstance(line_node.value.func, ast.Attribute)
                    and hasattr(line_node.value.func.value, "id")
                    and line_node.value.func.value.id == "self"
                    and not did_delete
                    and line_node.value.func.attr
                    in [
                        "assertEqual",
                        "assertNotEqual",
                        "assertTrue",
                        "assertFalse",
                        "assertIs",
                        "assertIsNot",
                        "assertIsNone",
                        "assertIsNotNone",
                        "assertIn",
                        "assertNotIn",
                        "assertIsInstance",
                        "assertNotIsInstance",
                        "assertRaises",
                        "assertRaisesRegex",
                        "assertWarns",
                        "assertWarnsRegex",
                        "assertLogs",
                        "assertNoLogs",
                        "assertAlmostEqual",
                        "assertNotAlmostEqual",
                        "assertGreater",
                        "assertGreaterEqual",
                        "assertLess",
                        "assertLessEqual",
                        "assertRegex",
                        "assertNotRegex",
                        "assertCountEqual",
                        "assertMultiLineEqual",
                        "assertSequenceEqual",
                        "assertListEqual",
                        "assertTupleEqual",
                        "assertSetEqual",
                        "assertDictEqual",
                    ]
                ):
                    del node.body[i]
                i -= 1
            if self.test_framework == "unittest":
                # TODO: Make sure that if the test times out, the test's time is excluded from the total time calculation and comparison
                node.decorator_list.append(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="timeout_decorator", ctx=ast.Load()),
                            attr="timeout",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=self.individual_test_timeout)],
                        keywords=[ast.keyword(arg="use_signals", value=ast.Constant(value=True))],
                    )
                )
                # TODO: The value of use_signals should be False, otherwise this will fail in multiprocessing environments
        return node


def inject_logging_code(test_code: str, tmp_dir: str = "/tmp") -> str:
    # TODO : Port this to Sqlite3 - makes it standard and easier to use
    logging_code = f"""import pickle
import os
def _log__test__values(values, duration, test_name):
    iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    with open(os.path.join('{tmp_dir}', f'test_return_values_{{iteration}}.bin'), 'ab') as f:
        return_bytes = pickle.dumps(values)
        _test_name = f"{{test_name}}".encode("ascii")
        f.write(len(_test_name).to_bytes(4, byteorder='big'))
        f.write(_test_name)
        f.write(duration.to_bytes(8, byteorder='big'))
        f.write(len(return_bytes).to_bytes(4, byteorder='big'))
        f.write(return_bytes)
"""
    return logging_code + test_code
