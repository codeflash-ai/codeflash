from __future__ import annotations

import ast
import logging

from codeflash_python.models.models import TestingMode, VerificationType
from codeflash_python.verification.device_sync import (
    create_device_sync_precompute_statements,
    create_device_sync_statements,
)

logger = logging.getLogger("codeflash_python")


def create_wrapper_function(
    mode: TestingMode = TestingMode.BEHAVIOR, used_frameworks: dict[str, str] | None = None
) -> ast.FunctionDef:
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
        # Pre-compute device sync conditions before profiling to avoid overhead during timing
        *create_device_sync_precompute_statements(used_frameworks),
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
                # Pre-sync: synchronize device before starting timer
                *create_device_sync_statements(used_frameworks, for_return_value=False),
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
                # Post-sync: synchronize device after function call to ensure all device work is complete
                *create_device_sync_statements(used_frameworks, for_return_value=True),
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
