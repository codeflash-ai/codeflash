import tempfile

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_context import (
    get_constrained_function_context_and_helper_functions,
)


def test_get_outside_method_helper() -> None:
    code = """def OptimizeMe(a, b, c):
    return HelperClass().helper_method(a, b, c)
    
Class HelperClass:
    def helper_method(self, a, b, c):
        return a + b + c
"""


code_to_optimize = """def OptimizeMe(a, b, c):
    return HelperClass().helper_method(a, b, c)
"""

with tempfile.NamedTemporaryFile("w") as f:
    f.write(code)
    f.flush()
    function_to_optimize = FunctionToOptimize("OptimizeMe",
                                              f.name, [])
    helper_code, helper_functions = get_constrained_function_context_and_helper_functions(function_to_optimize,
                                                                                          "/this/is/", code_to_optimize)
    assert helper_code == ""
    assert helper_functions == []
