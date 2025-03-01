from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def go() -> None:
    file_path = Path("/Users/krrt7/Desktop/work/seaborn/seaborn/_marks/base.py")
    function_to_optimize = FunctionToOptimize(
        function_name="resolve_properties", file_path=str(file_path), parents=[], starting_line=None, ending_line=None
    )

    func_optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=TestConfig(
            tests_root=file_path,
            tests_project_rootdir=file_path.parent,
            project_root_path=file_path.parent,
            test_framework="pytest",
            pytest_cmd="pytest",
        ),
    )
    ctx_result = func_optimizer.get_code_optimization_context()
    code_context = ctx_result.unwrap()
    with Path(__file__).parent.joinpath("test_file.py").open("w") as f:
        f.write(code_context.code_to_optimize_with_helpers)


if __name__ == "__main__":
    go()
