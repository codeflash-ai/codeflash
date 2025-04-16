from argparse import Namespace
from pathlib import Path

from codeflash.context.code_context_extractor import get_code_optimization_context
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
from codeflash.optimization.optimizer import Optimizer


def test_benchmark_extract(benchmark)->None:
    file_path = Path(__file__).parent.parent.parent.resolve() / "codeflash"
    opt = Optimizer(
        Namespace(
            project_root=file_path.resolve(),
            disable_telemetry=True,
            tests_root=(file_path / "tests").resolve(),
            test_framework="pytest",
            pytest_cmd="pytest",
            experiment_id=None,
            test_project_root=Path.cwd(),
        )
    )
    function_to_optimize = FunctionToOptimize(
        function_name="replace_function_and_helpers_with_optimized_code",
        file_path=file_path / "optimization" / "function_optimizer.py",
        parents=[FunctionParent(name="FunctionOptimizer", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    benchmark(get_code_optimization_context,function_to_optimize, opt.args.project_root)
