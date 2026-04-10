from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context
from codeflash.models.models import FunctionParent


def test_benchmark_extract(benchmark) -> None:
    project_root = Path(__file__).parent.parent.parent.resolve() / "codeflash"
    function_to_optimize = FunctionToOptimize(
        function_name="replace_function_and_helpers_with_optimized_code",
        file_path=project_root / "languages" / "function_optimizer.py",
        parents=[FunctionParent(name="FunctionOptimizer", type="ClassDef")],
        starting_line=None,
        ending_line=None,
    )

    benchmark(get_code_optimization_context, function_to_optimize, project_root)
