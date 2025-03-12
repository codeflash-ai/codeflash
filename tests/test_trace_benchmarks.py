from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file

def test_trace_benchmarks():
    # Test the trace_benchmarks function
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks"
    output_file = Path("test_trace_benchmarks.trace").resolve()
    trace_benchmarks_pytest(benchmarks_root, project_root, output_file)
    assert output_file.exists()
    output_file.unlink()