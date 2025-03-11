from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from pathlib import Path

def test_trace_benchmarks():
    # Test the trace_benchmarks function
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks"
    trace_benchmarks_pytest(benchmarks_root, project_root)