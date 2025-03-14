from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file
import shutil

def test_trace_benchmarks():
    # Test the trace_benchmarks function
    project_root = Path(__file__).parent.parent / "code_to_optimize"
    benchmarks_root = project_root / "tests" / "pytest" / "benchmarks"
    # make directory in project_root / "tests"


    tests_root = project_root / "tests" / "test_trace_benchmarks"
    tests_root.mkdir(parents=False, exist_ok=False)
    output_file = (tests_root / Path("test_trace_benchmarks.trace")).resolve()
    trace_benchmarks_pytest(benchmarks_root, tests_root, project_root, output_file)
    assert output_file.exists()

    test1_path = tests_root / Path("test_benchmark_bubble_sort_py_test_sort__replay_test_0.py")
    assert test1_path.exists()

    # test1_code = """"""
    # assert test1_path.read_text("utf-8").strip()==test1_code.strip()
    # cleanup
    # shutil.rmtree(tests_root)
    # output_file.unlink()