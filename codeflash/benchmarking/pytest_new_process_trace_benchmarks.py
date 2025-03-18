import sys
from pathlib import Path

from codeflash.benchmarking.benchmark_database_utils import BenchmarkDatabaseUtils
from codeflash.verification.verification_utils import get_test_file_path
from plugin.plugin import CodeFlashBenchmarkPlugin
from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.code_utils.code_utils import get_run_tmp_file

benchmarks_root = sys.argv[1]
tests_root = sys.argv[2]
trace_file = sys.argv[3]
# current working directory
project_root = Path.cwd()
if __name__ == "__main__":
    import pytest

    try:
        db = BenchmarkDatabaseUtils(trace_path=Path(trace_file))
        db.setup()
        exitcode = pytest.main(
            [benchmarks_root, "--codeflash-trace", "-p", "no:benchmark", "-s", "-o", "addopts="], plugins=[CodeFlashBenchmarkPlugin()]
        )
        db.write_function_timings(codeflash_trace.function_calls_data)
        db.write_benchmark_timings(CodeFlashBenchmarkPlugin.benchmark_timings)
        db.print_function_timings()
        db.print_benchmark_timings()
        db.close()

    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1