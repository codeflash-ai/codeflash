import sys
from pathlib import Path

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin

benchmarks_root = sys.argv[1]
tests_root = sys.argv[2]
trace_file = sys.argv[3]
# current working directory
project_root = Path.cwd()
if __name__ == "__main__":
    import pytest

    try:
        codeflash_benchmark_plugin.setup(trace_file, project_root)
        codeflash_trace.setup(trace_file)
        exitcode = pytest.main(
            [
                benchmarks_root,
                "--codeflash-trace",
                "-p",
                "no:benchmark",
                "-p",
                "no:codspeed",
                "-p",
                "no:cov",
                "-p",
                "no:profiling",
                "-s",
                "-o",
                "addopts=",
            ],
            plugins=[codeflash_benchmark_plugin],
        )  # Errors will be printed to stdout, not stderr

    except Exception as e:
        print(f"Failed to collect tests: {e!s}", file=sys.stderr)
        exitcode = -1
    sys.exit(exitcode)
