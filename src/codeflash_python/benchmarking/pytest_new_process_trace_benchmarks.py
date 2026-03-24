import logging
import sys
from pathlib import Path

from codeflash_python.benchmarking.codeflash_trace import codeflash_trace
from codeflash_python.benchmarking.plugin.plugin import codeflash_benchmark_plugin

logger = logging.getLogger("codeflash_python")

benchmarks_root = sys.argv[1]
tests_root = sys.argv[2]
trace_file = sys.argv[3]
project_root = Path.cwd()

if __name__ == "__main__":
    import pytest

    orig_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(orig_recursion_limit * 2)

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
                "-p",
                "no:codeflash-benchmark",
                "-s",
                "-o",
                "addopts=",
            ],
            plugins=[codeflash_benchmark_plugin],
        )
    except Exception as e:
        logger.warning("Failed to collect tests: %s", e)
        exitcode = -1
    finally:
        sys.setrecursionlimit(orig_recursion_limit)

    sys.exit(exitcode)
