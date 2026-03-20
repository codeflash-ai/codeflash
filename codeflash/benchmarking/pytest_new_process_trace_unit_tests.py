import sys
from pathlib import Path

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.benchmarking.plugin.unit_test_trace_plugin import unit_test_trace_plugin

tests_root = sys.argv[1]
trace_file = sys.argv[2]
project_root = Path.cwd()

if __name__ == "__main__":
    import pytest

    orig_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(orig_recursion_limit * 2)

    try:
        unit_test_trace_plugin.setup(trace_file, project_root)
        codeflash_trace.setup(trace_file)
        exitcode = pytest.main(
            [
                tests_root,
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
            plugins=[unit_test_trace_plugin],
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}", file=sys.stderr)
        exitcode = -1
    finally:
        sys.setrecursionlimit(orig_recursion_limit)

    sys.exit(exitcode)
