"""Subprocess entry point for memory profiling benchmarks via pytest-memray.

Runs pytest with --memray --native to profile peak memory per test function.
The codeflash-benchmark plugin is left active (without --codeflash-trace) so it
provides a no-op ``benchmark`` fixture for tests that depend on it.
"""

import sys
from pathlib import Path

benchmarks_root = sys.argv[1]
memray_bin_dir = sys.argv[2]
memray_bin_prefix = sys.argv[3]

if __name__ == "__main__":
    import pytest

    Path(memray_bin_dir).mkdir(parents=True, exist_ok=True)

    exitcode = pytest.main(
        [
            benchmarks_root,
            "--memray",
            "--native",
            f"--memray-bin-path={memray_bin_dir}",
            f"--memray-bin-prefix={memray_bin_prefix}",
            "--hide-memray-summary",
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
        ]
    )

    sys.exit(exitcode)
