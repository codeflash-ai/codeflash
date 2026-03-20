import sys
from pathlib import Path

from codeflash.benchmarking.plugin.unit_test_timing_plugin import UnitTestTimingPlugin

output_json = sys.argv[1]
test_files = sys.argv[2:]

if __name__ == "__main__":
    import pytest

    timing_plugin = UnitTestTimingPlugin(output_path=Path(output_json))
    try:
        exitcode = pytest.main(
            [*test_files, "-p", "no:benchmark", "-p", "no:codspeed", "-p", "no:cov", "-s", "-o", "addopts="],
            plugins=[timing_plugin],
        )
    except Exception as e:
        print(f"Failed to run timing tests: {e!s}", file=sys.stderr)
        exitcode = -1

    sys.exit(exitcode)
