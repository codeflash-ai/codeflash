import sys
from plugin.plugin import CodeFlashPlugin

benchmarks_root = sys.argv[1]
if __name__ == "__main__":
    import pytest

    try:
        exitcode = pytest.main(
            [benchmarks_root, "--codeflash-trace", "-p", "no:benchmark", "-s", "-o", "addopts="], plugins=[CodeFlashPlugin()]
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1