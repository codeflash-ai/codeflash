import sys
from plugin.plugin import CodeFlashPlugin

benchmarks_root = sys.argv[1]
function_list = sys.argv[2]
if __name__ == "__main__":
    import pytest

    try:
        exitcode = pytest.main(
            [benchmarks_root, "--benchmarks-root", benchmarks_root, "--codeflash-trace", "-p", "no:benchmark", "-s", "--functions", function_list,"-o", "addopts="], plugins=[CodeFlashPlugin()]
        )
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1