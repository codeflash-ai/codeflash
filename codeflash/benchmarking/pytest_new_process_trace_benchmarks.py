import sys
from plugin.plugin import CodeFlashPlugin
from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.code_utils.code_utils import get_run_tmp_file

benchmarks_root = sys.argv[1]
output_file = sys.argv[2]
if __name__ == "__main__":
    import pytest

    try:
        exitcode = pytest.main(
            [benchmarks_root, "--codeflash-trace", "-p", "no:benchmark", "-s", "-o", "addopts="], plugins=[CodeFlashPlugin()]
        )
        codeflash_trace.write_to_db(output_file)
        codeflash_trace.print_codeflash_db()
    except Exception as e:
        print(f"Failed to collect tests: {e!s}")
        exitcode = -1