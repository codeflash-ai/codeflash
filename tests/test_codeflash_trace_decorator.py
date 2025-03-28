from codeflash.benchmarking.codeflash_trace import codeflash_trace
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file

@codeflash_trace
def example_function(arr):
    arr.sort()
    return arr


def test_codeflash_trace_decorator():
    arr = [3, 1, 2]
    result = example_function(arr)
    # cleanup test trace file using Path
    assert result == [1, 2, 3]
