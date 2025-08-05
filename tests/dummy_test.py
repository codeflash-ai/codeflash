import time
import pytest
from codeflash.dummy.op_features import get_all_optimization_features


def test_fetch_optimization_features():
    # print current time
    print(f"Current time: {time.time()}")
    results = get_all_optimization_features()
    
    # Assert we got something back (optional)
    assert isinstance(results, list)
    
    # Print results (for dev/debugging purposes)
    for row in results:
        print(row)