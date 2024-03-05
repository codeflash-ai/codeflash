import pytest
from cli.code_to_optimize.sample4 import solve

@pytest.mark.parametrize("input_string, expected_result", [
    ("FC", "Yes"),
    ("FFCC", "Yes"),
    ("CCFF", "No"),
    ("CFCF", "Yes"),
    ("FFFF", "No"),
    ("CCCC", "No"),
    ("", "No"),
])
def test_solve(input_string, expected_result):
    assert solve(input_string) == expected_result
