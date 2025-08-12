import os
import pathlib

from fastmcp import FastMCP

from tests.scripts.end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command

mcp = FastMCP("My MCP Server")

@mcp.tool
def optimize_code(file: str, function: str) -> bool: #todo add file and function name as arguments
    config = TestConfig(
        file_path=pathlib.Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py"),
        function_name="sorter",
        test_framework="pytest",
        min_improvement_x=1.0,
        coverage_expectations=[
            CoverageExpectation(
                function_name="sorter", expected_coverage=100.0, expected_lines=[2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
        ],
    )
    cwd = (pathlib.Path(__file__).parent/ "code_to_optimize").resolve()
    return run_codeflash_command(
        cwd, config, 100, ['print("codeflash stdout: Sorting list")', 'print(f"result: {arr}")']
    )

if __name__ == "__main__":
    mcp.run()