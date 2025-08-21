from fastmcp import FastMCP
from pathlib import Path

from tests.scripts.end_to_end_test_utilities import TestConfig, run_codeflash_command

mcp = FastMCP(
    name="codeflash",
    instructions="""
        This server provides code optimization tools.
        Call optimize_code(file, function) to optimize your code.
    """,
)


@mcp.tool
def optimize_code(file: str, function: str) -> str:
    # TODO ask for pr or no pr if successful
    config = TestConfig(
            file_path=Path(f"{file}"),
            function_name=f"{function}",
            test_framework="pytest",
        )
    cwd = Path(file).resolve().parent
    status = run_codeflash_command(cwd, config, expected_improvement_pct=5)
    if status:
        return "Optimization Successful, file has been edited"
    else:
        return "Codeflash run did not meet expected requirements for testing, reverting file changes."


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # Optimize my codebase, the file is "/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py" and the function is "sorter"