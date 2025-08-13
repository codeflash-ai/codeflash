import pathlib

from fastmcp import FastMCP

from tests.scripts.end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command

mcp = FastMCP(
    name="Code Optimization Assistant",
    instructions="""
        This server provides code optimization tools.
        Call optimize_code(file, function) to optimize your code.
    """,
)


@mcp.tool
def optimize_code(file: str, function: str) -> bool:
    config = TestConfig(
        file_path=pathlib.Path(file),
        function_name=function,
        test_framework="pytest",
        min_improvement_x=1.0,
        coverage_expectations=[
            CoverageExpectation(
                function_name=function, expected_coverage=100.0, expected_lines=[2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
        ],
    )
    cwd = (pathlib.Path(__file__).parent / "code_to_optimize").resolve()  # TODO remove it
    return run_codeflash_command(
        cwd, config, 100, ['print("codeflash stdout: Sorting list")', 'print(f"result: {arr}")']
    )


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
