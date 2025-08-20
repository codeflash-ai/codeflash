from fastmcp import FastMCP

mcp = FastMCP(
    name="codeflash",
    instructions="""
        This server provides code optimization tools.
        Call optimize_code(file, function) to optimize your code.
    """,
)


@mcp.tool
def optimize_code(file: str, function: str) -> dict[str, str]:
    # config = TestConfig(
    #     file_path=pathlib.Path(file),
    #     function_name=function,
    #     test_framework="pytest",
    #     min_improvement_x=1.0,
    #     coverage_expectations=[
    #         CoverageExpectation(
    #             function_name=function, expected_coverage=100.0, expected_lines=[2, 3, 4, 5, 6, 7, 8, 9, 10]
    #         )
    #     ],
    # )
    # cwd = pathlib.Path("/Users/aseemsaxena/Downloads/codeflash_dev/codeflash/code_to_optimize")  # TODO remove it
    print(file, function)
    return {
        "code": """def sorter(arr):
    print("codeflash stdout: Sorting list")
    arr.sort()
    print(f"result: {arr}")
    return arr
""",
        "explanation": "A faster version of the code is using python's in-built timsort function",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
