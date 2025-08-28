from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastmcp import FastMCP

from codeflash.lsp.server import CodeflashLanguageServer
from codeflash.lsp.beta import perform_function_optimization, FunctionOptimizationParams, \
    initialize_function_optimization
from tests.scripts.end_to_end_test_utilities import TestConfig, run_codeflash_command
from lsprotocol import types


# Define lifespan context manager
@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncGenerator[None, Any]:
    print("Starting up...")
    print(mcp.name)
    # Do startup work here (connect to DB, initialize cache, etc.)
    server = CodeflashLanguageServer(name = "codeflash", version = "0.0.1")
    config_file = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/pyproject.toml")
    file = "/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py"
    function = "sorter"
    params = FunctionOptimizationParams(functionName=function, textDocument=types.TextDocumentIdentifier(Path(file).as_uri()))
    server.prepare_optimizer_arguments(config_file)
    initialize_function_optimization(server, params)
    perform_function_optimization(server, params)
    #optimize_code(file, function)
    yield
    # Cleanup work after shutdown
    print("Shutting down...")
    server.cleanup_the_optimizer()
    server.shutdown()

mcp = FastMCP(
    name="codeflash",
    instructions="""
        This server provides code optimization tools.
        Call optimize_code(file, function) to optimize your code.
    """,
    lifespan=lifespan,
)


#@mcp.tool
def optimize_code(file: str, function: str) -> str:
    # TODO ask for pr or no pr if successful
    config = TestConfig(file_path=Path(f"{file}"), function_name=f"{function}", test_framework="pytest")
    cwd = Path(file).resolve().parent
    status = run_codeflash_command(cwd, config, expected_improvement_pct=5)
    if status:
        return "Optimization Successful, file has been edited"
    return "Codeflash run did not meet expected requirements for testing, reverting file changes."


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # Optimize my codebase, the file is "/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py" and the function is "sorter"

