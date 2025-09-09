import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastmcp import FastMCP
from lsprotocol.types import TextDocumentItem

from codeflash.lsp.server import CodeflashLanguageServer
from codeflash.lsp.beta import perform_function_optimization, FunctionOptimizationParams, \
    initialize_function_optimization, validate_project, discover_function_tests
from tests.scripts.end_to_end_test_utilities import TestConfig, run_codeflash_command
from lsprotocol import types

# dummy method for getting pyproject.toml path
def _find_pyproject_toml(workspace_path: str) -> Optional[Path]:
    workspace_path_obj = Path(workspace_path)
    for file_path in workspace_path_obj.rglob("pyproject.toml"):
        return file_path.resolve()
    return None


# Define lifespan context manager
@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncGenerator[None, Any]:
    print("Starting up...")
    print(mcp.name)
    # # Do startup work here (connect to DB, initialize cache, etc.)
    # server = CodeflashLanguageServer(name = "codeflash", version = "0.0.1")
    # config_file = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/pyproject.toml")
    # file = "/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py"
    # function = "sorter"
    # params = FunctionOptimizationParams(functionName=function, textDocument=types.TextDocumentIdentifier(Path(file).as_uri()))
    # server.prepare_optimizer_arguments(config_file)
    # initialize_function_optimization(server, params)
    # perform_function_optimization(server, params)
    # #optimize_code(file, function)

    #################### initialize the server #############################
    server = CodeflashLanguageServer("codeflash-language-server", "v1.0")
    # suppose the pyproject.toml is in the current directory
    server.prepare_optimizer_arguments(_find_pyproject_toml("."))
    result = validate_project(server, None)
    if result["status"] == "error":
        # handle if the project is not valid, it can be because pyproject.toml is not valid or the repository is in bare state or the repository has no commits, which will stop the worktree from working
        print(result["message"])
        sys.exit(1)

    #################### start the optimization for file, function #############################
    file_path = "/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py"
    function_name = "sorter"

    # This is not necessary, just for testing
    server.args.module_root = Path("/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize")
    result = initialize_function_optimization(server, FunctionOptimizationParams(
        functionName=function_name,
        textDocument=TextDocumentItem(
            uri=file_path,
            language_id="python",
            version=1,
            text=""
        )
    )
                                              )
    if result["status"] == "error":
        # handle if the function is not optimizable
        print(result["message"])
        sys.exit(1)

    discover_function_tests(server, FunctionOptimizationParams(functionName=function_name, textDocument=None))
    final_result = perform_function_optimization(server, FunctionOptimizationParams(functionName=function_name,
                                                                                    textDocument=None))
    if final_result["status"] == "success":
        print(final_result)
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

