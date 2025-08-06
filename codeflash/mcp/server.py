#!/usr/bin/env python3
"""Codeflash MCP Server - Expose codeflash optimization capabilities to Claude Code.

This server provides comprehensive access to all codeflash optimization features
through the Model Context Protocol (MCP), making it available as a Claude Code subagent.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Error: Missing MCP dependencies. Install with: pip install codeflash[mcp]")
    print(f"Import error: {e}")
    sys.exit(1)

from codeflash.version import __version__
from codeflash.cli_cmds.cli import parse_args
from codeflash.cli_cmds.console import logger

# Initialize FastMCP server
mcp = FastMCP("codeflash")

# Pydantic models for structured responses
class OptimizationResult(BaseModel):
    """Result of a codeflash optimization operation."""
    success: bool = Field(description="Whether the optimization was successful")
    message: str = Field(description="Human-readable status message")
    optimized_functions: List[str] = Field(default=[], description="List of functions that were optimized")
    performance_improvement: Optional[str] = Field(default=None, description="Performance improvement summary")
    file_path: Optional[str] = Field(default=None, description="Path to the optimized file")
    errors: List[str] = Field(default=[], description="Any errors encountered during optimization")

class ProjectStatus(BaseModel):
    """Status of codeflash setup in a project."""
    is_initialized: bool = Field(description="Whether codeflash is initialized in the project")
    config_file: Optional[str] = Field(default=None, description="Path to codeflash config file")
    module_root: Optional[str] = Field(default=None, description="Project's Python module root")
    tests_root: Optional[str] = Field(default=None, description="Project's tests root")
    version: str = Field(description="Codeflash version")

class TraceResult(BaseModel):
    """Result of a codeflash trace operation."""
    success: bool = Field(description="Whether tracing was successful")
    trace_file: Optional[str] = Field(default=None, description="Path to generated trace file")
    functions_traced: int = Field(default=0, description="Number of functions traced")
    message: str = Field(description="Status message")

def _run_codeflash_command(args: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run a codeflash command and return structured results."""
    try:
        cmd = [sys.executable, "-m", "codeflash"] + args
        result = subprocess.run(
            cmd,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out after 5 minutes",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

@mcp.tool()
def get_codeflash_status(project_path: str = ".") -> ProjectStatus:
    """Get the current status of codeflash setup in a project.
    
    Args:
        project_path: Path to the project directory (default: current directory)
        
    Returns:
        ProjectStatus with current setup information
    """
    project_path = Path(project_path).resolve()
    
    # Check for pyproject.toml with codeflash config
    pyproject_file = project_path / "pyproject.toml"
    config_file = None
    module_root = None
    tests_root = None
    is_initialized = False
    
    if pyproject_file.exists():
        try:
            import tomlkit
            with open(pyproject_file, 'r') as f:
                config = tomlkit.parse(f.read())
            
            if "tool" in config and "codeflash" in config["tool"]:
                is_initialized = True
                config_file = str(pyproject_file)
                codeflash_config = config["tool"]["codeflash"]
                module_root = codeflash_config.get("module-root")
                tests_root = codeflash_config.get("tests-root")
        except Exception:
            pass
    
    return ProjectStatus(
        is_initialized=is_initialized,
        config_file=config_file,
        module_root=module_root,
        tests_root=tests_root,
        version=__version__
    )

@mcp.tool()
def optimize_function(
    function_name: str,
    file_path: Optional[str] = None,
    project_path: str = ".",
    create_pr: bool = True
) -> OptimizationResult:
    """Optimize a specific Python function using Codeflash AI optimization.
    
    Args:
        function_name: Name of the function to optimize
        file_path: Path to the file containing the function (optional if auto-discoverable)
        project_path: Path to the project root directory
        create_pr: Whether to create a pull request with the optimization
        
    Returns:
        OptimizationResult with optimization details and performance improvements
    """
    args = ["--function", function_name]
    
    if file_path:
        args.extend(["--file", file_path])
    
    if not create_pr:
        args.append("--no-pr")
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    # Parse the output to extract optimization details
    optimized_functions = []
    performance_improvement = None
    errors = []
    
    if result["success"]:
        # Extract optimization information from stdout
        stdout = result["stdout"]
        if "Optimized function" in stdout:
            optimized_functions = [function_name]
        if "performance improvement" in stdout.lower() or "speedup" in stdout.lower():
            lines = stdout.split('\n')
            for line in lines:
                if "speedup" in line.lower() or "faster" in line.lower():
                    performance_improvement = line.strip()
                    break
    else:
        if result["stderr"]:
            errors.append(result["stderr"])
    
    return OptimizationResult(
        success=result["success"],
        message=result["stdout"] if result["success"] else result["stderr"],
        optimized_functions=optimized_functions,
        performance_improvement=performance_improvement,
        file_path=file_path,
        errors=errors
    )

@mcp.tool()
def optimize_file(
    file_path: str,
    project_path: str = ".",
    create_pr: bool = True
) -> OptimizationResult:
    """Optimize all functions in a specific Python file.
    
    Args:
        file_path: Path to the Python file to optimize
        project_path: Path to the project root directory  
        create_pr: Whether to create a pull request with optimizations
        
    Returns:
        OptimizationResult with details of all optimizations made
    """
    args = ["--file", file_path]
    
    if not create_pr:
        args.append("--no-pr")
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    optimized_functions = []
    errors = []
    
    if result["success"]:
        # Extract function names from output
        stdout = result["stdout"]
        lines = stdout.split('\n')
        for line in lines:
            if "Optimized function" in line:
                # Extract function name from the line
                parts = line.split("Optimized function")
                if len(parts) > 1:
                    func_name = parts[1].strip().split()[0]
                    optimized_functions.append(func_name)
    else:
        if result["stderr"]:
            errors.append(result["stderr"])
    
    return OptimizationResult(
        success=result["success"],
        message=result["stdout"] if result["success"] else result["stderr"],
        optimized_functions=optimized_functions,
        file_path=file_path,
        errors=errors
    )

@mcp.tool()
def trace_and_optimize(
    script_command: str,
    project_path: str = ".",
    output_file: str = "codeflash.trace",
    max_functions: int = 100,
    timeout: Optional[int] = None
) -> TraceResult:
    """Trace a Python script execution and optimize based on the trace.
    
    Args:
        script_command: The Python command to trace (e.g., "python main.py")
        project_path: Path to the project root directory
        output_file: File to save the trace to
        max_functions: Maximum number of function calls to trace
        timeout: Maximum time in seconds to run the trace
        
    Returns:
        TraceResult with tracing and optimization details
    """
    # Parse the script command to extract the Python file
    cmd_parts = script_command.split()
    if len(cmd_parts) < 2:
        return TraceResult(
            success=False,
            message="Invalid script command. Expected format: 'python script.py [args]'"
        )
    
    script_args = cmd_parts[1:]  # Everything after 'python'
    
    args = ["optimize", "--output", output_file, "--max-function-count", str(max_functions)]
    
    if timeout:
        args.extend(["--timeout", str(timeout)])
    
    # Add the script arguments
    args.extend(script_args)
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    functions_traced = 0
    trace_file = None
    
    if result["success"]:
        trace_file = str(Path(project_path) / output_file)
        # Try to count functions from trace file
        try:
            if Path(trace_file).exists():
                with open(trace_file, 'r') as f:
                    content = f.read()
                    functions_traced = content.count('"function":')
        except Exception:
            pass
    
    return TraceResult(
        success=result["success"],
        trace_file=trace_file if result["success"] else None,
        functions_traced=functions_traced,
        message=result["stdout"] if result["success"] else result["stderr"]
    )

@mcp.tool() 
def optimize_from_replay_tests(
    test_files: List[str],
    project_path: str = ".",
    create_pr: bool = True
) -> OptimizationResult:
    """Optimize functions using existing replay test files.
    
    Args:
        test_files: List of paths to replay test files
        project_path: Path to the project root directory
        create_pr: Whether to create a pull request with optimizations
        
    Returns:
        OptimizationResult with optimization details
    """
    args = ["--replay-test"] + test_files
    
    if not create_pr:
        args.append("--no-pr")
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    optimized_functions = []
    errors = []
    
    if result["success"]:
        stdout = result["stdout"]
        lines = stdout.split('\n')
        for line in lines:
            if "Optimized function" in line:
                parts = line.split("Optimized function")
                if len(parts) > 1:
                    func_name = parts[1].strip().split()[0]
                    optimized_functions.append(func_name)
    else:
        if result["stderr"]:
            errors.append(result["stderr"])
    
    return OptimizationResult(
        success=result["success"],
        message=result["stdout"] if result["success"] else result["stderr"],
        optimized_functions=optimized_functions,
        errors=errors
    )

@mcp.tool()
def optimize_all_functions(
    start_directory: str = ".",
    project_path: str = ".",
    create_pr: bool = True
) -> OptimizationResult:
    """Optimize all functions in a project or directory.
    
    WARNING: This can take a very long time for large projects.
    
    Args:
        start_directory: Directory to start optimization from
        project_path: Path to the project root directory
        create_pr: Whether to create a pull request with optimizations
        
    Returns:
        OptimizationResult with details of all optimizations
    """
    args = ["--all"]
    
    if start_directory and start_directory != ".":
        args.append(start_directory)
    
    if not create_pr:
        args.append("--no-pr")
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    optimized_functions = []
    errors = []
    
    if result["success"]:
        stdout = result["stdout"]
        lines = stdout.split('\n')
        for line in lines:
            if "Optimized function" in line:
                parts = line.split("Optimized function")
                if len(parts) > 1:
                    func_name = parts[1].strip().split()[0]
                    optimized_functions.append(func_name)
    else:
        if result["stderr"]:
            errors.append(result["stderr"])
    
    return OptimizationResult(
        success=result["success"],
        message=result["stdout"] if result["success"] else result["stderr"],
        optimized_functions=optimized_functions,
        errors=errors
    )

@mcp.tool()
def initialize_project(
    project_path: str = ".",
    module_root: Optional[str] = None,
    tests_root: Optional[str] = None,
    test_framework: str = "pytest"
) -> Dict[str, Any]:
    """Initialize codeflash in a Python project.
    
    Args:
        project_path: Path to the project directory
        module_root: Path to the Python module root (auto-detected if not provided)
        tests_root: Path to the tests directory (auto-detected if not provided)
        test_framework: Testing framework to use ('pytest' or 'unittest')
        
    Returns:
        Dictionary with initialization results
    """
    args = ["init"]
    
    # Change to project directory and run init
    result = _run_codeflash_command(args, cwd=project_path)
    
    return {
        "success": result["success"],
        "message": result["stdout"] if result["success"] else result["stderr"],
        "config_created": result["success"]
    }

@mcp.tool()
def setup_github_actions(project_path: str = ".") -> Dict[str, Any]:
    """Set up GitHub Actions workflow for automatic codeflash optimization.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dictionary with setup results
    """
    args = ["init-actions"]
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    return {
        "success": result["success"],
        "message": result["stdout"] if result["success"] else result["stderr"],
        "workflow_created": result["success"]
    }

@mcp.tool()
def verify_installation(project_path: str = ".") -> Dict[str, Any]:
    """Verify that codeflash is working correctly by running a test optimization.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dictionary with verification results
    """
    args = ["--verify-setup"]
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    return {
        "success": result["success"],
        "message": result["stdout"] if result["success"] else result["stderr"],
        "verification_passed": result["success"]
    }

@mcp.tool()
def run_benchmarks(
    project_path: str = ".",
    benchmarks_root: Optional[str] = None
) -> Dict[str, Any]:
    """Run benchmark tests and calculate optimization impact.
    
    Args:
        project_path: Path to the project directory
        benchmarks_root: Path to benchmarks directory (auto-detected if not provided)
        
    Returns:
        Dictionary with benchmark results
    """
    args = ["--benchmark"]
    
    if benchmarks_root:
        args.extend(["--benchmarks-root", benchmarks_root])
    
    result = _run_codeflash_command(args, cwd=project_path)
    
    return {
        "success": result["success"],
        "message": result["stdout"] if result["success"] else result["stderr"],
        "benchmarks_completed": result["success"]
    }

@mcp.tool()
def get_optimization_help() -> Dict[str, Any]:
    """Get comprehensive help and usage information for codeflash optimization.
    
    Returns:
        Dictionary with help information and usage examples
    """
    return {
        "version": __version__,
        "description": "Codeflash is an AI-powered Python performance optimizer that automatically speeds up your code while verifying correctness.",
        "common_workflows": {
            "optimize_single_function": {
                "description": "Optimize a specific function by name",
                "example": "optimize_function('bubble_sort', file_path='algorithms.py')"
            },
            "optimize_entire_file": {
                "description": "Optimize all functions in a Python file",
                "example": "optimize_file('algorithms.py')"
            },
            "trace_and_optimize_script": {
                "description": "Trace a script execution and optimize based on usage patterns",
                "example": "trace_and_optimize('python main.py --data test_data.csv')"
            },
            "optimize_from_tests": {
                "description": "Use existing test files to guide optimization",
                "example": "optimize_from_replay_tests(['tests/test_algorithms.py'])"
            }
        },
        "best_practices": [
            "Always review optimizations before merging to ensure correctness",
            "Use trace_and_optimize for end-to-end script optimization",
            "Start with single functions before optimizing entire files",
            "Set up GitHub Actions for continuous optimization",
            "Run benchmarks to measure performance improvements"
        ],
        "supported_optimizations": [
            "Algorithm improvements (better time/space complexity)",
            "Library method optimization (using more efficient methods)",
            "Loop optimization and vectorization",
            "Caching and memoization",
            "Data structure optimization",
            "Removing wasteful computations"
        ]
    }

# Resource for providing project context
@mcp.resource("codeflash://project-config")
def get_project_config(uri: str) -> str:
    """Get codeflash configuration from the current project."""
    try:
        config_path = Path.cwd() / "pyproject.toml"
        if config_path.exists():
            return config_path.read_text()
        return "No codeflash configuration found in current project."
    except Exception as e:
        return f"Error reading project configuration: {e}"

def main():
    """Main entry point for the MCP server."""
    import uvloop
    uvloop.install()
    mcp.run()

if __name__ == "__main__":
    main()