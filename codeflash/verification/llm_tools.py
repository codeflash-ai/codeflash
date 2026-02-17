"""LLM Tool definitions for verification functions.

This module exposes verification functions as tools that can be called by LLMs.
Each tool has a JSON schema definition and a simplified wrapper function.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, Field

from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType
from codeflash.verification.parse_test_output import parse_test_xml
from codeflash.verification.test_runner import run_behavioral_tests
from codeflash.verification.verification_utils import TestConfig


class TestFileInput(BaseModel):
    """Input schema for a single test file."""

    test_file_path: str = Field(description="Absolute path to the test file to run")
    test_type: str = Field(
        default="existing_unit_test",
        description="Type of test: 'existing_unit_test', 'generated_regression', 'replay_test', or 'concolic_coverage_test'",
    )


class RunBehavioralTestsInput(BaseModel):
    """Input schema for the run_behavioral_tests tool."""

    test_files: list[TestFileInput] = Field(description="List of test files to run")
    test_framework: str = Field(default="pytest", description="Test framework to use: 'pytest' or 'unittest'")
    project_root: str = Field(description="Absolute path to the project root directory")
    pytest_timeout: int | None = Field(default=30, description="Timeout in seconds for each pytest test")
    verbose: bool = Field(default=False, description="Enable verbose output")


class TestResultOutput(BaseModel):
    """Output schema for a single test result."""

    test_id: str = Field(description="Unique identifier for the test")
    test_file: str = Field(description="Path to the test file")
    test_function: str | None = Field(description="Name of the test function")
    passed: bool = Field(description="Whether the test passed")
    runtime_ns: int | None = Field(description="Runtime in nanoseconds, if available")
    timed_out: bool = Field(description="Whether the test timed out")


class RunBehavioralTestsOutput(BaseModel):
    """Output schema for the run_behavioral_tests tool."""

    success: bool = Field(description="Whether the test run completed successfully")
    total_tests: int = Field(description="Total number of tests run")
    passed_tests: int = Field(description="Number of tests that passed")
    failed_tests: int = Field(description="Number of tests that failed")
    results: list[TestResultOutput] = Field(description="Detailed results for each test")
    stdout: str = Field(description="Standard output from the test run")
    stderr: str = Field(description="Standard error from the test run")
    error: str | None = Field(default=None, description="Error message if the run failed")


# JSON Schema for OpenAI-style function calling
RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_behavioral_tests",
        "description": (
            "Run behavioral tests to verify code correctness. "
            "This executes test files using pytest or unittest and returns detailed results "
            "including pass/fail status, runtime information, and any errors encountered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "test_files": {
                    "type": "array",
                    "description": "List of test files to run",
                    "items": {
                        "type": "object",
                        "properties": {
                            "test_file_path": {
                                "type": "string",
                                "description": "Absolute path to the test file to run",
                            },
                            "test_type": {
                                "type": "string",
                                "enum": [
                                    "existing_unit_test",
                                    "generated_regression",
                                    "replay_test",
                                    "concolic_coverage_test",
                                ],
                                "default": "existing_unit_test",
                                "description": "Type of test being run",
                            },
                        },
                        "required": ["test_file_path"],
                    },
                },
                "test_framework": {
                    "type": "string",
                    "enum": ["pytest", "unittest"],
                    "default": "pytest",
                    "description": "Test framework to use",
                },
                "project_root": {"type": "string", "description": "Absolute path to the project root directory"},
                "pytest_timeout": {
                    "type": "integer",
                    "default": 30,
                    "description": "Timeout in seconds for each pytest test",
                },
                "verbose": {"type": "boolean", "default": False, "description": "Enable verbose output"},
            },
            "required": ["test_files", "project_root"],
        },
    },
}


def _test_type_from_string(test_type_str: str) -> TestType:
    """Convert a string test type to TestType enum."""
    mapping = {
        "existing_unit_test": TestType.EXISTING_UNIT_TEST,
        "generated_regression": TestType.GENERATED_REGRESSION,
        "replay_test": TestType.REPLAY_TEST,
        "concolic_test": TestType.CONCOLIC_COVERAGE_TEST,
        "concolic_coverage_test": TestType.CONCOLIC_COVERAGE_TEST,
    }
    return mapping.get(test_type_str.lower(), TestType.EXISTING_UNIT_TEST)


def run_behavioral_tests_tool(
    test_files: list[dict[str, Any]],
    project_root: str,
    test_framework: str = "pytest",
    pytest_timeout: int | None = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run behavioral tests and return results in an LLM-friendly format.

    This is a simplified wrapper around run_behavioral_tests that accepts
    primitive types suitable for LLM tool calling and returns a structured
    dictionary response.

    Args:
        test_files: List of dicts with 'test_file_path' and optional 'test_type'
        project_root: Absolute path to the project root directory
        test_framework: Test framework to use ('pytest' or 'unittest')
        pytest_timeout: Timeout in seconds for each pytest test
        verbose: Enable verbose output

    Returns:
        Dictionary containing test results with success status, counts, and details

    Example:
        >>> result = run_behavioral_tests_tool(
        ...     test_files=[{"test_file_path": "/path/to/test_example.py"}], project_root="/path/to/project"
        ... )
        >>> print(result["passed_tests"], "tests passed")

    """
    try:
        project_root_path = Path(project_root).resolve()

        # Build TestFiles structure
        test_file_objects = []
        for tf in test_files:
            test_file_path = Path(tf["test_file_path"]).resolve()
            test_type_str = tf.get("test_type", "existing_unit_test")
            test_type = _test_type_from_string(test_type_str)

            test_file_objects.append(
                TestFile(
                    instrumented_behavior_file_path=test_file_path,
                    benchmarking_file_path=test_file_path,
                    original_file_path=test_file_path,
                    test_type=test_type,
                )
            )

        test_files_model = TestFiles(test_files=test_file_objects)

        # Set up test environment
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_TRACER_DISABLE"] = "1"

        # Ensure PYTHONPATH includes project root
        if "PYTHONPATH" not in test_env:
            test_env["PYTHONPATH"] = str(project_root_path)
        else:
            test_env["PYTHONPATH"] += os.pathsep + str(project_root_path)

        # Run the tests
        result_file_path, process, _, _ = run_behavioral_tests(
            test_paths=test_files_model,
            test_framework=test_framework,
            test_env=test_env,
            cwd=project_root_path,
            pytest_timeout=pytest_timeout,
        )

        # Create test config for parsing results
        test_config = TestConfig(
            tests_root=project_root_path, project_root_path=project_root_path, tests_project_rootdir=project_root_path
        )

        # Parse test results
        test_results = parse_test_xml(
            test_xml_file_path=result_file_path,
            test_files=test_files_model,
            test_config=test_config,
            run_result=process,
        )

        # Clean up result file
        result_file_path.unlink(missing_ok=True)

        # Build response
        results_list = []
        passed_count = 0
        failed_count = 0

        for result in test_results:
            passed = result.did_pass
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            results_list.append(
                {
                    "test_id": result.id.id() if result.id else "",
                    "test_file": str(result.file_name) if result.file_name else "",
                    "test_function": result.id.test_function_name if result.id else None,
                    "passed": passed,
                    "runtime_ns": result.runtime,
                    "timed_out": result.timed_out or False,
                }
            )

        return {
            "success": True,
            "total_tests": len(test_results),
            "passed_tests": passed_count,
            "failed_tests": failed_count,
            "results": results_list,
            "stdout": process.stdout if process.stdout else "",
            "stderr": process.stderr if process.stderr else "",
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "results": [],
            "stdout": "",
            "stderr": "",
            "error": str(e),
        }


class ToolEntry(TypedDict):
    schema: dict[str, Any]
    function: Callable[..., dict[str, Any]]


# Registry of available tools
AVAILABLE_TOOLS: dict[str, ToolEntry] = {
    "run_behavioral_tests": {"schema": RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA, "function": run_behavioral_tests_tool}
}


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Get the JSON schema for a tool by name.

    Args:
        tool_name: Name of the tool to get schema for

    Returns:
        JSON schema dict or None if tool not found

    """
    tool = AVAILABLE_TOOLS.get(tool_name)
    return tool["schema"] if tool else None


def get_all_tool_schemas() -> list[dict[str, Any]]:
    """Get JSON schemas for all available tools.

    Returns:
        List of JSON schema dicts for all tools

    """
    return [tool["schema"] for tool in AVAILABLE_TOOLS.values()]


def execute_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool function

    Returns:
        Tool execution result as a dictionary

    Raises:
        ValueError: If tool_name is not found

    """
    tool = AVAILABLE_TOOLS.get(tool_name)
    if not tool:
        msg = f"Unknown tool: {tool_name}"
        raise ValueError(msg)
    return tool["function"](**kwargs)
