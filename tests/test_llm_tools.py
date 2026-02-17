"""Tests for LLM tools in the verification module."""

import tempfile
from pathlib import Path

import pytest

from codeflash.verification.llm_tools import (
    AVAILABLE_TOOLS,
    RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA,
    execute_tool,
    get_all_tool_schemas,
    get_tool_schema,
    run_behavioral_tests_tool,
)


def test_run_behavioral_tests_tool_schema_structure() -> None:
    """Test that the tool schema has the correct structure."""
    schema = RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA

    assert schema["type"] == "function"
    assert "function" in schema
    func = schema["function"]
    assert isinstance(func, dict)
    assert func["name"] == "run_behavioral_tests"
    assert "description" in func
    assert "parameters" in func

    params = func["parameters"]
    assert isinstance(params, dict)
    assert params["type"] == "object"
    assert "test_files" in params["properties"]
    assert "project_root" in params["properties"]
    assert "test_framework" in params["properties"]
    assert "test_files" in params["required"]
    assert "project_root" in params["required"]


def test_get_tool_schema() -> None:
    """Test getting tool schema by name."""
    schema = get_tool_schema("run_behavioral_tests")
    assert schema is not None
    assert schema["function"]["name"] == "run_behavioral_tests"

    # Non-existent tool should return None
    assert get_tool_schema("non_existent_tool") is None


def test_get_all_tool_schemas() -> None:
    """Test getting all tool schemas."""
    schemas = get_all_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 1

    # Check that run_behavioral_tests is in the list
    names = [s["function"]["name"] for s in schemas]
    assert "run_behavioral_tests" in names


def test_available_tools_registry() -> None:
    """Test that the AVAILABLE_TOOLS registry has correct structure."""
    assert "run_behavioral_tests" in AVAILABLE_TOOLS

    tool = AVAILABLE_TOOLS["run_behavioral_tests"]
    assert "schema" in tool
    assert "function" in tool
    assert callable(tool["function"])


def test_execute_tool_unknown_tool() -> None:
    """Test that execute_tool raises ValueError for unknown tools."""
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("non_existent_tool")


def test_run_behavioral_tests_tool_pytest() -> None:
    """Test running pytest tests through the LLM tool."""
    test_code = """
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
"""
    # Use repo root for project_root to avoid path resolution issues
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
        test_file_path = Path(temp_dir) / "test_example.py"
        test_file_path.write_text(test_code, encoding="utf-8")

        result = run_behavioral_tests_tool(
            test_files=[{"test_file_path": str(test_file_path)}],
            project_root=str(repo_root),
            test_framework="pytest",
            pytest_timeout=30,
        )

        assert result["success"] is True
        assert result["total_tests"] >= 1
        assert result["passed_tests"] >= 1
        assert result["failed_tests"] == 0
        assert result["error"] is None
        assert isinstance(result["results"], list)


def test_run_behavioral_tests_tool_failing_test() -> None:
    """Test running a failing test through the LLM tool."""
    test_code = """
def test_failing():
    assert 1 == 2, "This test should fail"
"""
    # Use repo root for project_root to avoid path resolution issues
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
        test_file_path = Path(temp_dir) / "test_failing.py"
        test_file_path.write_text(test_code, encoding="utf-8")

        result = run_behavioral_tests_tool(
            test_files=[{"test_file_path": str(test_file_path)}],
            project_root=str(repo_root),
            test_framework="pytest",
            pytest_timeout=30,
        )

        assert result["success"] is True  # The run completed, even if tests failed
        assert result["failed_tests"] >= 1


def test_run_behavioral_tests_tool_via_execute() -> None:
    """Test running tests through the execute_tool interface."""
    test_code = """
def test_simple():
    assert True
"""
    # Use repo root for project_root to avoid path resolution issues
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
        test_file_path = Path(temp_dir) / "test_simple.py"
        test_file_path.write_text(test_code, encoding="utf-8")

        result = execute_tool(
            "run_behavioral_tests",
            test_files=[{"test_file_path": str(test_file_path)}],
            project_root=str(repo_root),
        )

        assert result["success"] is True
        assert result["error"] is None


def test_run_behavioral_tests_tool_invalid_path() -> None:
    """Test handling of invalid test file path."""
    # Use repo root for project_root
    repo_root = Path(__file__).resolve().parent.parent

    result = run_behavioral_tests_tool(
        test_files=[{"test_file_path": "/non/existent/test_file.py"}],
        project_root=str(repo_root),
        test_framework="pytest",
    )

    # Should complete but with no tests found
    assert result["success"] is True
    assert result["total_tests"] == 0


def test_run_behavioral_tests_tool_with_test_type() -> None:
    """Test specifying test type."""
    test_code = """
def test_with_type():
    assert True
"""
    # Use repo root for project_root to avoid path resolution issues
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
        test_file_path = Path(temp_dir) / "test_typed.py"
        test_file_path.write_text(test_code, encoding="utf-8")

        result = run_behavioral_tests_tool(
            test_files=[
                {
                    "test_file_path": str(test_file_path),
                    "test_type": "existing_unit_test",
                }
            ],
            project_root=str(repo_root),
        )

        assert result["success"] is True
