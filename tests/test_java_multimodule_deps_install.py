"""Tests for ensure_multi_module_deps_installed in Java test runner."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.test_runner import _multimodule_deps_installed, ensure_multi_module_deps_installed


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the multi-module deps cache before each test."""
    _multimodule_deps_installed.clear()
    yield
    _multimodule_deps_installed.clear()


def test_skipped_for_single_module():
    """Single-module projects (test_module=None) should be a no-op."""
    result = ensure_multi_module_deps_installed(Path("/fake"), None, {})
    assert result is True
    assert len(_multimodule_deps_installed) == 0


@patch("codeflash.languages.java.test_runner.find_maven_executable", return_value="mvn")
@patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout")
def test_runs_install_command_with_correct_args(mock_run, mock_mvn):
    """Should run mvn install -DskipTests -pl <module> -am with validation skip flags."""
    mock_run.return_value = subprocess.CompletedProcess(args=["mvn"], returncode=0, stdout="", stderr="")

    root = Path("/project")
    result = ensure_multi_module_deps_installed(root, "guava-tests", {"JAVA_HOME": "/jdk"})

    assert result is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "mvn"
    assert "install" in cmd
    assert "-DskipTests" in cmd
    assert "-pl" in cmd
    assert "guava-tests" in cmd
    assert "-am" in cmd
    assert "-B" in cmd
    # Validation skip flags should be present
    assert "-Drat.skip=true" in cmd
    assert "-Dcheckstyle.skip=true" in cmd
    # cwd should be maven_root
    assert mock_run.call_args[1]["cwd"] == root


@patch("codeflash.languages.java.test_runner.find_maven_executable", return_value="mvn")
@patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout")
def test_caches_and_does_not_rerun(mock_run, mock_mvn):
    """Second call with same (root, module) should be cached — no Maven invocation."""
    mock_run.return_value = subprocess.CompletedProcess(args=["mvn"], returncode=0, stdout="", stderr="")

    root = Path("/project")
    ensure_multi_module_deps_installed(root, "guava-tests", {})
    assert mock_run.call_count == 1

    # Second call — should be cached
    result = ensure_multi_module_deps_installed(root, "guava-tests", {})
    assert result is True
    assert mock_run.call_count == 1  # NOT called again


@patch("codeflash.languages.java.test_runner.find_maven_executable", return_value="mvn")
@patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout")
def test_different_modules_not_cached(mock_run, mock_mvn):
    """Different test modules should each trigger their own install."""
    mock_run.return_value = subprocess.CompletedProcess(args=["mvn"], returncode=0, stdout="", stderr="")

    root = Path("/project")
    ensure_multi_module_deps_installed(root, "module-a", {})
    ensure_multi_module_deps_installed(root, "module-b", {})
    assert mock_run.call_count == 2


@patch("codeflash.languages.java.test_runner.find_maven_executable", return_value="mvn")
@patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout")
def test_returns_false_on_maven_failure(mock_run, mock_mvn):
    """Non-zero exit code should return False and NOT cache."""
    mock_run.return_value = subprocess.CompletedProcess(args=["mvn"], returncode=1, stdout="", stderr="BUILD FAILURE")

    root = Path("/project")
    result = ensure_multi_module_deps_installed(root, "guava-tests", {})
    assert result is False
    assert len(_multimodule_deps_installed) == 0


@patch("codeflash.languages.java.test_runner.find_maven_executable", return_value=None)
def test_returns_false_when_maven_not_found(mock_mvn):
    """Should return False if Maven executable is not found."""
    result = ensure_multi_module_deps_installed(Path("/fake"), "module", {})
    assert result is False
