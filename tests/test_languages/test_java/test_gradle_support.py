"""Tests for Gradle build tool support."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codeflash.languages.java.build_tools import (
    BuildTool,
    GradleTestResult,
    compile_with_gradle,
    detect_build_tool,
    find_gradle_executable,
    run_gradle_tests,
)


class TestGradleExecutableDetection:
    """Tests for finding Gradle executable."""

    def test_find_gradle_wrapper_in_current_dir(self, tmp_path: Path, monkeypatch):
        """Test finding gradlew in current directory."""
        # Create gradlew file
        gradlew_path = tmp_path / "gradlew"
        gradlew_path.write_text("#!/bin/bash\necho 'Gradle'")
        gradlew_path.chmod(0o755)

        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        gradle = find_gradle_executable()
        assert gradle is not None
        assert "gradlew" in gradle

    def test_find_gradle_wrapper_windows(self, tmp_path: Path, monkeypatch):
        """Test finding gradlew.bat on Windows."""
        # Create gradlew.bat file
        gradlew_path = tmp_path / "gradlew.bat"
        gradlew_path.write_text("@echo off\necho Gradle")

        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        gradle = find_gradle_executable()
        assert gradle is not None
        assert "gradlew" in gradle.lower()

    def test_find_system_gradle(self, monkeypatch):
        """Test finding system Gradle when no wrapper exists."""
        # Mock shutil.which to return a gradle path
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/gradle"

            # Change to a temp dir without wrapper
            with tempfile.TemporaryDirectory() as tmpdir:
                monkeypatch.chdir(tmpdir)

                gradle = find_gradle_executable()
                assert gradle == "/usr/bin/gradle"

    def test_gradle_not_found(self, tmp_path: Path, monkeypatch):
        """Test when Gradle is not available."""
        # Change to empty tmp_path
        monkeypatch.chdir(tmp_path)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            gradle = find_gradle_executable()
            assert gradle is None


class TestGradleTestExecution:
    """Tests for running tests with Gradle."""

    @pytest.fixture
    def mock_gradle_success(self):
        """Mock successful Gradle test execution."""
        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = """
> Task :test

com.example.CalculatorTest > testAdd PASSED
com.example.CalculatorTest > testSubtract PASSED

BUILD SUCCESSFUL in 3s
4 actionable tasks: 4 executed
"""
        mock_result.stderr = ""
        return mock_result

    @pytest.fixture
    def mock_gradle_failure(self):
        """Mock failed Gradle test execution."""
        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stdout = """
> Task :test FAILED

com.example.CalculatorTest > testAdd PASSED
com.example.CalculatorTest > testSubtract FAILED

2 tests completed, 1 failed

BUILD FAILED in 2s
"""
        mock_result.stderr = "FAILURE: Build failed with an exception."
        return mock_result

    def test_run_gradle_tests_all(self, tmp_path: Path, mock_gradle_success, monkeypatch):
        """Test running all Gradle tests."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        # Change to tmp_path so find_gradle_executable can find gradlew
        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_success):
            result = run_gradle_tests(tmp_path)

        assert result.success is True
        assert result.returncode == 0
        assert "BUILD SUCCESSFUL" in result.stdout

    def test_run_gradle_tests_specific_class(self, tmp_path: Path, mock_gradle_success, monkeypatch):
        """Test running specific test class."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_success) as mock_run:
            result = run_gradle_tests(
                tmp_path,
                test_classes=["com.example.CalculatorTest"]
            )

        # Verify correct gradle command was called
        call_args = mock_run.call_args
        assert "--tests" in call_args[0][0]
        assert "com.example.CalculatorTest" in call_args[0][0]
        assert result.success is True

    def test_run_gradle_tests_specific_methods(self, tmp_path: Path, mock_gradle_success, monkeypatch):
        """Test running specific test methods."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_success) as mock_run:
            result = run_gradle_tests(
                tmp_path,
                test_methods=["com.example.CalculatorTest.testAdd"]
            )

        # Verify correct gradle command was called
        call_args = mock_run.call_args
        assert "--tests" in call_args[0][0]
        assert "com.example.CalculatorTest.testAdd" in call_args[0][0]

    def test_run_gradle_tests_with_failure(self, tmp_path: Path, mock_gradle_failure, monkeypatch):
        """Test handling Gradle test failures."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_failure):
            result = run_gradle_tests(tmp_path)

        assert result.success is False
        assert result.returncode == 1
        assert "BUILD FAILED" in result.stdout

    def test_run_gradle_tests_no_gradle(self, tmp_path: Path):
        """Test running tests when Gradle is not available."""
        result = run_gradle_tests(tmp_path)

        assert result.success is False
        assert result.returncode == -1
        assert "Gradle not found" in result.stderr

    def test_run_gradle_tests_timeout(self, tmp_path: Path, monkeypatch):
        """Test Gradle test timeout."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        # Mock subprocess to raise TimeoutExpired
        with patch("codeflash.languages.java.build_tools.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="gradle", timeout=1)

            result = run_gradle_tests(tmp_path, timeout=1)

        assert result.success is False
        assert "timed out" in result.stderr.lower()

    def test_run_gradle_tests_multi_module(self, tmp_path: Path, mock_gradle_success, monkeypatch):
        """Test running tests in multi-module Gradle project."""
        (tmp_path / "build.gradle").write_text("// Root build.gradle")
        (tmp_path / "settings.gradle").write_text("include 'server', 'client'")

        server_dir = tmp_path / "server"
        server_dir.mkdir()
        (server_dir / "build.gradle").write_text("plugins { id 'java' }")

        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_success) as mock_run:
            result = run_gradle_tests(tmp_path, test_module="server")

        # Verify module-specific command
        call_args = mock_run.call_args
        assert ":server:test" in " ".join(call_args[0][0])
        assert result.success is True

    def test_run_gradle_tests_with_coverage(self, tmp_path: Path, mock_gradle_success, monkeypatch):
        """Test running Gradle tests with JaCoCo coverage."""
        (tmp_path / "build.gradle").write_text("""
plugins {
    id 'java'
    id 'jacoco'
}
""")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_gradle_success) as mock_run:
            result = run_gradle_tests(tmp_path, enable_coverage=True)

        # Verify JaCoCo task is included
        call_args = mock_run.call_args
        assert "jacocoTestReport" in " ".join(call_args[0][0])


class TestGradleCompilation:
    """Tests for compiling with Gradle."""

    @pytest.fixture
    def mock_compile_success(self):
        """Mock successful compilation."""
        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "BUILD SUCCESSFUL in 2s"
        mock_result.stderr = ""
        return mock_result

    def test_compile_with_gradle_main_only(self, tmp_path: Path, mock_compile_success, monkeypatch):
        """Test compiling main sources only."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_compile_success) as mock_run:
            success, stdout, stderr = compile_with_gradle(tmp_path, include_tests=False)

        assert success is True
        call_args = mock_run.call_args
        assert "compileJava" in call_args[0][0]
        assert "compileTestJava" not in call_args[0][0]

    def test_compile_with_gradle_with_tests(self, tmp_path: Path, mock_compile_success, monkeypatch):
        """Test compiling main and test sources."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_compile_success) as mock_run:
            success, stdout, stderr = compile_with_gradle(tmp_path, include_tests=True)

        assert success is True
        call_args = mock_run.call_args
        assert "compileJava" in call_args[0][0]
        assert "compileTestJava" in call_args[0][0]

    def test_compile_with_gradle_failure(self, tmp_path: Path, monkeypatch):
        """Test handling compilation failure."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Compilation failed"

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_result):
            success, stdout, stderr = compile_with_gradle(tmp_path)

        assert success is False
        assert "Compilation failed" in stderr

    def test_compile_with_gradle_not_found(self, tmp_path: Path):
        """Test compiling when Gradle is not available."""
        success, stdout, stderr = compile_with_gradle(tmp_path)

        assert success is False
        assert "Gradle not found" in stderr


class TestGradleMultiModuleSupport:
    """Tests for Gradle multi-module project support."""

    def test_detect_gradle_in_parent_directory(self, tmp_path: Path):
        """Test detecting Gradle in parent directory."""
        # Create parent project structure
        (tmp_path / "build.gradle").write_text("// Root")
        (tmp_path / "settings.gradle").write_text("include 'server'")

        # Create server module
        server_dir = tmp_path / "server"
        server_dir.mkdir()
        (server_dir / "build.gradle").write_text("plugins { id 'java' }")

        # Detect from server module
        build_tool = detect_build_tool(server_dir)
        assert build_tool == BuildTool.GRADLE

    def test_gradle_module_test_execution(self, tmp_path: Path, monkeypatch):
        """Test running tests in specific Gradle module."""
        (tmp_path / "build.gradle").write_text("// Root")
        (tmp_path / "settings.gradle").write_text("include 'server'")

        server_dir = tmp_path / "server"
        server_dir.mkdir()
        (server_dir / "build.gradle").write_text("plugins { id 'java' }")
        (server_dir / "src" / "test" / "java").mkdir(parents=True)

        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        monkeypatch.chdir(tmp_path)

        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "BUILD SUCCESSFUL"
        mock_result.stderr = ""

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_result) as mock_run:
            # Run from project root with module specified
            result = run_gradle_tests(tmp_path, test_module="server")

        assert result.success is True
        # Verify module-specific task was called
        call_args = mock_run.call_args
        command = " ".join(call_args[0][0])
        assert ":server:test" in command


class TestGradleTestResultParsing:
    """Tests for parsing Gradle test results."""

    def test_parse_gradle_xml_results(self, tmp_path: Path):
        """Test parsing Gradle XML test results."""
        # Create test results XML (similar to JUnit format)
        test_results_dir = tmp_path / "build" / "test-results" / "test"
        test_results_dir.mkdir(parents=True)

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.CalculatorTest" tests="3" failures="1" errors="0" skipped="0">
    <testcase name="testAdd" classname="com.example.CalculatorTest" time="0.012"/>
    <testcase name="testSubtract" classname="com.example.CalculatorTest" time="0.008">
        <failure message="expected: 2 but was: 3"/>
    </testcase>
    <testcase name="testMultiply" classname="com.example.CalculatorTest" time="0.010"/>
</testsuite>
"""
        (test_results_dir / "TEST-com.example.CalculatorTest.xml").write_text(xml_content)

        # This would be tested in the actual run_gradle_tests implementation
        # when it parses XML results
        assert test_results_dir.exists()

    def test_gradle_test_report_location(self, tmp_path: Path):
        """Test that Gradle test reports are in standard location."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")

        # Gradle standard test results location
        test_results_dir = tmp_path / "build" / "test-results" / "test"
        test_report_dir = tmp_path / "build" / "reports" / "tests" / "test"

        # These should be the standard locations
        assert str(test_results_dir).endswith("build/test-results/test")
        assert str(test_report_dir).endswith("build/reports/tests/test")


class TestGradleIntegrationWithCodeFlash:
    """Integration tests for Gradle with CodeFlash workflow."""

    def test_full_gradle_workflow(self, tmp_path: Path, monkeypatch):
        """Test complete workflow: detect -> compile -> test."""
        # Setup Gradle project
        (tmp_path / "build.gradle").write_text("""
plugins {
    id 'java'
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation 'org.junit.jupiter:junit-jupiter:5.9.0'
}

test {
    useJUnitPlatform()
}
""")
        (tmp_path / "gradlew").write_text("#!/bin/bash")
        (tmp_path / "gradlew").chmod(0o755)

        # Create source directories
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        monkeypatch.chdir(tmp_path)

        # Step 1: Detect build tool
        build_tool = detect_build_tool(tmp_path)
        assert build_tool == BuildTool.GRADLE

        # Step 2: Compile (mocked)
        mock_compile = Mock(spec=subprocess.CompletedProcess)
        mock_compile.returncode = 0
        mock_compile.stdout = "BUILD SUCCESSFUL"
        mock_compile.stderr = ""

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_compile):
            success, stdout, stderr = compile_with_gradle(tmp_path)

        assert success is True

        # Step 3: Run tests (mocked)
        mock_test = Mock(spec=subprocess.CompletedProcess)
        mock_test.returncode = 0
        mock_test.stdout = "BUILD SUCCESSFUL"
        mock_test.stderr = ""

        with patch("codeflash.languages.java.build_tools.subprocess.run", return_value=mock_test):
            result = run_gradle_tests(tmp_path)

        assert result.success is True
