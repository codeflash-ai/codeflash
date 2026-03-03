"""Tests for Mocha test runner functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from junitparser import JUnitXml


class TestMochaJsonToJunitXml:
    """Tests for converting Mocha JSON reporter output to JUnit XML."""

    def test_passing_tests(self):
        from codeflash.languages.javascript.mocha_runner import mocha_json_to_junit_xml

        mocha_json = json.dumps(
            {
                "stats": {"tests": 2, "passes": 2, "failures": 0, "duration": 50},
                "tests": [
                    {
                        "title": "should add numbers",
                        "fullTitle": "math should add numbers",
                        "duration": 20,
                        "err": {},
                    },
                    {
                        "title": "should subtract numbers",
                        "fullTitle": "math should subtract numbers",
                        "duration": 30,
                        "err": {},
                    },
                ],
                "passes": [],
                "failures": [],
                "pending": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"
            mocha_json_to_junit_xml(mocha_json, output_file)

            assert output_file.exists()
            xml = JUnitXml.fromfile(str(output_file))
            total_tests = sum(suite.tests for suite in xml)
            assert total_tests == 2

    def test_failing_tests(self):
        from codeflash.languages.javascript.mocha_runner import mocha_json_to_junit_xml

        mocha_json = json.dumps(
            {
                "stats": {"tests": 1, "passes": 0, "failures": 1, "duration": 10},
                "tests": [
                    {
                        "title": "should fail",
                        "fullTitle": "errors should fail",
                        "duration": 10,
                        "err": {
                            "message": "expected 1 to equal 2",
                            "stack": "AssertionError: expected 1 to equal 2\n    at Context.<anonymous>",
                        },
                    },
                ],
                "passes": [],
                "failures": [],
                "pending": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"
            mocha_json_to_junit_xml(mocha_json, output_file)

            assert output_file.exists()
            xml = JUnitXml.fromfile(str(output_file))
            total_failures = sum(suite.failures for suite in xml)
            assert total_failures == 1

    def test_pending_tests(self):
        from codeflash.languages.javascript.mocha_runner import mocha_json_to_junit_xml

        mocha_json = json.dumps(
            {
                "stats": {"tests": 1, "passes": 0, "failures": 0, "pending": 1, "duration": 0},
                "tests": [
                    {
                        "title": "should be pending",
                        "fullTitle": "todo should be pending",
                        "duration": 0,
                        "pending": True,
                        "err": {},
                    },
                ],
                "passes": [],
                "failures": [],
                "pending": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"
            mocha_json_to_junit_xml(mocha_json, output_file)

            assert output_file.exists()
            xml = JUnitXml.fromfile(str(output_file))
            # Should parse without error and have the test
            total_tests = sum(suite.tests for suite in xml)
            assert total_tests == 1

    def test_invalid_json_writes_empty_xml(self):
        from codeflash.languages.javascript.mocha_runner import mocha_json_to_junit_xml

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"
            mocha_json_to_junit_xml("not valid json {{{", output_file)

            assert output_file.exists()
            content = output_file.read_text()
            assert "<testsuites" in content

    def test_multiple_suites(self):
        from codeflash.languages.javascript.mocha_runner import mocha_json_to_junit_xml

        mocha_json = json.dumps(
            {
                "stats": {"tests": 3, "passes": 3, "failures": 0, "duration": 100},
                "tests": [
                    {"title": "test1", "fullTitle": "suite A test1", "duration": 10, "err": {}},
                    {"title": "test2", "fullTitle": "suite A test2", "duration": 20, "err": {}},
                    {"title": "test3", "fullTitle": "suite B test3", "duration": 30, "err": {}},
                ],
                "passes": [],
                "failures": [],
                "pending": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"
            mocha_json_to_junit_xml(mocha_json, output_file)

            xml = JUnitXml.fromfile(str(output_file))
            suite_names = [suite.name for suite in xml]
            assert "suite A" in suite_names
            assert "suite B" in suite_names


class TestExtractMochaJson:
    """Tests for extracting Mocha JSON from mixed stdout."""

    def test_clean_json(self):
        from codeflash.languages.javascript.mocha_runner import _extract_mocha_json

        data = {"stats": {"tests": 1}, "tests": []}
        result = _extract_mocha_json(json.dumps(data))
        assert result is not None
        assert json.loads(result)["stats"]["tests"] == 1

    def test_json_with_leading_output(self):
        from codeflash.languages.javascript.mocha_runner import _extract_mocha_json

        stdout = 'Some console output\n{"stats": {"tests": 1}, "tests": []}'
        result = _extract_mocha_json(stdout)
        assert result is not None
        assert json.loads(result)["stats"]["tests"] == 1

    def test_json_with_codeflash_markers(self):
        from codeflash.languages.javascript.mocha_runner import _extract_mocha_json

        data = {"stats": {"tests": 1}, "tests": []}
        stdout = f"!######START:test:module:0:test_name######!\n{json.dumps(data)}\n!######END######!"
        result = _extract_mocha_json(stdout)
        assert result is not None

    def test_no_json_returns_none(self):
        from codeflash.languages.javascript.mocha_runner import _extract_mocha_json

        result = _extract_mocha_json("no json here at all")
        assert result is None


class TestFindMochaProjectRoot:
    """Tests for finding Mocha project root."""

    def test_finds_mocharc_yml(self):
        from codeflash.languages.javascript.mocha_runner import _find_mocha_project_root

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".mocharc.yml").write_text("timeout: 5000\n")
            sub = root / "src" / "lib"
            sub.mkdir(parents=True)
            test_file = sub / "test.js"
            test_file.write_text("// test")

            result = _find_mocha_project_root(test_file)
            assert result == root

    def test_finds_mocharc_json(self):
        from codeflash.languages.javascript.mocha_runner import _find_mocha_project_root

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".mocharc.json").write_text("{}")
            test_file = root / "test.js"
            test_file.write_text("// test")

            result = _find_mocha_project_root(test_file)
            assert result == root

    def test_falls_back_to_package_json(self):
        from codeflash.languages.javascript.mocha_runner import _find_mocha_project_root

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "package.json").write_text('{"name": "test"}')
            sub = root / "test"
            sub.mkdir()
            test_file = sub / "test.js"
            test_file.write_text("// test")

            result = _find_mocha_project_root(test_file)
            assert result == root


class TestMochaBehavioralCommand:
    """Tests for building Mocha behavioral commands."""

    def test_basic_command(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_behavioral_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_behavioral_command(test_files=[test_file])
            assert "npx" in cmd
            assert "mocha" in cmd
            assert "--reporter" in cmd
            assert "json" in cmd
            assert "--jobs" in cmd
            assert "1" in cmd
            assert "--exit" in cmd

    def test_timeout_flag(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_behavioral_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_behavioral_command(test_files=[test_file], timeout=30)
            timeout_idx = cmd.index("--timeout")
            assert cmd[timeout_idx + 1] == "30000"

    def test_default_timeout(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_behavioral_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_behavioral_command(test_files=[test_file])
            timeout_idx = cmd.index("--timeout")
            assert cmd[timeout_idx + 1] == "60000"


class TestMochaBenchmarkingCommand:
    """Tests for building Mocha benchmarking commands."""

    def test_basic_command(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_benchmarking_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_benchmarking_command(test_files=[test_file])
            assert "npx" in cmd
            assert "mocha" in cmd
            assert "--exit" in cmd

    def test_default_timeout_is_longer(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_benchmarking_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_benchmarking_command(test_files=[test_file])
            timeout_idx = cmd.index("--timeout")
            assert cmd[timeout_idx + 1] == "120000"


class TestMochaLineProfileCommand:
    """Tests for building Mocha line profile commands."""

    def test_basic_command(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_line_profile_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_line_profile_command(test_files=[test_file])
            assert "npx" in cmd
            assert "mocha" in cmd
            assert "--exit" in cmd

    def test_timeout_conversion(self):
        from codeflash.languages.javascript.mocha_runner import _build_mocha_line_profile_command

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.js"
            test_file.write_text("// test")

            cmd = _build_mocha_line_profile_command(test_files=[test_file], timeout=45)
            timeout_idx = cmd.index("--timeout")
            assert cmd[timeout_idx + 1] == "45000"


class TestRunMochaBehavioralTests:
    """Tests for running Mocha behavioral tests with mocked subprocess."""

    @patch("codeflash.languages.javascript.mocha_runner.subprocess.run")
    @patch("codeflash.languages.javascript.mocha_runner._ensure_runtime_files")
    def test_sets_codeflash_env_vars(self, mock_ensure, mock_run):
        from codeflash.languages.javascript.mocha_runner import run_mocha_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        mocha_output = json.dumps(
            {"stats": {"tests": 1, "passes": 1, "failures": 0, "duration": 10}, "tests": [{"title": "t", "fullTitle": "s t", "duration": 10, "err": {}}], "passes": [], "failures": [], "pending": []}
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=mocha_output, stderr="", args=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_file = tmpdir_path / "test.test.js"
            test_file.write_text("// test")

            test_paths = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            result_file, result, cov, _ = run_mocha_behavioral_tests(
                test_paths=test_paths,
                test_env={},
                cwd=tmpdir_path,
                candidate_index=3,
            )

            # Verify env vars were passed
            call_kwargs = mock_run.call_args
            env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
            assert env.get("CODEFLASH_MODE") == "behavior"
            assert env.get("CODEFLASH_TEST_ITERATION") == "3"
            assert env.get("CODEFLASH_RANDOM_SEED") == "42"

    @patch("codeflash.languages.javascript.mocha_runner.subprocess.run")
    @patch("codeflash.languages.javascript.mocha_runner._ensure_runtime_files")
    def test_returns_none_coverage(self, mock_ensure, mock_run):
        from codeflash.languages.javascript.mocha_runner import run_mocha_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        mocha_output = json.dumps(
            {"stats": {"tests": 0, "passes": 0, "failures": 0, "duration": 0}, "tests": [], "passes": [], "failures": [], "pending": []}
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=mocha_output, stderr="", args=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_file = tmpdir_path / "test.test.js"
            test_file.write_text("// test")

            test_paths = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            _, _, coverage_path, _ = run_mocha_behavioral_tests(
                test_paths=test_paths,
                test_env={},
                cwd=tmpdir_path,
            )
            assert coverage_path is None


class TestRunMochaBenchmarkingTests:
    """Tests for running Mocha benchmarking tests with mocked subprocess."""

    @patch("codeflash.languages.javascript.mocha_runner.subprocess.run")
    @patch("codeflash.languages.javascript.mocha_runner._ensure_runtime_files")
    def test_sets_perf_env_vars(self, mock_ensure, mock_run):
        from codeflash.languages.javascript.mocha_runner import run_mocha_benchmarking_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        mocha_output = json.dumps(
            {"stats": {"tests": 1, "passes": 1, "failures": 0, "duration": 100}, "tests": [{"title": "perf", "fullTitle": "bench perf", "duration": 100, "err": {}}], "passes": [], "failures": [], "pending": []}
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=mocha_output, stderr="", args=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_file = tmpdir_path / "perf.test.js"
            test_file.write_text("// perf test")

            test_paths = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            run_mocha_benchmarking_tests(
                test_paths=test_paths,
                test_env={},
                cwd=tmpdir_path,
                min_loops=3,
                max_loops=50,
                target_duration_ms=5000,
                stability_check=False,
            )

            call_kwargs = mock_run.call_args
            env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
            assert env.get("CODEFLASH_MODE") == "performance"
            assert env.get("CODEFLASH_PERF_LOOP_COUNT") == "50"
            assert env.get("CODEFLASH_PERF_MIN_LOOPS") == "3"
            assert env.get("CODEFLASH_PERF_TARGET_DURATION_MS") == "5000"
            assert env.get("CODEFLASH_PERF_STABILITY_CHECK") == "false"


class TestSanitizeMochaImports:
    """Tests for stripping wrong framework imports from Mocha tests."""

    def test_strips_vitest_import(self):
        from codeflash.languages.javascript.edit_tests import sanitize_mocha_imports

        source = "import { describe, test, expect, vi } from 'vitest'\nconst x = 1;\n"
        result = sanitize_mocha_imports(source)
        assert "vitest" not in result
        assert "const x = 1;" in result

    def test_strips_jest_globals_import(self):
        from codeflash.languages.javascript.edit_tests import sanitize_mocha_imports

        source = "import { jest, describe, it, expect } from '@jest/globals'\nconst x = 1;\n"
        result = sanitize_mocha_imports(source)
        assert "@jest/globals" not in result
        assert "const x = 1;" in result

    def test_strips_mocha_require(self):
        from codeflash.languages.javascript.edit_tests import sanitize_mocha_imports

        source = "const { describe, it, expect } = require('mocha');\nconst x = 1;\n"
        result = sanitize_mocha_imports(source)
        assert "require('mocha')" not in result
        assert "const x = 1;" in result

    def test_strips_vitest_comment(self):
        from codeflash.languages.javascript.edit_tests import sanitize_mocha_imports

        source = "// vitest imports (REQUIRED for vitest)\nimport { describe } from 'vitest'\nconst x = 1;\n"
        result = sanitize_mocha_imports(source)
        assert "vitest" not in result
        assert "const x = 1;" in result

    def test_preserves_unrelated_imports(self):
        from codeflash.languages.javascript.edit_tests import sanitize_mocha_imports

        source = "const sinon = require('sinon');\nconst assert = require('node:assert/strict');\n"
        result = sanitize_mocha_imports(source)
        assert "sinon" in result
        assert "node:assert/strict" in result


class TestInjectTestGlobalsModuleSystem:
    """Tests for inject_test_globals with different module systems."""

    def test_mocha_esm_uses_import(self):
        from codeflash.languages.javascript.edit_tests import inject_test_globals
        from codeflash.models.models import GeneratedTests, GeneratedTestsList

        tests = GeneratedTestsList(
            generated_tests=[
                GeneratedTests(
                    generated_original_test_source="describe('test', () => {});",
                    instrumented_behavior_test_source="describe('test', () => {});",
                    instrumented_perf_test_source="describe('test', () => {});",
                    behavior_file_path=Path("test.test.js"),
                    perf_file_path=Path("test.perf.test.js"),
                )
            ]
        )

        result = inject_test_globals(tests, test_framework="mocha", module_system="esm")
        assert "import assert from 'node:assert/strict'" in result.generated_tests[0].generated_original_test_source

    def test_mocha_cjs_uses_require(self):
        from codeflash.languages.javascript.edit_tests import inject_test_globals
        from codeflash.models.models import GeneratedTests, GeneratedTestsList

        tests = GeneratedTestsList(
            generated_tests=[
                GeneratedTests(
                    generated_original_test_source="describe('test', () => {});",
                    instrumented_behavior_test_source="describe('test', () => {});",
                    instrumented_perf_test_source="describe('test', () => {});",
                    behavior_file_path=Path("test.test.js"),
                    perf_file_path=Path("test.perf.test.js"),
                )
            ]
        )

        result = inject_test_globals(tests, test_framework="mocha", module_system="commonjs")
        src = result.generated_tests[0].generated_original_test_source
        assert "const assert = require('node:assert/strict')" in src
        assert "import assert" not in src

    def test_vitest_always_uses_import(self):
        from codeflash.languages.javascript.edit_tests import inject_test_globals
        from codeflash.models.models import GeneratedTests, GeneratedTestsList

        tests = GeneratedTestsList(
            generated_tests=[
                GeneratedTests(
                    generated_original_test_source="describe('test', () => {});",
                    instrumented_behavior_test_source="describe('test', () => {});",
                    instrumented_perf_test_source="describe('test', () => {});",
                    behavior_file_path=Path("test.test.js"),
                    perf_file_path=Path("test.perf.test.js"),
                )
            ]
        )

        result = inject_test_globals(tests, test_framework="vitest", module_system="commonjs")
        assert "from 'vitest'" in result.generated_tests[0].generated_original_test_source


class TestEnsureModuleSystemCompatibilityMixed:
    """Tests for ensure_module_system_compatibility with mixed ESM+CJS code."""

    def test_converts_imports_in_mixed_code_to_cjs(self):
        from codeflash.languages.javascript.module_system import ensure_module_system_compatibility

        # Code with both import (from inject_test_globals) and require (from backend)
        code = "import assert from 'node:assert/strict';\nconst { foo } = require('./module');\n"
        result = ensure_module_system_compatibility(code, "commonjs")
        assert "require('node:assert/strict')" in result
        assert "import assert" not in result

    def test_converts_require_in_mixed_code_to_esm(self):
        from codeflash.languages.javascript.module_system import ensure_module_system_compatibility

        code = "import { describe } from 'vitest';\nconst foo = require('./module');\n"
        result = ensure_module_system_compatibility(code, "esm")
        assert "require" not in result
        assert "import" in result

    def test_pure_esm_to_cjs(self):
        from codeflash.languages.javascript.module_system import ensure_module_system_compatibility

        code = "import assert from 'node:assert/strict';\nimport { foo } from './module';\n"
        result = ensure_module_system_compatibility(code, "commonjs")
        assert "require('node:assert/strict')" in result
        assert "import" not in result


class TestRunMochaLineProfileTests:
    """Tests for running Mocha line profile tests with mocked subprocess."""

    @patch("codeflash.languages.javascript.mocha_runner.subprocess.run")
    @patch("codeflash.languages.javascript.mocha_runner._ensure_runtime_files")
    def test_sets_line_profile_env_vars(self, mock_ensure, mock_run):
        from codeflash.languages.javascript.mocha_runner import run_mocha_line_profile_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        mocha_output = json.dumps(
            {"stats": {"tests": 0, "passes": 0, "failures": 0, "duration": 0}, "tests": [], "passes": [], "failures": [], "pending": []}
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=mocha_output, stderr="", args=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_file = tmpdir_path / "test.test.js"
            test_file.write_text("// test")
            profile_output = tmpdir_path / "profile.json"

            test_paths = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            run_mocha_line_profile_tests(
                test_paths=test_paths,
                test_env={},
                cwd=tmpdir_path,
                line_profile_output_file=profile_output,
            )

            call_kwargs = mock_run.call_args
            env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
            assert env.get("CODEFLASH_MODE") == "line_profile"
            assert env.get("CODEFLASH_LINE_PROFILE_OUTPUT") == str(profile_output)


class TestParserUnknownTestNameFallback:
    """Tests for the parser's fallback when perf markers have 'unknown' test name."""

    def test_unknown_markers_matched_to_first_testcase(self):
        """When capturePerf markers have 'unknown' test name (Vitest beforeEach not firing),
        the parser should still match them to testcases via the fallback logic."""
        from codeflash.languages.javascript.parse import parse_jest_test_xml
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a JUnit XML with one test suite and one testcase
            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="src/test_func__perf_test_0.test.ts" tests="1" failures="0" time="10.5">
    <testcase name="should compute correctly" classname="src/test_func__perf_test_0.test.ts" time="10.5">
    </testcase>
  </testsuite>
</testsuites>"""
            xml_path = tmpdir_path / "results.xml"
            xml_path.write_text(xml_content, encoding="utf-8")

            # Create test files
            test_file = tmpdir_path / "test_func__perf_test_0.test.ts"
            test_file.write_text("// perf test", encoding="utf-8")

            test_files = TestFiles(
                test_files=[
                    TestFile(
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            # Create a mock subprocess result with perf markers using "unknown" test name
            # This simulates what happens when Vitest's beforeEach doesn't fire
            markers = []
            for i in range(1, 6):
                markers.append(f"!######test_mod:unknown:computeFunc:{i}:1_0:{1000 + i * 100}######!")
            stdout = "\n".join(markers)

            mock_result = MagicMock()
            mock_result.stdout = stdout

            test_config = MagicMock()
            test_config.tests_project_rootdir = tmpdir_path
            test_config.test_framework = "vitest"

            results = parse_jest_test_xml(
                test_xml_file_path=xml_path,
                test_files=test_files,
                test_config=test_config,
                run_result=mock_result,
            )

            # The "unknown" fallback should assign all 5 markers to the testcase
            assert len(results.test_results) == 5
            # Verify runtimes were extracted (not the 10.5s XML fallback)
            runtimes = [r.runtime for r in results.test_results if r.runtime is not None]
            assert len(runtimes) == 5
            assert all(r < 100_000 for r in runtimes)  # All under 100 microseconds (nanoseconds)
