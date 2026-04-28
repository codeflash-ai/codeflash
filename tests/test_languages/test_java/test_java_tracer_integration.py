"""End-to-end integration test for the Java tracer → optimizer pipeline.

Tests the full flow: trace → replay test generation → function discovery →
test discovery → function ranking, using the simple Workload fixture.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codeflash.languages.java.tracer import run_java_tracer

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "java_tracer_e2e"
WORKLOAD_SOURCE = FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Workload.java"
WORKLOAD_CLASS = "com.example.Workload"
WORKLOAD_PACKAGE = "com.example"


@pytest.fixture(scope="module")
def compiled_workload() -> Path:
    classes_dir = FIXTURE_DIR / "target" / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["javac", "--release", "11", "-d", str(classes_dir), str(WORKLOAD_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"javac failed: {result.stderr}"
    return classes_dir


@pytest.fixture
def traced_workload(compiled_workload: Path, tmp_path: Path) -> tuple[Path, Path, Path, int]:
    """Trace the workload and generate replay tests. Returns (trace_db, jfr_file, output_dir, test_count)."""
    trace_db_path = tmp_path / "trace.db"
    output_dir = tmp_path / "replay_tests"

    trace_db, jfr_file, test_count = run_java_tracer(
        java_command=["java", "-cp", str(compiled_workload), WORKLOAD_CLASS],
        trace_db_path=trace_db_path,
        packages=[WORKLOAD_PACKAGE],
        project_root=FIXTURE_DIR,
        output_dir=output_dir,
    )

    assert trace_db.exists(), "Trace DB not created"
    assert test_count >= 1, f"Expected at least 1 replay test file, got {test_count}"
    return trace_db, jfr_file, output_dir, test_count


class TestFunctionDiscoveryFromReplayTests:
    """Test that functions are correctly discovered from replay test metadata."""

    def test_discover_functions_from_replay_tests(self, traced_workload: tuple) -> None:
        _trace_db, _jfr_file, output_dir, _test_count = traced_workload

        from codeflash.discovery.functions_to_optimize import _get_java_replay_test_functions
        from codeflash.verification.verification_utils import TestConfig

        replay_test_paths = list(output_dir.glob("*.java"))
        assert len(replay_test_paths) >= 1

        test_cfg = TestConfig(
            tests_root=FIXTURE_DIR / "src" / "test" / "java",
            tests_project_rootdir=FIXTURE_DIR,
            project_root_path=FIXTURE_DIR,
            pytest_cmd="pytest",
        )

        functions, trace_file_path = _get_java_replay_test_functions(replay_test_paths, test_cfg, FIXTURE_DIR)

        # Should have found functions in the Workload source file
        assert len(functions) > 0, "No functions discovered from replay tests"
        assert trace_file_path.exists(), f"Trace file not found: {trace_file_path}"

        # Collect all discovered function names
        all_func_names = set()
        for file_path, func_list in functions.items():
            assert file_path.exists(), f"Source file not found: {file_path}"
            assert "Workload" in file_path.name
            for func in func_list:
                all_func_names.add(func.function_name)
                assert func.language == "java", f"Expected language='java', got '{func.language}'"
                assert func.file_path == file_path

        assert "repeatString" in all_func_names

    def test_discover_tests_for_replay_tests(self, traced_workload: tuple) -> None:
        """Test that test discovery maps replay tests to source functions."""
        _trace_db, _jfr_file, output_dir, _test_count = traced_workload

        from codeflash.languages.java.discovery import discover_functions_from_source
        from codeflash.languages.java.test_discovery import discover_tests

        source_code = WORKLOAD_SOURCE.read_text(encoding="utf-8")
        source_functions = discover_functions_from_source(source_code, file_path=WORKLOAD_SOURCE)

        result = discover_tests(output_dir, source_functions)

        # Replay tests should be mapped to source functions
        assert len(result) > 0, "No test mappings found from replay tests"

        # Check specific functions are mapped
        matched_func_names = set()
        for qualified_name in result:
            func_name = qualified_name.split(".")[-1] if "." in qualified_name else qualified_name
            matched_func_names.add(func_name)

        assert "repeatString" in matched_func_names, f"repeatString not found in: {result.keys()}"

        # Each function should have at least one test
        for func_name, test_infos in result.items():
            assert len(test_infos) > 0, f"No tests for {func_name}"
            for test_info in test_infos:
                assert test_info.test_file.exists()
                assert "ReplayTest" in test_info.test_file.name


class TestJfrProfiling:
    """Test JFR profiling and function ranking."""

    def test_jfr_parsing(self, traced_workload: tuple) -> None:
        _trace_db, jfr_file, _output_dir, _test_count = traced_workload

        if not jfr_file.exists():
            pytest.skip("JFR file not created (JFR may not be available)")

        from codeflash.languages.java.jfr_parser import JfrProfile

        profile = JfrProfile(jfr_file, [WORKLOAD_PACKAGE])
        ranking = profile.get_method_ranking()

        # The workload is very short, so JFR might not capture many samples
        # Just verify the parser doesn't crash and returns a list
        assert isinstance(ranking, list)

    def test_java_function_ranker(self, traced_workload: tuple) -> None:
        _trace_db, jfr_file, _output_dir, _test_count = traced_workload

        if not jfr_file.exists():
            pytest.skip("JFR file not created (JFR may not be available)")

        from codeflash.benchmarking.function_ranker import JavaFunctionRanker
        from codeflash.languages.java.discovery import discover_functions_from_source
        from codeflash.languages.java.jfr_parser import JfrProfile

        profile = JfrProfile(jfr_file, [WORKLOAD_PACKAGE])
        ranker = JavaFunctionRanker(profile)

        source_code = WORKLOAD_SOURCE.read_text(encoding="utf-8")
        source_functions = discover_functions_from_source(source_code, file_path=WORKLOAD_SOURCE)

        # Rank functions - should not crash even with minimal JFR data
        ranked = ranker.rank_functions(source_functions)
        assert isinstance(ranked, list)


class TestFullDiscoveryPipeline:
    """Test the complete discovery pipeline as the optimizer would run it."""

    def test_full_pipeline(self, compiled_workload: Path, tmp_path: Path) -> None:
        """Simulate what optimizer.run() does: discover functions, discover tests, rank.

        Uses the same directory layout as the real flow: replay tests go into
        src/test/java/codeflash/replay/ so test discovery can find them.
        """
        trace_db_path = tmp_path / "trace.db"

        # Generate replay tests into the project's test source tree (like _run_java_tracer does)
        test_root = FIXTURE_DIR / "src" / "test" / "java"
        output_dir = test_root / "codeflash" / "replay"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            _trace_db, jfr_file, test_count = run_java_tracer(
                java_command=["java", "-cp", str(compiled_workload), WORKLOAD_CLASS],
                trace_db_path=trace_db_path,
                packages=[WORKLOAD_PACKAGE],
                project_root=FIXTURE_DIR,
                output_dir=output_dir,
            )
            assert test_count >= 1

            # Step 1: Discover functions from replay tests (like get_optimizable_functions)
            from codeflash.discovery.functions_to_optimize import _get_java_replay_test_functions
            from codeflash.verification.verification_utils import TestConfig

            replay_test_paths = list(output_dir.glob("*.java"))
            test_cfg = TestConfig(
                tests_root=test_root,
                tests_project_rootdir=FIXTURE_DIR,
                project_root_path=FIXTURE_DIR,
                pytest_cmd="pytest",
            )

            file_to_funcs, trace_file_path = _get_java_replay_test_functions(replay_test_paths, test_cfg, FIXTURE_DIR)
            assert len(file_to_funcs) > 0
            assert trace_file_path.exists()

            # Step 2: Set language (like optimizer.run lines 496-502)
            from codeflash.languages import set_current_language
            from codeflash.languages.base import Language

            set_current_language(Language.JAVA)

            # Step 3: Discover tests (like optimizer.discover_tests)
            from codeflash.discovery.discover_unit_tests import discover_tests_for_language

            all_functions = [func for funcs in file_to_funcs.values() for func in funcs]
            function_to_tests, num_unit_tests, num_replay_tests = discover_tests_for_language(
                test_cfg, "java", file_to_funcs
            )

            assert num_unit_tests + num_replay_tests > 0, "No tests discovered"
            assert num_replay_tests > 0, f"Expected replay tests, got {num_replay_tests}"
            assert len(function_to_tests) > 0, "No function-to-test mappings"

            # Verify function_to_tests has entries for our traced functions
            has_repeat_string = any("repeatString" in key for key in function_to_tests)
            assert has_repeat_string, f"repeatString not in function_to_tests keys: {list(function_to_tests.keys())}"

            # Step 4: Rank functions (like optimizer.rank_all_functions_globally)
            if jfr_file.exists():
                from codeflash.benchmarking.function_ranker import JavaFunctionRanker
                from codeflash.languages.java.jfr_parser import JfrProfile

                packages = set()
                for func in all_functions:
                    parts = func.qualified_name.split(".")
                    if len(parts) >= 2:
                        packages.add(".".join(parts[:-1]))

                profile = JfrProfile(jfr_file, list(packages))
                ranker = JavaFunctionRanker(profile)
                ranked = ranker.rank_functions(all_functions)
                assert isinstance(ranked, list)

        finally:
            # Clean up generated replay tests from fixture directory
            for f in output_dir.glob("*.java"):
                f.unlink()
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            codeflash_dir = output_dir.parent
            if codeflash_dir.exists() and codeflash_dir.name == "codeflash" and not any(codeflash_dir.iterdir()):
                codeflash_dir.rmdir()

    def test_instrument_and_compile_replay_tests(self, compiled_workload: Path, tmp_path: Path) -> None:
        """Test that replay tests can be instrumented and compiled by Maven."""
        trace_db_path = tmp_path / "trace.db"

        test_root = FIXTURE_DIR / "src" / "test" / "java"
        output_dir = test_root / "codeflash" / "replay"
        output_dir.mkdir(parents=True, exist_ok=True)

        cleanup_paths: list[Path] = []
        try:
            _trace_db, _jfr_file, test_count = run_java_tracer(
                java_command=["java", "-cp", str(compiled_workload), WORKLOAD_CLASS],
                trace_db_path=trace_db_path,
                packages=[WORKLOAD_PACKAGE],
                project_root=FIXTURE_DIR,
                output_dir=output_dir,
            )
            assert test_count >= 1

            replay_test_paths = list(output_dir.glob("*.java"))
            cleanup_paths.extend(replay_test_paths)

            # Instrument a replay test (like instrument_existing_tests does)
            from codeflash.languages.java.discovery import discover_functions_from_source
            from codeflash.languages.java.instrumentation import instrument_existing_test

            source_code = WORKLOAD_SOURCE.read_text(encoding="utf-8")
            source_functions = discover_functions_from_source(source_code, file_path=WORKLOAD_SOURCE)
            # Pick the first function with a return type for instrumentation
            target_func = next(f for f in source_functions if f.function_name == "repeatString")

            replay_test_file = replay_test_paths[0]
            test_source = replay_test_file.read_text(encoding="utf-8")

            # Instrument for behavior mode
            success, instrumented_source = instrument_existing_test(
                test_string=test_source, function_to_optimize=target_func, mode="behavior", test_path=replay_test_file
            )
            assert success, "Failed to instrument replay test for behavior mode"
            assert instrumented_source is not None
            assert "__perfinstrumented" in instrumented_source

            # Write the instrumented test
            instrumented_path = replay_test_file.parent / f"{replay_test_file.stem}__perfinstrumented.java"
            instrumented_path.write_text(instrumented_source, encoding="utf-8")
            cleanup_paths.append(instrumented_path)

            # Instrument for performance mode
            success, perf_source = instrument_existing_test(
                test_string=test_source,
                function_to_optimize=target_func,
                mode="performance",
                test_path=replay_test_file,
            )
            assert success, "Failed to instrument replay test for performance mode"
            assert perf_source is not None

            perf_path = replay_test_file.parent / f"{replay_test_file.stem}__perfonlyinstrumented.java"
            perf_path.write_text(perf_source, encoding="utf-8")
            cleanup_paths.append(perf_path)

            # Install codeflash-runtime as Maven dependency and compile
            from codeflash.languages.java.build_tool_strategy import get_strategy

            strategy = get_strategy(FIXTURE_DIR)
            strategy.ensure_runtime(FIXTURE_DIR, None)

            import os

            compile_env = os.environ.copy()
            compile_result = strategy.compile_tests(FIXTURE_DIR, compile_env, None, timeout=120)

            assert compile_result.returncode == 0, (
                f"Maven compilation failed (rc={compile_result.returncode}):\n"
                f"stdout: {compile_result.stdout}\n"
                f"stderr: {compile_result.stderr}"
            )

        finally:
            for f in cleanup_paths:
                f.unlink(missing_ok=True)
            # Also clean up Maven build artifacts for the replay package
            replay_classes = FIXTURE_DIR / "target" / "test-classes" / "codeflash"
            if replay_classes.exists():
                import shutil

                shutil.rmtree(replay_classes, ignore_errors=True)
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            codeflash_dir = output_dir.parent
            if codeflash_dir.exists() and codeflash_dir.name == "codeflash" and not any(codeflash_dir.iterdir()):
                codeflash_dir.rmdir()
