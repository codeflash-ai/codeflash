"""Tests for JavaScript/Jest test runner functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestJestRootsConfiguration:
    """Tests for Jest --roots flag handling."""

    def test_behavioral_tests_adds_roots_for_test_directories(self):
        """Test that run_jest_behavioral_tests adds --roots for test directories."""
        from codeflash.languages.javascript.test_runner import run_jest_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        # Create mock test files in a test directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            # Create package.json to simulate a Node project
            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            # Create mock test files
            test_file1 = test_dir / "test_func__unit_test_0.test.ts"
            test_file2 = test_dir / "test_func__unit_test_1.test.ts"
            test_file1.write_text("// test 1")
            test_file2.write_text("// test 2")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file1,
                        instrumented_behavior_file_path=test_file1,
                        benchmarking_file_path=test_file1,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                    TestFile(
                        original_file_path=test_file2,
                        instrumented_behavior_file_path=test_file2,
                        benchmarking_file_path=test_file2,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            # Mock subprocess.run to capture the command
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_behavioral_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass  # Expected to fail since no real Jest

                # Verify the command included --roots
                if mock_run.called:
                    call_args = mock_run.call_args
                    cmd = call_args[0][0]

                    # Find --roots flags in the command
                    roots_flags = []
                    for i, arg in enumerate(cmd):
                        if arg == "--roots" and i + 1 < len(cmd):
                            roots_flags.append(cmd[i + 1])

                    # Should have added the test directory as a root
                    assert len(roots_flags) > 0, "Expected --roots flag in Jest command"
                    assert str(test_dir) in roots_flags or any(
                        str(test_dir) in root for root in roots_flags
                    ), f"Expected test directory {test_dir} in --roots flags: {roots_flags}"

    def test_benchmarking_tests_adds_roots_for_test_directories(self):
        """Test that run_jest_benchmarking_tests adds --roots for test directories."""
        from codeflash.languages.javascript.test_runner import run_jest_benchmarking_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            test_file = test_dir / "test_func__perf_test_0.test.ts"
            test_file.write_text("// perf test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_benchmarking_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    call_args = mock_run.call_args
                    cmd = call_args[0][0]

                    roots_flags = []
                    for i, arg in enumerate(cmd):
                        if arg == "--roots" and i + 1 < len(cmd):
                            roots_flags.append(cmd[i + 1])

                    assert len(roots_flags) > 0, "Expected --roots flag in Jest command"

    def test_line_profile_tests_adds_roots_for_test_directories(self):
        """Test that run_jest_line_profile_tests adds --roots for test directories."""
        from codeflash.languages.javascript.test_runner import run_jest_line_profile_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            test_file = test_dir / "test_func__line_profile.test.ts"
            test_file.write_text("// line profile test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_line_profile_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    call_args = mock_run.call_args
                    cmd = call_args[0][0]

                    roots_flags = []
                    for i, arg in enumerate(cmd):
                        if arg == "--roots" and i + 1 < len(cmd):
                            roots_flags.append(cmd[i + 1])

                    assert len(roots_flags) > 0, "Expected --roots flag in Jest command"

    def test_multiple_test_directories_all_added_to_roots(self):
        """Test that multiple test directories are all added as --roots."""
        from codeflash.languages.javascript.test_runner import run_jest_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir1 = tmpdir_path / "test"
            test_dir2 = tmpdir_path / "spec"
            test_dir1.mkdir()
            test_dir2.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            test_file1 = test_dir1 / "test_func__unit_test_0.test.ts"
            test_file2 = test_dir2 / "test_func__unit_test_1.test.ts"
            test_file1.write_text("// test 1")
            test_file2.write_text("// test 2")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file1,
                        instrumented_behavior_file_path=test_file1,
                        benchmarking_file_path=test_file1,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                    TestFile(
                        original_file_path=test_file2,
                        instrumented_behavior_file_path=test_file2,
                        benchmarking_file_path=test_file2,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_behavioral_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    call_args = mock_run.call_args
                    cmd = call_args[0][0]

                    roots_flags = []
                    for i, arg in enumerate(cmd):
                        if arg == "--roots" and i + 1 < len(cmd):
                            roots_flags.append(cmd[i + 1])

                    # Should have two --roots flags (one for each directory)
                    assert len(roots_flags) == 2, f"Expected 2 --roots flags, got {len(roots_flags)}"


class TestVitestTimeoutConfiguration:
    """Tests for Vitest subprocess timeout handling."""

    def test_vitest_behavioral_subprocess_timeout_larger_than_test_timeout(self):
        """Test that subprocess timeout is larger than per-test timeout for Vitest behavioral tests."""
        from codeflash.languages.javascript.vitest_runner import run_vitest_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"vitest": "^1.0.0"}}')

            test_file = test_dir / "test_func.test.ts"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                # Run with a 15 second per-test timeout
                run_vitest_behavioral_tests(
                    test_paths=mock_test_files,
                    test_env={},
                    cwd=tmpdir_path,
                    timeout=15,  # 15 second per-test timeout
                    project_root=tmpdir_path,
                )

                # Verify subprocess was called with a larger timeout
                assert mock_run.called
                call_kwargs = mock_run.call_args[1]
                subprocess_timeout = call_kwargs.get("timeout")

                # Subprocess timeout should be at least 120 seconds (minimum)
                # or 10x the per-test timeout (150 seconds)
                assert subprocess_timeout >= 120, f"Expected subprocess timeout >= 120s, got {subprocess_timeout}s"
                assert subprocess_timeout >= 15 * 10, f"Expected subprocess timeout >= 150s (10x per-test), got {subprocess_timeout}s"

    def test_vitest_line_profile_subprocess_timeout_larger_than_test_timeout(self):
        """Test that subprocess timeout is larger than per-test timeout for Vitest line profile tests."""
        from codeflash.languages.javascript.vitest_runner import run_vitest_line_profile_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"vitest": "^1.0.0"}}')

            test_file = test_dir / "test_func.test.ts"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                run_vitest_line_profile_tests(
                    test_paths=mock_test_files,
                    test_env={},
                    cwd=tmpdir_path,
                    timeout=15,
                    project_root=tmpdir_path,
                )

                assert mock_run.called
                call_kwargs = mock_run.call_args[1]
                subprocess_timeout = call_kwargs.get("timeout")

                assert subprocess_timeout >= 120, f"Expected subprocess timeout >= 120s, got {subprocess_timeout}s"

    def test_vitest_default_subprocess_timeout_is_reasonable(self):
        """Test that default subprocess timeout is at least 120 seconds when no timeout specified."""
        from codeflash.languages.javascript.vitest_runner import run_vitest_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"vitest": "^1.0.0"}}')

            test_file = test_dir / "test_func.test.ts"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                # Run without specifying a timeout
                run_vitest_behavioral_tests(
                    test_paths=mock_test_files,
                    test_env={},
                    cwd=tmpdir_path,
                    project_root=tmpdir_path,
                )

                assert mock_run.called
                call_kwargs = mock_run.call_args[1]
                subprocess_timeout = call_kwargs.get("timeout")

                # Default should be at least 120 seconds (or 600 from the default)
                assert subprocess_timeout >= 120, f"Expected subprocess timeout >= 120s, got {subprocess_timeout}s"


class TestVitestInternalLoopingConfiguration:
    """Tests for Vitest internal looping (no external loop-runner)."""

    def test_vitest_benchmarking_does_not_set_current_batch_env(self):
        """Test that Vitest runner does NOT set CODEFLASH_PERF_CURRENT_BATCH.

        This is critical: when CODEFLASH_PERF_CURRENT_BATCH is not set,
        capturePerf() in the npm package will do all loops internally
        (PERF_LOOP_COUNT iterations) instead of just PERF_BATCH_SIZE.
        """
        from codeflash.languages.javascript.vitest_runner import run_vitest_benchmarking_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"vitest": "^1.0.0"}}')

            test_file = test_dir / "test_func.test.ts"
            test_file.write_text("// perf test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                run_vitest_benchmarking_tests(
                    test_paths=mock_test_files,
                    test_env={},
                    cwd=tmpdir_path,
                    project_root=tmpdir_path,
                    max_loops=100,
                    min_loops=5,
                )

                assert mock_run.called
                call_kwargs = mock_run.call_args[1]
                env = call_kwargs.get("env", {})

                # CODEFLASH_PERF_CURRENT_BATCH should NOT be set
                # This allows capturePerf() to do all loops internally
                assert "CODEFLASH_PERF_CURRENT_BATCH" not in env, (
                    "CODEFLASH_PERF_CURRENT_BATCH should not be set for Vitest - "
                    "internal looping relies on this being undefined"
                )

                # But CODEFLASH_PERF_LOOP_COUNT should be set
                assert "CODEFLASH_PERF_LOOP_COUNT" in env, "CODEFLASH_PERF_LOOP_COUNT should be set"
                assert env["CODEFLASH_PERF_LOOP_COUNT"] == "100"

    def test_vitest_benchmarking_sets_loop_configuration_env_vars(self):
        """Test that Vitest benchmarking sets correct loop configuration environment variables."""
        from codeflash.languages.javascript.vitest_runner import run_vitest_benchmarking_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()

            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"vitest": "^1.0.0"}}')

            test_file = test_dir / "test_func.test.ts"
            test_file.write_text("// perf test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                run_vitest_benchmarking_tests(
                    test_paths=mock_test_files,
                    test_env={},
                    cwd=tmpdir_path,
                    project_root=tmpdir_path,
                    max_loops=50,
                    min_loops=10,
                    target_duration_ms=5000,
                    stability_check=True,
                )

                assert mock_run.called
                call_kwargs = mock_run.call_args[1]
                env = call_kwargs.get("env", {})

                # Verify all loop configuration env vars are set correctly
                assert env.get("CODEFLASH_PERF_LOOP_COUNT") == "50"
                assert env.get("CODEFLASH_PERF_MIN_LOOPS") == "10"
                assert env.get("CODEFLASH_PERF_TARGET_DURATION_MS") == "5000"
                assert env.get("CODEFLASH_PERF_STABILITY_CHECK") == "true"
                assert env.get("CODEFLASH_MODE") == "performance"


class TestBundlerModuleResolutionFix:
    """Tests for bundler moduleResolution compatibility fix."""

    def test_detect_bundler_module_resolution_true(self):
        """Test detection of bundler moduleResolution in tsconfig."""
        import json

        from codeflash.languages.javascript.test_runner import _detect_bundler_module_resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create tsconfig with bundler moduleResolution
            tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "bundler",
                    "module": "preserve",
                    "target": "ES2022",
                }
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(tsconfig))

            assert _detect_bundler_module_resolution(tmpdir_path) is True

    def test_detect_bundler_module_resolution_false(self):
        """Test detection returns false for Node moduleResolution."""
        import json

        from codeflash.languages.javascript.test_runner import _detect_bundler_module_resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create tsconfig with Node moduleResolution
            tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "Node",
                    "module": "ESNext",
                }
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(tsconfig))

            assert _detect_bundler_module_resolution(tmpdir_path) is False

    def test_detect_bundler_module_resolution_no_tsconfig(self):
        """Test detection returns false when no tsconfig exists."""
        from codeflash.languages.javascript.test_runner import _detect_bundler_module_resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            assert _detect_bundler_module_resolution(tmpdir_path) is False

    def test_detect_bundler_module_resolution_extended_config(self):
        """Test detection works with extended tsconfig files."""
        import json

        from codeflash.languages.javascript.test_runner import _detect_bundler_module_resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a base config with bundler in a subdirectory (simulating node_modules)
            node_modules = tmpdir_path / "node_modules" / "@myorg" / "tsconfig"
            node_modules.mkdir(parents=True)
            base_tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "bundler",
                    "module": "preserve",
                }
            }
            (node_modules / "tsconfig.json").write_text(json.dumps(base_tsconfig))

            # Create a project tsconfig that extends the base
            project_tsconfig = {
                "extends": "@myorg/tsconfig/tsconfig.json",
                "compilerOptions": {
                    "target": "ES2022",
                }
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(project_tsconfig))

            # Should detect bundler from extended config
            assert _detect_bundler_module_resolution(tmpdir_path) is True

    def test_create_codeflash_tsconfig(self):
        """Test creation of codeflash-compatible tsconfig."""
        import json

        from codeflash.languages.javascript.test_runner import _create_codeflash_tsconfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create original tsconfig
            original_tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "bundler",
                    "module": "preserve",
                    "target": "ES2022",
                },
                "include": ["src/**/*.ts"],
                "exclude": ["node_modules"],
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(original_tsconfig))

            # Create codeflash tsconfig
            result_path = _create_codeflash_tsconfig(tmpdir_path)

            assert result_path.exists()
            assert result_path.name == "tsconfig.codeflash.json"

            # Verify contents
            codeflash_tsconfig = json.loads(result_path.read_text())
            assert codeflash_tsconfig["extends"] == "./tsconfig.json"
            assert codeflash_tsconfig["compilerOptions"]["moduleResolution"] == "Node"
            assert "include" in codeflash_tsconfig

    def test_create_codeflash_jest_config(self):
        """Test creation of codeflash Jest config."""
        from codeflash.languages.javascript.test_runner import _create_codeflash_jest_config

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create codeflash Jest config without original
            result_path = _create_codeflash_jest_config(tmpdir_path, None)

            assert result_path is not None
            assert result_path.exists()
            assert result_path.name == "jest.codeflash.config.js"

            # Verify it contains ESM package transformation patterns
            content = result_path.read_text()
            assert "transformIgnorePatterns" in content
            assert "node_modules" in content

    def test_get_jest_config_for_project_with_bundler(self):
        """Test that bundler projects get codeflash Jest config."""
        import json

        from codeflash.languages.javascript.test_runner import _get_jest_config_for_project

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create tsconfig with bundler
            tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "bundler",
                    "module": "preserve",
                }
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(tsconfig))
            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            result = _get_jest_config_for_project(tmpdir_path)

            assert result is not None
            assert result.name == "jest.codeflash.config.js"
            # Also verify tsconfig.codeflash.json was created
            assert (tmpdir_path / "tsconfig.codeflash.json").exists()

    def test_get_jest_config_for_project_without_bundler(self):
        """Test that non-bundler projects use original Jest config."""
        import json

        from codeflash.languages.javascript.test_runner import _get_jest_config_for_project

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create tsconfig with Node moduleResolution
            tsconfig = {
                "compilerOptions": {
                    "moduleResolution": "Node",
                    "module": "ESNext",
                }
            }
            (tmpdir_path / "tsconfig.json").write_text(json.dumps(tsconfig))
            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            # Create original Jest config
            (tmpdir_path / "jest.config.js").write_text("module.exports = {};")

            result = _get_jest_config_for_project(tmpdir_path)

            assert result is not None
            assert result.name == "jest.config.js"
            # Verify codeflash configs were NOT created
            assert not (tmpdir_path / "jest.codeflash.config.js").exists()
            assert not (tmpdir_path / "tsconfig.codeflash.json").exists()


class TestBundledJestReporter:
    """Tests for the bundled codeflash/jest-reporter.

    Verifies that:
    1. The reporter JS file exists in the runtime package
    2. Jest commands reference 'codeflash/jest-reporter' (not jest-junit)
    3. The reporter produces valid JUnit XML
    4. The CODEFLASH_JEST_REPORTER constant is correct
    """

    def test_reporter_js_file_exists(self):
        """The jest-reporter.js file must exist in the runtime directory."""
        reporter_path = Path(__file__).resolve().parents[2] / "packages" / "codeflash" / "runtime" / "jest-reporter.js"
        assert reporter_path.exists(), f"jest-reporter.js not found at {reporter_path}"

    def test_reporter_constant_value(self):
        """CODEFLASH_JEST_REPORTER should be 'codeflash/jest-reporter'."""
        from codeflash.languages.javascript.test_runner import CODEFLASH_JEST_REPORTER

        assert CODEFLASH_JEST_REPORTER == "codeflash/jest-reporter"

    def test_behavioral_command_uses_bundled_reporter(self):
        """run_jest_behavioral_tests should use codeflash/jest-reporter in --reporters flag."""
        from codeflash.languages.javascript.test_runner import run_jest_behavioral_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()
            test_file = test_dir / "test_func.test.js"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_behavioral_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    cmd = mock_run.call_args[0][0]
                    reporter_args = [a for a in cmd if "--reporters=" in a and "jest-reporter" in a]
                    assert len(reporter_args) == 1, f"Expected exactly one codeflash/jest-reporter flag, got: {reporter_args}"
                    assert reporter_args[0] == "--reporters=codeflash/jest-reporter"
                    # Must NOT reference jest-junit
                    jest_junit_args = [a for a in cmd if "jest-junit" in a]
                    assert len(jest_junit_args) == 0, f"Should not reference jest-junit: {jest_junit_args}"

    def test_benchmarking_command_uses_bundled_reporter(self):
        """run_jest_benchmarking_tests should use codeflash/jest-reporter."""
        from codeflash.languages.javascript.test_runner import run_jest_benchmarking_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()
            test_file = test_dir / "test_func__perf.test.js"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_benchmarking_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    cmd = mock_run.call_args[0][0]
                    reporter_args = [a for a in cmd if "--reporters=codeflash/jest-reporter" in a]
                    assert len(reporter_args) == 1

    def test_line_profile_command_uses_bundled_reporter(self):
        """run_jest_line_profile_tests should use codeflash/jest-reporter."""
        from codeflash.languages.javascript.test_runner import run_jest_line_profile_tests
        from codeflash.models.models import TestFile, TestFiles
        from codeflash.models.test_type import TestType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").write_text('{"name": "test"}')
            test_dir = tmpdir_path / "test"
            test_dir.mkdir()
            test_file = test_dir / "test_func__line.test.js"
            test_file.write_text("// test")

            mock_test_files = TestFiles(
                test_files=[
                    TestFile(
                        original_file_path=test_file,
                        instrumented_behavior_file_path=test_file,
                        benchmarking_file_path=test_file,
                        test_type=TestType.GENERATED_REGRESSION,
                    ),
                ]
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = ""
                mock_result.stderr = ""
                mock_result.returncode = 1
                mock_run.return_value = mock_result

                try:
                    run_jest_line_profile_tests(
                        test_paths=mock_test_files,
                        test_env={},
                        cwd=tmpdir_path,
                        project_root=tmpdir_path,
                    )
                except Exception:
                    pass

                if mock_run.called:
                    cmd = mock_run.call_args[0][0]
                    reporter_args = [a for a in cmd if "--reporters=codeflash/jest-reporter" in a]
                    assert len(reporter_args) == 1

    def test_reporter_produces_valid_junit_xml(self):
        """The reporter JS should produce JUnit XML parseable by junitparser."""
        import subprocess

        reporter_path = Path(__file__).resolve().parents[2] / "packages" / "codeflash" / "runtime" / "jest-reporter.js"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.xml"

            # Create a Node.js script that exercises the reporter with mock data
            test_script = Path(tmpdir) / "test_reporter.js"
            test_script.write_text(f"""
// Set env vars BEFORE requiring reporter (matches real Jest behavior)
process.env.JEST_JUNIT_OUTPUT_FILE = '{output_file.as_posix()}';
process.env.JEST_JUNIT_CLASSNAME = '{{filepath}}';
process.env.JEST_JUNIT_SUITE_NAME = '{{filepath}}';
process.env.JEST_JUNIT_ADD_FILE_ATTRIBUTE = 'true';
process.env.JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT = 'true';

const Reporter = require('{reporter_path.as_posix()}');

// Mock Jest globalConfig
const globalConfig = {{ rootDir: '/tmp/project' }};
const reporter = new Reporter(globalConfig, {{}});

// Mock test results (matches Jest's aggregatedResults structure)
const results = {{
  testResults: [
    {{
      testFilePath: '/tmp/project/test/math.test.js',
      displayName: 'math tests',
      console: [{{ type: 'log', message: 'CODEFLASH_START test1' }}],
      testResults: [
        {{
          fullName: 'math > adds numbers',
          title: 'adds numbers',
          status: 'passed',
          duration: 12,
        }},
        {{
          fullName: 'math > handles failure',
          title: 'handles failure',
          status: 'failed',
          duration: 5,
          failureMessages: ['Expected 4 but got 5'],
        }},
        {{
          fullName: 'math > skipped test',
          title: 'skipped test',
          status: 'pending',
          duration: 0,
        }},
      ],
    }},
  ],
}};

// Simulate onTestFileResult for console capture
reporter.onTestFileResult(null, results.testResults[0], null);

// Simulate onRunComplete
reporter.onRunComplete([], results);

console.log('OK');
""")

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0, f"Reporter script failed: {result.stderr}"
            assert output_file.exists(), "Reporter did not create output file"

            xml_content = output_file.read_text()

            # Verify basic XML structure
            assert '<?xml version="1.0"' in xml_content
            assert "<testsuites" in xml_content
            assert "<testsuite" in xml_content
            assert "<testcase" in xml_content

            # Verify classname uses filepath template
            assert 'classname="/tmp/project/test/math.test.js"' in xml_content

            # Verify file attribute is present
            assert 'file="/tmp/project/test/math.test.js"' in xml_content

            # Verify failure element
            assert "<failure" in xml_content
            assert "Expected 4 but got 5" in xml_content

            # Verify skipped element
            assert "<skipped/>" in xml_content

            # Verify system-out with console output
            assert "<system-out>" in xml_content
            assert "CODEFLASH_START" in xml_content

            # Verify it's parseable by junitparser (our actual parser)
            from junitparser import JUnitXml

            parsed = JUnitXml.fromfile(str(output_file))
            suites = list(parsed)
            assert len(suites) == 1
            testcases = list(suites[0])
            assert len(testcases) == 3

    def test_reporter_export_in_package_json(self):
        """package.json should export codeflash/jest-reporter."""
        import json

        pkg_path = Path(__file__).resolve().parents[2] / "packages" / "codeflash" / "package.json"
        with pkg_path.open() as f:
            pkg = json.load(f)

        exports = pkg.get("exports", {})
        assert "./jest-reporter" in exports, "Missing ./jest-reporter export in package.json"
        assert exports["./jest-reporter"]["require"] == "./runtime/jest-reporter.js"



class TestUnsupportedFrameworkError:
    """Tests for clear error on unsupported test frameworks."""

    def test_unknown_framework_raises_error_behavioral(self):
        """run_behavioral_tests should raise NotImplementedError for unknown frameworks."""
        from codeflash.languages.javascript.support import JavaScriptSupport

        support = JavaScriptSupport()
        with pytest.raises(NotImplementedError, match="not yet supported"):
            support.run_behavioral_tests(
                test_paths=MagicMock(),
                test_env={},
                cwd=Path("."),
                test_framework="tap",
            )

    def test_unknown_framework_raises_error_benchmarking(self):
        """run_benchmarking_tests should raise NotImplementedError for unknown frameworks."""
        from codeflash.languages.javascript.support import JavaScriptSupport

        support = JavaScriptSupport()
        with pytest.raises(NotImplementedError, match="not yet supported"):
            support.run_benchmarking_tests(
                test_paths=MagicMock(),
                test_env={},
                cwd=Path("."),
                test_framework="tap",
            )

    def test_unknown_framework_raises_error_line_profile(self):
        """run_line_profile_tests should raise NotImplementedError for unknown frameworks."""
        from codeflash.languages.javascript.support import JavaScriptSupport

        support = JavaScriptSupport()
        with pytest.raises(NotImplementedError, match="not yet supported"):
            support.run_line_profile_tests(
                test_paths=MagicMock(),
                test_env={},
                cwd=Path("."),
                test_framework="tap",
            )

    def test_jest_framework_does_not_raise_not_implemented(self):
        """jest framework should NOT raise NotImplementedError."""
        from codeflash.languages.javascript.support import JavaScriptSupport

        support = JavaScriptSupport()
        try:
            support.run_behavioral_tests(
                test_paths=MagicMock(),
                test_env={},
                cwd=Path("."),
                test_framework="jest",
            )
        except NotImplementedError:
            pytest.fail("jest framework should not raise NotImplementedError")
        except Exception:
            pass  # Other exceptions are fine — Jest isn't installed in test env

    def test_mocha_framework_does_not_raise_not_implemented(self):
        """mocha framework should NOT raise NotImplementedError."""
        from codeflash.languages.javascript.support import JavaScriptSupport

        support = JavaScriptSupport()
        try:
            support.run_behavioral_tests(
                test_paths=MagicMock(),
                test_env={},
                cwd=Path("."),
                test_framework="mocha",
            )
        except NotImplementedError:
            pytest.fail("mocha framework should not raise NotImplementedError")
        except Exception:
            pass  # Other exceptions are fine — Mocha isn't installed in test env
