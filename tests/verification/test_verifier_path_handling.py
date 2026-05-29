"""Test that verifier.py handles test files outside tests_project_rootdir gracefully.

This tests the fix for the bug where JavaScript/TypeScript test files generated
in __tests__ subdirectories (adjacent to source files) caused ValueError when
verifier.py tried to compute their module path relative to tests_project_rootdir.

Trace ID: 84f5467f-8acf-427f-b468-02cb3342097e
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.models.function_types import FunctionToOptimize
from codeflash.verification.verifier import generate_tests


class TestVerifierPathHandling:
    """Test path handling in verifier.py for test files outside tests_root."""

    def test_module_name_from_file_path_raises_valueerror_when_outside_root(self) -> None:
        """Verify that module_name_from_file_path raises ValueError when file is outside root.

        This is the current behavior that causes the bug in verifier.py line 37.

        Scenario:
        - JavaScript support generates test at: /workspace/target/src/gateway/server/__tests__/codeflash-generated/test_foo.test.ts
        - tests_project_rootdir is: /workspace/target/test
        - Test file is NOT within tests_root, so relative_to() fails
        """
        test_path = Path("/workspace/target/src/gateway/server/__tests__/codeflash-generated/test_foo.test.ts")
        tests_root = Path("/workspace/target/test")

        # This should raise ValueError before the fix
        with pytest.raises(ValueError, match="is not within the project root"):
            module_name_from_file_path(test_path, tests_root)

    def test_module_name_from_file_path_with_fallback_succeeds(self) -> None:
        """Test that adding a fallback (try-except) allows graceful handling.

        This is the pattern used in javascript/parse.py:330-333 that should
        also be applied to verifier.py:37.
        """
        test_path = Path("/workspace/target/src/gateway/server/__tests__/codeflash-generated/test_foo.test.ts")
        tests_root = Path("/workspace/target/test")

        # Simulate the fix: try-except with fallback to filename
        try:
            test_module_path = module_name_from_file_path(test_path, tests_root)
        except ValueError:
            # Fallback: use just the filename (or relative path from parent)
            # This is what javascript/parse.py does
            test_module_path = test_path.name

        # After fallback, we should have a valid path
        assert test_module_path == "test_foo.test.ts"

    def test_generate_tests_uses_forward_slashes_for_javascript_module_paths(self, tmp_path: Path) -> None:
        """Generated JS import paths should stay valid on Windows by using forward slashes."""
        project_root = tmp_path / "project"
        source_dir = project_root / "src"
        source_dir.mkdir(parents=True)

        source_file = source_dir / "async_utils.js"
        source_file.write_text("export async function processItemsSequential() {}", encoding="utf-8")

        generated_tests_dir = source_dir / "__tests__" / "codeflash-generated"
        generated_tests_dir.mkdir(parents=True)
        test_path = generated_tests_dir / "test_processItemsSequential__unit_test_0.test.js"
        test_perf_path = generated_tests_dir / "test_processItemsSequential__perf_test_0.test.js"

        function_to_optimize = FunctionToOptimize(
            function_name="processItemsSequential", file_path=source_file, language="javascript"
        )
        test_cfg = MagicMock(tests_project_rootdir=project_root / "tests", test_framework="jest")
        ai_client = MagicMock()
        ai_client.generate_regression_tests.return_value = ("generated", "behavior", "perf", None)

        mock_support = MagicMock()
        mock_support.detect_module_system.return_value = "esm"
        mock_support.language_version = None
        mock_support.process_generated_test_strings.side_effect = lambda **kwargs: (
            kwargs["generated_test_source"],
            kwargs["instrumented_behavior_test_source"],
            kwargs["instrumented_perf_test_source"],
        )

        with patch("codeflash.verification.verifier.current_language_support", return_value=mock_support):
            result = generate_tests(
                aiservice_client=ai_client,
                source_code_being_tested=source_file.read_text(encoding="utf-8"),
                function_to_optimize=function_to_optimize,
                helper_function_names=[],
                module_path=source_file,
                test_cfg=test_cfg,
                test_timeout=30,
                function_trace_id="trace-id",
                test_index=0,
                test_path=test_path,
                test_perf_path=test_perf_path,
            )

        assert result is not None
        module_path = ai_client.generate_regression_tests.call_args.kwargs["module_path"]
        assert module_path == "../../async_utils.js"
        assert "\\" not in module_path
        assert not module_path.startswith("./..")
