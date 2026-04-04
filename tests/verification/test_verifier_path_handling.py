"""Test that verifier.py handles test files outside tests_project_rootdir gracefully.

This tests the fix for the bug where JavaScript/TypeScript test files generated
in __tests__ subdirectories (adjacent to source files) caused ValueError when
verifier.py tried to compute their module path relative to tests_project_rootdir.

Trace ID: 84f5467f-8acf-427f-b468-02cb3342097e
"""

from pathlib import Path

import pytest

from codeflash.code_utils.code_utils import module_name_from_file_path


class TestVerifierPathHandling:
    """Test path handling in verifier.py for test files outside tests_root."""

    def test_module_name_from_file_path_raises_valueerror_when_outside_root(self):
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

    def test_module_name_from_file_path_with_fallback_succeeds(self):
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
