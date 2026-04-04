"""Tests for module_name_from_file_path with co-located test directories."""

import pytest
from pathlib import Path
from codeflash.code_utils.code_utils import module_name_from_file_path


class TestModuleNameFromFilePath:
    """Test module name resolution for various directory structures."""

    def test_file_inside_project_root(self, tmp_path: Path) -> None:
        """Test normal case where file is inside project root."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        test_file = project_root / "test" / "test_foo.py"
        test_file.parent.mkdir()
        test_file.touch()

        result = module_name_from_file_path(test_file, project_root)
        assert result == "test.test_foo"

    def test_file_outside_project_root_without_traverse_up(self, tmp_path: Path) -> None:
        """Test that file outside project root raises ValueError by default."""
        project_root = tmp_path / "project" / "test"
        project_root.mkdir(parents=True)

        # File is in a sibling directory, not under project_root
        test_file = tmp_path / "project" / "src" / "__tests__" / "test_foo.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        with pytest.raises(ValueError, match="is not within the project root"):
            module_name_from_file_path(test_file, project_root)

    def test_file_outside_project_root_with_traverse_up(self, tmp_path: Path) -> None:
        """Test that traverse_up=True handles files outside project root."""
        project_root = tmp_path / "project" / "test"
        project_root.mkdir(parents=True)

        # File is in a sibling directory, not under project_root
        test_file = tmp_path / "project" / "src" / "__tests__" / "codeflash-generated" / "test_foo.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        # With traverse_up=True, it should find a common ancestor
        result = module_name_from_file_path(test_file, project_root, traverse_up=True)

        # Should return a relative path from some ancestor directory
        assert "test_foo" in result
        assert not result.startswith(".")

    def test_colocated_test_directory_structure(self, tmp_path: Path) -> None:
        """Test real-world scenario with co-located __tests__ directory.

        This reproduces the bug from trace 7b97ddba-6ecd-42fd-b572-d40658746836:
        - Source: /workspace/target/src/gateway/server/ws-connection/connect-policy.ts
        - Tests root: /workspace/target/test
        - Generated test: /workspace/target/src/gateway/server/__tests__/codeflash-generated/test_xxx.test.ts

        Without traverse_up=True, this should fail.
        """
        project_root = tmp_path / "target"
        project_root.mkdir()

        tests_root = project_root / "test"
        tests_root.mkdir()

        # Source file location
        source_file = project_root / "src" / "gateway" / "server" / "ws-connection" / "connect-policy.ts"
        source_file.parent.mkdir(parents=True)
        source_file.touch()

        # Generated test in co-located __tests__ directory
        test_file = project_root / "src" / "gateway" / "server" / "__tests__" / "codeflash-generated" / "test_resolveControlUiAuthPolicy.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        # This should fail WITHOUT traverse_up
        with pytest.raises(ValueError, match="is not within the project root"):
            module_name_from_file_path(test_file, tests_root)

        # This should succeed WITH traverse_up
        result = module_name_from_file_path(test_file, tests_root, traverse_up=True)
        assert "test_resolveControlUiAuthPolicy" in result
