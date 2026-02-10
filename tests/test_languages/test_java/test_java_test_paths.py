"""Tests for Java test path handling in FunctionOptimizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetJavaSourcesRoot:
    """Tests for the _get_java_sources_root method."""

    def _create_mock_optimizer(self, tests_root: str):
        """Create a mock FunctionOptimizer with the given tests_root."""
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        # Create a minimal mock
        mock_optimizer = MagicMock(spec=FunctionOptimizer)
        mock_optimizer.test_cfg = MagicMock()
        mock_optimizer.test_cfg.tests_root = Path(tests_root)

        # Bind the actual method to the mock
        mock_optimizer._get_java_sources_root = lambda: FunctionOptimizer._get_java_sources_root(mock_optimizer)

        return mock_optimizer

    def test_detects_com_package_prefix(self):
        """Test that it correctly detects 'com' package prefix and returns parent."""
        optimizer = self._create_mock_optimizer("/project/test/src/com/aerospike/test")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/test/src")

    def test_detects_org_package_prefix(self):
        """Test that it correctly detects 'org' package prefix and returns parent."""
        optimizer = self._create_mock_optimizer("/project/src/test/org/example/tests")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/src/test")

    def test_detects_net_package_prefix(self):
        """Test that it correctly detects 'net' package prefix."""
        optimizer = self._create_mock_optimizer("/project/test/net/company/utils")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/test")

    def test_detects_io_package_prefix(self):
        """Test that it correctly detects 'io' package prefix."""
        optimizer = self._create_mock_optimizer("/project/src/test/java/io/github/project")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/src/test/java")

    def test_detects_edu_package_prefix(self):
        """Test that it correctly detects 'edu' package prefix."""
        optimizer = self._create_mock_optimizer("/project/test/edu/university/cs")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/test")

    def test_detects_gov_package_prefix(self):
        """Test that it correctly detects 'gov' package prefix."""
        optimizer = self._create_mock_optimizer("/project/test/gov/agency/tools")
        result = optimizer._get_java_sources_root()
        assert result == Path("/project/test")

    def test_maven_structure_with_java_dir(self):
        """Test standard Maven structure: src/test/java."""
        optimizer = self._create_mock_optimizer("/project/src/test/java")
        result = optimizer._get_java_sources_root()
        # Should return the path including 'java'
        assert result == Path("/project/src/test/java")

    def test_fallback_when_no_package_prefix(self):
        """Test fallback behavior when no standard package prefix found."""
        optimizer = self._create_mock_optimizer("/project/custom/tests")
        result = optimizer._get_java_sources_root()
        # Should return tests_root as-is
        assert result == Path("/project/custom/tests")

    def test_relative_path_with_com_prefix(self):
        """Test with relative path containing 'com' prefix."""
        optimizer = self._create_mock_optimizer("test/src/com/example")
        result = optimizer._get_java_sources_root()
        assert result == Path("test/src")

    def test_aerospike_project_structure(self):
        """Test with the actual aerospike project structure that had the bug."""
        # This is the actual path from the bug report
        optimizer = self._create_mock_optimizer("/Users/test/Work/aerospike-client-java/test/src/com/aerospike/test")
        result = optimizer._get_java_sources_root()
        assert result == Path("/Users/test/Work/aerospike-client-java/test/src")


class TestFixJavaTestPathsIntegration:
    """Integration tests for _fix_java_test_paths with the path fix."""

    def _create_mock_optimizer(self, tests_root: str):
        """Create a mock FunctionOptimizer with the given tests_root."""
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        mock_optimizer = MagicMock(spec=FunctionOptimizer)
        mock_optimizer.test_cfg = MagicMock()
        mock_optimizer.test_cfg.tests_root = Path(tests_root)

        # Bind the actual methods
        mock_optimizer._get_java_sources_root = lambda: FunctionOptimizer._get_java_sources_root(mock_optimizer)
        mock_optimizer._fix_java_test_paths = lambda behavior_source, perf_source, used_paths: FunctionOptimizer._fix_java_test_paths(mock_optimizer, behavior_source, perf_source, used_paths)

        return mock_optimizer

    def test_no_path_duplication_with_package_in_tests_root(self, tmp_path):
        """Test that paths are not duplicated when tests_root includes package structure."""
        # Create a tests_root that includes package path (like aerospike project)
        tests_root = tmp_path / "test" / "src" / "com" / "aerospike" / "test"
        tests_root.mkdir(parents=True)

        optimizer = self._create_mock_optimizer(str(tests_root))

        behavior_source = """
package com.aerospike.client.util;

public class UnpackerTest__perfinstrumented {
    @Test
    public void testUnpack() {}
}
"""
        perf_source = """
package com.aerospike.client.util;

public class UnpackerTest__perfonlyinstrumented {
    @Test
    public void testUnpack() {}
}
"""
        behavior_path, perf_path, _, _ = optimizer._fix_java_test_paths(behavior_source, perf_source, set())

        # The path should be test/src/com/aerospike/client/util/UnpackerTest__perfinstrumented.java
        # NOT test/src/com/aerospike/test/com/aerospike/client/util/...
        expected_java_root = tmp_path / "test" / "src"
        assert behavior_path == expected_java_root / "com" / "aerospike" / "client" / "util" / "UnpackerTest__perfinstrumented.java"
        assert perf_path == expected_java_root / "com" / "aerospike" / "client" / "util" / "UnpackerTest__perfonlyinstrumented.java"

        # Verify there's no duplication in the path
        assert "com/aerospike/test/com" not in str(behavior_path)
        assert "com/aerospike/test/com" not in str(perf_path)

    def test_standard_maven_structure(self, tmp_path):
        """Test with standard Maven structure (src/test/java)."""
        tests_root = tmp_path / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        optimizer = self._create_mock_optimizer(str(tests_root))

        behavior_source = """
package com.example;

public class CalculatorTest__perfinstrumented {
    @Test
    public void testAdd() {}
}
"""
        perf_source = """
package com.example;

public class CalculatorTest__perfonlyinstrumented {
    @Test
    public void testAdd() {}
}
"""
        behavior_path, perf_path, _, _ = optimizer._fix_java_test_paths(behavior_source, perf_source, set())

        # Should be src/test/java/com/example/CalculatorTest__perfinstrumented.java
        assert behavior_path == tests_root / "com" / "example" / "CalculatorTest__perfinstrumented.java"
        assert perf_path == tests_root / "com" / "example" / "CalculatorTest__perfonlyinstrumented.java"
