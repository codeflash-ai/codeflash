"""Tests for Java test path handling in FunctionOptimizer."""

from pathlib import Path
from unittest.mock import MagicMock

from codeflash.languages.java.test_runner import _extract_custom_source_dirs, _path_to_class_name


class TestGetJavaSourcesRoot:
    """Tests for the _get_java_sources_root method."""

    def _create_mock_optimizer(self, tests_root: str):
        """Create a mock FunctionOptimizer with the given tests_root."""
        from codeflash.languages.java.function_optimizer import JavaFunctionOptimizer

        mock_optimizer = MagicMock(spec=JavaFunctionOptimizer)
        mock_optimizer.test_cfg = MagicMock()
        mock_optimizer.test_cfg.tests_root = Path(tests_root)
        mock_optimizer.function_to_optimize = MagicMock()
        mock_optimizer.function_to_optimize.file_path = Path("/nonexistent/Foo.java")

        mock_optimizer._get_java_sources_root = lambda: JavaFunctionOptimizer._get_java_sources_root(mock_optimizer)

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


class TestGetJavaSourcesRootMultiModule:
    """Tests for _get_java_sources_root with multi-module projects."""

    def _create_mock_optimizer(self, tests_root: str, file_path: str):
        from codeflash.languages.java.function_optimizer import JavaFunctionOptimizer

        mock_optimizer = MagicMock(spec=JavaFunctionOptimizer)
        mock_optimizer.test_cfg = MagicMock()
        mock_optimizer.test_cfg.tests_root = Path(tests_root)
        mock_optimizer.function_to_optimize = MagicMock()
        mock_optimizer.function_to_optimize.file_path = Path(file_path)
        mock_optimizer._get_java_sources_root = lambda: JavaFunctionOptimizer._get_java_sources_root(mock_optimizer)
        return mock_optimizer

    def test_kafka_streams_module(self, tmp_path):
        """Kafka: function in streams module should use streams test dir, not clients."""
        (tmp_path / "streams" / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "streams" / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "clients" / "src" / "test" / "java").mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "clients" / "src" / "test" / "java"),
            file_path=str(tmp_path / "streams" / "src" / "main" / "java" / "org" / "apache" / "kafka" / "streams" / "query" / "QueryConfig.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "streams" / "src" / "test" / "java"

    def test_kafka_connect_module(self, tmp_path):
        """Kafka: function in connect/runtime should use connect/runtime test dir."""
        (tmp_path / "connect" / "runtime" / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "connect" / "runtime" / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "clients" / "src" / "test" / "java").mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "clients" / "src" / "test" / "java"),
            file_path=str(tmp_path / "connect" / "runtime" / "src" / "main" / "java" / "org" / "apache" / "kafka" / "connect" / "runtime" / "Worker.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "connect" / "runtime" / "src" / "test" / "java"

    def test_kafka_clients_module_same_as_config(self, tmp_path):
        """Kafka: function in clients module should still use clients test dir."""
        (tmp_path / "clients" / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "clients" / "src" / "test" / "java").mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "clients" / "src" / "test" / "java"),
            file_path=str(tmp_path / "clients" / "src" / "main" / "java" / "org" / "apache" / "kafka" / "common" / "utils" / "Bytes.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "clients" / "src" / "test" / "java"

    def test_opensearch_libs_module(self, tmp_path):
        """OpenSearch: function in libs/core should use libs/core test dir."""
        (tmp_path / "libs" / "core" / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "libs" / "core" / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "server" / "src" / "test" / "java").mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "server" / "src" / "test" / "java"),
            file_path=str(tmp_path / "libs" / "core" / "src" / "main" / "java" / "org" / "opensearch" / "core" / "common" / "Strings.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "libs" / "core" / "src" / "test" / "java"

    def test_spring_boot_subproject(self, tmp_path):
        """Spring Boot: function in autoconfigure should use autoconfigure test dir."""
        (tmp_path / "spring-boot-autoconfigure" / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "spring-boot-autoconfigure" / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "spring-boot" / "src" / "test" / "java").mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "spring-boot" / "src" / "test" / "java"),
            file_path=str(tmp_path / "spring-boot-autoconfigure" / "src" / "main" / "java" / "org" / "springframework" / "boot" / "autoconfigure" / "web" / "ServerProperties.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "spring-boot-autoconfigure" / "src" / "test" / "java"

    def test_fallback_when_derived_test_dir_missing(self, tmp_path):
        """When derived test dir doesn't exist, fall back to tests_root logic."""
        (tmp_path / "module-a" / "src" / "main" / "java").mkdir(parents=True)
        # Deliberately NOT creating module-a/src/test/java
        tests_root = tmp_path / "src" / "test" / "java"
        tests_root.mkdir(parents=True)

        optimizer = self._create_mock_optimizer(
            tests_root=str(tests_root),
            file_path=str(tmp_path / "module-a" / "src" / "main" / "java" / "com" / "example" / "Foo.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tests_root

    def test_non_standard_layout_falls_through(self, tmp_path):
        """Non-standard layout (no src/main/java) falls through to existing logic."""
        optimizer = self._create_mock_optimizer(
            tests_root=str(tmp_path / "custom" / "tests"),
            file_path=str(tmp_path / "custom" / "src" / "Foo.java"),
        )
        result = optimizer._get_java_sources_root()
        assert result == tmp_path / "custom" / "tests"


class TestFixJavaTestPathsIntegration:
    """Integration tests for _fix_java_test_paths with the path fix."""

    def _create_mock_optimizer(self, tests_root: str):
        """Create a mock FunctionOptimizer with the given tests_root."""
        from codeflash.languages.java.function_optimizer import JavaFunctionOptimizer

        mock_optimizer = MagicMock(spec=JavaFunctionOptimizer)
        mock_optimizer.test_cfg = MagicMock()
        mock_optimizer.test_cfg.tests_root = Path(tests_root)
        mock_optimizer.function_to_optimize = MagicMock()
        mock_optimizer.function_to_optimize.file_path = Path("/nonexistent/Foo.java")

        mock_optimizer._get_java_sources_root = lambda: JavaFunctionOptimizer._get_java_sources_root(mock_optimizer)
        mock_optimizer._fix_java_test_paths = lambda behavior_source, perf_source, used_paths, display_source="": (
            JavaFunctionOptimizer._fix_java_test_paths(
                mock_optimizer, behavior_source, perf_source, used_paths, display_source
            )
        )

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
        behavior_path, perf_path, _, _, _ = optimizer._fix_java_test_paths(behavior_source, perf_source, set())

        # The path should be test/src/com/aerospike/client/util/UnpackerTest__perfinstrumented.java
        # NOT test/src/com/aerospike/test/com/aerospike/client/util/...
        expected_java_root = tmp_path / "test" / "src"
        assert (
            behavior_path
            == expected_java_root / "com" / "aerospike" / "client" / "util" / "UnpackerTest__perfinstrumented.java"
        )
        assert (
            perf_path
            == expected_java_root / "com" / "aerospike" / "client" / "util" / "UnpackerTest__perfonlyinstrumented.java"
        )

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
        behavior_path, perf_path, _, _, _ = optimizer._fix_java_test_paths(behavior_source, perf_source, set())

        # Should be src/test/java/com/example/CalculatorTest__perfinstrumented.java
        assert behavior_path == tests_root / "com" / "example" / "CalculatorTest__perfinstrumented.java"
        assert perf_path == tests_root / "com" / "example" / "CalculatorTest__perfonlyinstrumented.java"


class TestPathToClassNameWithCustomDirs:
    """Tests for _path_to_class_name with custom source directories."""

    def test_standard_maven_layout(self):
        path = Path("src/test/java/com/example/CalculatorTest.java")
        assert _path_to_class_name(path) == "com.example.CalculatorTest"

    def test_standard_maven_main_layout(self):
        path = Path("src/main/java/com/example/StringUtils.java")
        assert _path_to_class_name(path) == "com.example.StringUtils"

    def test_custom_source_dir(self):
        path = Path("/project/src/main/custom/com/example/Foo.java")
        result = _path_to_class_name(path, source_dirs=["src/main/custom"])
        assert result == "com.example.Foo"

    def test_non_standard_layout(self):
        path = Path("/project/app/java/com/example/Foo.java")
        result = _path_to_class_name(path, source_dirs=["app/java"])
        assert result == "com.example.Foo"

    def test_custom_dir_takes_priority(self):
        path = Path("/project/src/main/custom/com/example/Bar.java")
        result = _path_to_class_name(path, source_dirs=["src/main/custom"])
        assert result == "com.example.Bar"

    def test_fallback_to_standard_when_custom_no_match(self):
        path = Path("src/test/java/com/example/Test.java")
        result = _path_to_class_name(path, source_dirs=["nonexistent/dir"])
        assert result == "com.example.Test"

    def test_fallback_to_stem_when_no_patterns_match(self):
        path = Path("/project/weird/layout/MyClass.java")
        result = _path_to_class_name(path)
        assert result == "MyClass"

    def test_non_java_file_returns_none(self):
        path = Path("src/test/java/com/example/Readme.txt")
        assert _path_to_class_name(path) is None

    def test_multiple_custom_dirs(self):
        path = Path("/project/app/src/com/example/Foo.java")
        result = _path_to_class_name(path, source_dirs=["app/src", "lib/src"])
        assert result == "com.example.Foo"

    def test_empty_source_dirs_list(self):
        path = Path("src/test/java/com/example/Test.java")
        result = _path_to_class_name(path, source_dirs=[])
        assert result == "com.example.Test"


class TestExtractSourceDirsFromPom:
    """Tests for extracting custom source directories from pom.xml."""

    def test_custom_source_directory(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <build>
        <sourceDirectory>src/main/custom</sourceDirectory>
        <testSourceDirectory>src/test/custom</testSourceDirectory>
    </build>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        dirs = _extract_custom_source_dirs(tmp_path)
        assert "src/main/custom" in dirs
        assert "src/test/custom" in dirs

    def test_standard_dirs_excluded(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <build>
        <sourceDirectory>src/main/java</sourceDirectory>
        <testSourceDirectory>src/test/java</testSourceDirectory>
    </build>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        dirs = _extract_custom_source_dirs(tmp_path)
        assert dirs == []

    def test_no_pom_returns_empty(self, tmp_path):
        dirs = _extract_custom_source_dirs(tmp_path)
        assert dirs == []

    def test_pom_without_build_section(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        dirs = _extract_custom_source_dirs(tmp_path)
        assert dirs == []

    def test_malformed_xml(self, tmp_path):
        (tmp_path / "pom.xml").write_text("this is not valid xml <<<<")
        dirs = _extract_custom_source_dirs(tmp_path)
        assert dirs == []
