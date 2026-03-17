"""Tests for Java project configuration detection."""

from pathlib import Path

import pytest

from codeflash.languages.java.build_tools import BuildTool
from codeflash.languages.java.config import (
    JavaProjectConfig,
    detect_java_project,
    get_test_class_pattern,
    get_test_file_pattern,
    is_java_project,
)


class TestIsJavaProject:
    """Tests for is_java_project function."""

    def test_maven_project(self, tmp_path: Path):
        """Test detecting a Maven project."""
        (tmp_path / "pom.xml").write_text("<project></project>")
        assert is_java_project(tmp_path) is True

    def test_gradle_project(self, tmp_path: Path):
        """Test detecting a Gradle project."""
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
        assert is_java_project(tmp_path) is True

    def test_gradle_kotlin_project(self, tmp_path: Path):
        """Test detecting a Gradle Kotlin DSL project."""
        (tmp_path / "build.gradle.kts").write_text("plugins { java }")
        assert is_java_project(tmp_path) is True

    def test_java_files_only(self, tmp_path: Path):
        """Test detecting project with only Java files."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "Main.java").write_text("public class Main {}")
        assert is_java_project(tmp_path) is True

    def test_not_java_project(self, tmp_path: Path):
        """Test non-Java directory."""
        (tmp_path / "README.md").write_text("# Not a Java project")
        assert is_java_project(tmp_path) is False

    def test_empty_directory(self, tmp_path: Path):
        """Test empty directory."""
        assert is_java_project(tmp_path) is False


class TestDetectJavaProject:
    """Tests for detect_java_project function."""

    def test_detect_maven_with_junit5(self, tmp_path: Path):
        """Test detecting Maven project with JUnit 5."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.build_tool == BuildTool.MAVEN
        assert config.has_junit5 is True
        assert config.group_id == "com.example"
        assert config.artifact_id == "my-app"
        assert config.java_version == "11"

    def test_detect_maven_with_junit4(self, tmp_path: Path):
        """Test detecting Maven project with JUnit 4."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>legacy-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.2</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.has_junit4 is True

    def test_detect_maven_with_testng(self, tmp_path: Path):
        """Test detecting Maven project with TestNG."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>testng-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <dependency>
            <groupId>org.testng</groupId>
            <artifactId>testng</artifactId>
            <version>7.7.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.has_testng is True

    def test_detect_gradle_project(self, tmp_path: Path):
        """Test detecting Gradle project."""
        gradle_content = """
plugins {
    id 'java'
}

dependencies {
    testImplementation 'org.junit.jupiter:junit-jupiter:5.9.0'
}

test {
    useJUnitPlatform()
}
"""
        (tmp_path / "build.gradle").write_text(gradle_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.build_tool == BuildTool.GRADLE
        assert config.has_junit5 is True

    def test_detect_from_test_files(self, tmp_path: Path):
        """Test detecting test framework from test file imports."""
        (tmp_path / "pom.xml").write_text("<project></project>")
        test_root = tmp_path / "src" / "test" / "java"
        test_root.mkdir(parents=True)

        # Create a test file with JUnit 5 imports
        (test_root / "ExampleTest.java").write_text("""
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ExampleTest {
    @Test
    void test() {}
}
""")

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.has_junit5 is True

    def test_detect_mockito(self, tmp_path: Path):
        """Test detecting Mockito dependency."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>mock-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <version>5.3.0</version>
        </dependency>
    </dependencies>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.has_mockito is True

    def test_detect_assertj(self, tmp_path: Path):
        """Test detecting AssertJ dependency."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>assertj-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <dependency>
            <groupId>org.assertj</groupId>
            <artifactId>assertj-core</artifactId>
            <version>3.24.0</version>
        </dependency>
    </dependencies>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        config = detect_java_project(tmp_path)

        assert config is not None
        assert config.has_assertj is True

    def test_detect_non_java_project(self, tmp_path: Path):
        """Test detecting non-Java directory."""
        (tmp_path / "package.json").write_text('{"name": "js-project"}')

        config = detect_java_project(tmp_path)

        assert config is None


class TestJavaProjectConfig:
    """Tests for JavaProjectConfig dataclass."""

    def test_config_fields(self, tmp_path: Path):
        """Test that all config fields are accessible."""
        config = JavaProjectConfig(
            project_root=tmp_path,
            build_tool=BuildTool.MAVEN,
            source_root=tmp_path / "src" / "main" / "java",
            test_root=tmp_path / "src" / "test" / "java",
            java_version="17",
            encoding="UTF-8",
            test_framework="junit5",
            group_id="com.example",
            artifact_id="my-app",
            version="1.0.0",
            has_junit5=True,
            has_junit4=False,
            has_testng=False,
            has_mockito=True,
            has_assertj=False,
        )

        assert config.build_tool == BuildTool.MAVEN
        assert config.java_version == "17"
        assert config.has_junit5 is True
        assert config.has_mockito is True


class TestGetTestPatterns:
    """Tests for test pattern functions."""

    def test_get_test_file_pattern(self, tmp_path: Path):
        """Test getting test file pattern."""
        config = JavaProjectConfig(
            project_root=tmp_path,
            build_tool=BuildTool.MAVEN,
            source_root=None,
            test_root=None,
            java_version=None,
            encoding="UTF-8",
            test_framework="junit5",
            group_id=None,
            artifact_id=None,
            version=None,
        )

        pattern = get_test_file_pattern(config)
        assert pattern == "*Test.java"

    def test_get_test_class_pattern(self, tmp_path: Path):
        """Test getting test class pattern."""
        config = JavaProjectConfig(
            project_root=tmp_path,
            build_tool=BuildTool.MAVEN,
            source_root=None,
            test_root=None,
            java_version=None,
            encoding="UTF-8",
            test_framework="junit5",
            group_id=None,
            artifact_id=None,
            version=None,
        )

        pattern = get_test_class_pattern(config)
        assert "Test" in pattern


class TestDetectWithFixture:
    """Tests using the Java fixture project."""

    @pytest.fixture
    def java_fixture_path(self):
        """Get path to the Java fixture project."""
        fixture_path = Path(__file__).parent.parent.parent / "test_languages" / "fixtures" / "java_maven"
        if not fixture_path.exists():
            pytest.skip("Java fixture project not found")
        return fixture_path

    def test_detect_fixture_project(self, java_fixture_path: Path):
        """Test detecting the fixture project."""
        config = detect_java_project(java_fixture_path)

        assert config is not None
        assert config.build_tool == BuildTool.MAVEN
        assert config.source_root is not None
        assert config.test_root is not None
        assert config.has_junit5 is True
