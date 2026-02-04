"""Tests for Java build tool detection and integration."""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.java.build_tools import (
    BuildTool,
    detect_build_tool,
    find_maven_executable,
    find_source_root,
    find_test_root,
    get_project_info,
)


class TestBuildToolDetection:
    """Tests for build tool detection."""

    def test_detect_maven_project(self, tmp_path: Path):
        """Test detecting a Maven project."""
        # Create pom.xml
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)

        assert detect_build_tool(tmp_path) == BuildTool.MAVEN

    def test_detect_gradle_project(self, tmp_path: Path):
        """Test detecting a Gradle project."""
        # Create build.gradle
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")

        assert detect_build_tool(tmp_path) == BuildTool.GRADLE

    def test_detect_gradle_kotlin_project(self, tmp_path: Path):
        """Test detecting a Gradle Kotlin DSL project."""
        # Create build.gradle.kts
        (tmp_path / "build.gradle.kts").write_text('plugins { java }')

        assert detect_build_tool(tmp_path) == BuildTool.GRADLE

    def test_detect_unknown_project(self, tmp_path: Path):
        """Test detecting unknown project type."""
        # Empty directory
        assert detect_build_tool(tmp_path) == BuildTool.UNKNOWN

    def test_maven_takes_precedence(self, tmp_path: Path):
        """Test that Maven takes precedence if both exist."""
        # Create both pom.xml and build.gradle
        (tmp_path / "pom.xml").write_text("<project></project>")
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }")

        # Maven should be detected first
        assert detect_build_tool(tmp_path) == BuildTool.MAVEN


class TestMavenProjectInfo:
    """Tests for Maven project info extraction."""

    def test_get_maven_project_info(self, tmp_path: Path):
        """Test extracting project info from pom.xml."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)

        # Create standard Maven directory structure
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)

        assert info is not None
        assert info.build_tool == BuildTool.MAVEN
        assert info.group_id == "com.example"
        assert info.artifact_id == "my-app"
        assert info.version == "1.0.0"
        assert info.java_version == "11"
        assert len(info.source_roots) == 1
        assert len(info.test_roots) == 1

    def test_get_maven_project_info_with_java_version_property(self, tmp_path: Path):
        """Test extracting Java version from java.version property."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>

    <properties>
        <java.version>17</java.version>
    </properties>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)

        assert info is not None
        assert info.java_version == "17"


class TestDirectoryDetection:
    """Tests for source and test directory detection."""

    def test_find_maven_source_root(self, tmp_path: Path):
        """Test finding Maven source root."""
        (tmp_path / "pom.xml").write_text("<project></project>")
        src_root = tmp_path / "src" / "main" / "java"
        src_root.mkdir(parents=True)

        result = find_source_root(tmp_path)
        assert result is not None
        assert result == src_root

    def test_find_maven_test_root(self, tmp_path: Path):
        """Test finding Maven test root."""
        (tmp_path / "pom.xml").write_text("<project></project>")
        test_root = tmp_path / "src" / "test" / "java"
        test_root.mkdir(parents=True)

        result = find_test_root(tmp_path)
        assert result is not None
        assert result == test_root

    def test_find_source_root_not_found(self, tmp_path: Path):
        """Test when source root doesn't exist."""
        result = find_source_root(tmp_path)
        assert result is None

    def test_find_test_root_not_found(self, tmp_path: Path):
        """Test when test root doesn't exist."""
        result = find_test_root(tmp_path)
        assert result is None

    def test_find_alternative_test_root(self, tmp_path: Path):
        """Test finding alternative test directory."""
        # Create a 'test' directory (non-Maven style)
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        result = find_test_root(tmp_path)
        assert result is not None
        assert result == test_dir


class TestMavenExecutable:
    """Tests for Maven executable detection."""

    def test_find_maven_executable_system(self):
        """Test finding system Maven."""
        # This test may pass or fail depending on whether Maven is installed
        mvn = find_maven_executable()
        # We can't assert it exists, just that the function doesn't crash
        if mvn:
            assert "mvn" in mvn.lower() or "maven" in mvn.lower()

    def test_find_maven_wrapper(self, tmp_path: Path, monkeypatch):
        """Test finding Maven wrapper."""
        # Create mvnw file
        mvnw_path = tmp_path / "mvnw"
        mvnw_path.write_text("#!/bin/bash\necho 'Maven Wrapper'")
        mvnw_path.chmod(0o755)

        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        mvn = find_maven_executable()
        # Should find the wrapper
        assert mvn is not None


class TestPomXmlParsing:
    """Tests for pom.xml parsing edge cases."""

    def test_pom_without_namespace(self, tmp_path: Path):
        """Test parsing pom.xml without XML namespace."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>simple-app</artifactId>
    <version>1.0</version>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)

        assert info is not None
        assert info.group_id == "com.example"
        assert info.artifact_id == "simple-app"

    def test_pom_with_parent(self, tmp_path: Path):
        """Test parsing pom.xml with parent POM."""
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.0.0</version>
    </parent>

    <groupId>com.example</groupId>
    <artifactId>child-app</artifactId>
    <version>1.0</version>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)

        assert info is not None
        assert info.artifact_id == "child-app"

    def test_invalid_pom_xml(self, tmp_path: Path):
        """Test handling invalid pom.xml."""
        # Create invalid XML
        (tmp_path / "pom.xml").write_text("this is not valid xml")

        info = get_project_info(tmp_path)
        # Should return None or handle gracefully
        assert info is None


class TestGradleProjectInfo:
    """Tests for Gradle project info extraction."""

    def test_get_gradle_project_info(self, tmp_path: Path):
        """Test extracting basic Gradle project info."""
        (tmp_path / "build.gradle").write_text("""
plugins {
    id 'java'
}

group = 'com.example'
version = '1.0.0'
""")

        # Create standard Gradle directory structure
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)

        assert info is not None
        assert info.build_tool == BuildTool.GRADLE
        assert len(info.source_roots) == 1
        assert len(info.test_roots) == 1
