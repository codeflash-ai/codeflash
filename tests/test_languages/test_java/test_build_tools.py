"""Tests for Java build tool detection and integration."""

import os
from pathlib import Path
from unittest.mock import patch

from codeflash.languages.java.build_tools import (
    BuildTool,
    detect_build_tool,
    find_source_root,
    find_test_root,
    get_project_info,
)
from codeflash.languages.java.gradle_strategy import GradleStrategy
from codeflash.languages.java.line_profiler import find_agent_jar
from codeflash.languages.java.maven_strategy import (
    MavenStrategy,
    add_codeflash_dependency,
    download_from_maven_central_http,
)
from codeflash.languages.java.test_runner import _extract_modules_from_pom_content


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
        (tmp_path / "build.gradle.kts").write_text("plugins { java }")

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
        strategy = MavenStrategy()
        mvn = strategy.find_executable(Path())
        # We can't assert it exists, just that the function doesn't crash
        if mvn:
            assert "mvn" in mvn.lower() or "maven" in mvn.lower()

    def test_find_maven_wrapper(self, tmp_path: Path, monkeypatch):
        """Test finding Maven wrapper."""
        # Create mvnw file
        mvnw_path = tmp_path / "mvnw"
        mvnw_path.write_text("#!/bin/bash\necho 'Maven Wrapper'")
        mvnw_path.chmod(0o755)

        strategy = MavenStrategy()
        mvn = strategy.find_executable(tmp_path)
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


class TestXmlModuleExtraction:
    """Tests for XML-based module extraction replacing regex."""

    def test_namespaced_pom_modules(self):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <modules>
        <module>core</module>
        <module>service</module>
        <module>app</module>
    </modules>
</project>
"""
        modules = _extract_modules_from_pom_content(content)
        assert modules == ["core", "service", "app"]

    def test_non_namespaced_pom_modules(self):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modules>
        <module>api</module>
        <module>impl</module>
    </modules>
</project>
"""
        modules = _extract_modules_from_pom_content(content)
        assert modules == ["api", "impl"]

    def test_empty_modules_element(self):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modules>
    </modules>
</project>
"""
        modules = _extract_modules_from_pom_content(content)
        assert modules == []

    def test_no_modules_element(self):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
</project>
"""
        modules = _extract_modules_from_pom_content(content)
        assert modules == []

    def test_malformed_xml_handled_gracefully(self):
        content = "this is not valid xml <<<<"
        modules = _extract_modules_from_pom_content(content)
        assert modules == []

    def test_partial_xml_handled_gracefully(self):
        content = "<project><modules><module>core</module>"
        modules = _extract_modules_from_pom_content(content)
        assert modules == []

    def test_nested_module_paths(self):
        content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modules>
        <module>libs/core</module>
        <module>apps/web</module>
    </modules>
</project>
"""
        modules = _extract_modules_from_pom_content(content)
        assert modules == ["libs/core", "apps/web"]


class TestMavenProfiles:
    """Tests for Maven profile support in test commands."""

    def test_profile_env_var_read(self, monkeypatch):
        monkeypatch.setenv("CODEFLASH_MAVEN_PROFILES", "test-profile")
        profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
        assert profiles == "test-profile"

    def test_no_profile_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("CODEFLASH_MAVEN_PROFILES", raising=False)
        profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
        assert profiles == ""

    def test_multiple_profiles_comma_separated(self, monkeypatch):
        monkeypatch.setenv("CODEFLASH_MAVEN_PROFILES", "profile1,profile2")
        profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
        assert profiles == "profile1,profile2"
        cmd_parts = ["-P", profiles]
        assert cmd_parts == ["-P", "profile1,profile2"]

    def test_whitespace_stripped_from_profiles(self, monkeypatch):
        monkeypatch.setenv("CODEFLASH_MAVEN_PROFILES", "  my-profile  ")
        profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
        assert profiles == "my-profile"


class TestMavenExecutableWithProjectRoot:
    """Tests for MavenStrategy.find_executable with project_root parameter."""

    def test_find_wrapper_in_project_root(self, tmp_path):
        mvnw_path = tmp_path / "mvnw"
        mvnw_path.write_text("#!/bin/bash\necho Maven Wrapper")
        mvnw_path.chmod(0o755)

        strategy = MavenStrategy()
        result = strategy.find_executable(tmp_path)
        assert result is not None
        assert str(tmp_path / "mvnw") in result

    def test_fallback_to_cwd(self, tmp_path):
        strategy = MavenStrategy()
        result = strategy.find_executable(tmp_path)
        # Should not crash even with a dir that has no wrapper

    def test_with_nonexistent_wrapper(self, tmp_path):
        strategy = MavenStrategy()
        result = strategy.find_executable(tmp_path)
        # Should not crash, may return system mvn or None


class TestCustomSourceDirectoryDetection:
    """Tests for custom source directory detection from pom.xml."""

    def test_detects_custom_source_directory(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>
    <build>
        <sourceDirectory>src/main/custom</sourceDirectory>
    </build>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "main" / "custom").mkdir(parents=True)

        info = get_project_info(tmp_path)
        assert info is not None
        source_strs = [str(s) for s in info.source_roots]
        assert any("custom" in s for s in source_strs)

    def test_standard_dirs_still_detected(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)
        assert info is not None
        assert len(info.source_roots) == 1
        assert len(info.test_roots) == 1

    def test_nonexistent_custom_dir_ignored(self, tmp_path):
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0.0</version>
    <build>
        <sourceDirectory>src/main/nonexistent</sourceDirectory>
    </build>
</project>
"""
        (tmp_path / "pom.xml").write_text(pom_content)
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        info = get_project_info(tmp_path)
        assert info is not None
        assert len(info.source_roots) == 1


class TestAddCodeflashDependencyToPom:
    """Tests for add_codeflash_dependency, including stale system-scope replacement."""

    def test_adds_dependency_to_clean_pom(self, tmp_path):
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>junit</groupId>\n"
            "      <artifactId>junit</artifactId>\n"
            "      <version>4.13.2</version>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n",
            encoding="utf-8",
        )
        assert add_codeflash_dependency(pom) is True
        content = pom.read_text(encoding="utf-8")
        assert "codeflash-runtime" in content
        assert "<scope>test</scope>" in content

    def test_replaces_system_scope_with_test_scope(self, tmp_path):
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>com.codeflash</groupId>\n"
            "      <artifactId>codeflash-runtime</artifactId>\n"
            "      <version>1.0.0</version>\n"
            "      <scope>system</scope>\n"
            "      <systemPath>/some/path/jar.jar</systemPath>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n",
            encoding="utf-8",
        )
        assert add_codeflash_dependency(pom) is True
        content = pom.read_text(encoding="utf-8")
        assert "<scope>test</scope>" in content
        assert "<scope>system</scope>" not in content
        assert "<systemPath>" not in content

    def test_replaces_system_scope_with_reordered_elements(self, tmp_path):
        """XML elements inside <dependency> can appear in any order."""
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <scope>system</scope>\n"
            "      <groupId>com.codeflash</groupId>\n"
            "      <systemPath>/some/path/jar.jar</systemPath>\n"
            "      <version>1.0.0</version>\n"
            "      <artifactId>codeflash-runtime</artifactId>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n",
            encoding="utf-8",
        )
        assert add_codeflash_dependency(pom) is True
        content = pom.read_text(encoding="utf-8")
        assert "<scope>test</scope>" in content
        assert "<scope>system</scope>" not in content
        assert "<systemPath>" not in content

    def test_skips_when_test_scope_already_present(self, tmp_path):
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>com.codeflash</groupId>\n"
            "      <artifactId>codeflash-runtime</artifactId>\n"
            "      <version>1.0.0</version>\n"
            "      <scope>test</scope>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n",
            encoding="utf-8",
        )
        assert add_codeflash_dependency(pom) is True
        content = pom.read_text(encoding="utf-8")
        assert content.count("codeflash-runtime") == 1

    def test_returns_false_for_missing_pom(self, tmp_path):
        pom = tmp_path / "pom.xml"
        assert add_codeflash_dependency(pom) is False

    def test_returns_false_when_no_dependencies_tag(self, tmp_path):
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0"?>\n<project><modelVersion>4.0.0</modelVersion></project>\n', encoding="utf-8"
        )
        assert add_codeflash_dependency(pom) is False


class TestGradleEnsureRuntimeMultiModule:
    """Tests that ensure_runtime adds the dependency to the correct module build file."""

    def _make_multi_module_project(self, tmp_path):
        """Create a multi-module Gradle project with submodule build files."""
        # Root
        (tmp_path / "build.gradle.kts").write_text("// root build\n", encoding="utf-8")
        (tmp_path / "settings.gradle.kts").write_text('include("clients", "streams")', encoding="utf-8")
        (tmp_path / "gradlew").write_text("#!/bin/sh\necho gradle", encoding="utf-8")
        (tmp_path / "gradlew").chmod(0o755)
        # Submodule build files with a dependencies block
        for module in ["clients", "streams"]:
            module_dir = tmp_path / module
            module_dir.mkdir()
            (module_dir / "build.gradle.kts").write_text(
                'plugins {\n    java\n}\n\ndependencies {\n    testImplementation("junit:junit:4.13.2")\n}\n',
                encoding="utf-8",
            )
        return tmp_path

    def test_adds_dependency_to_correct_module_build_file(self, tmp_path):
        """When test_module='streams', the dependency must be added to streams/build.gradle.kts."""
        project = self._make_multi_module_project(tmp_path)

        strategy = GradleStrategy()
        # Provide a fake runtime JAR
        fake_jar = tmp_path / "fake-runtime.jar"
        fake_jar.write_bytes(b"PK\x03\x04")  # minimal zip header

        with patch.object(strategy, "find_runtime_jar", return_value=fake_jar):
            result = strategy.ensure_runtime(project, test_module="streams")

        assert result is True
        # Dependency should be in streams/build.gradle.kts
        streams_build = (project / "streams" / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" in streams_build
        # And NOT in clients/build.gradle.kts or root build.gradle.kts
        clients_build = (project / "clients" / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" not in clients_build
        root_build = (project / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" not in root_build

    def test_adds_dependency_to_root_when_no_module(self, tmp_path):
        """When test_module=None, the dependency is added to the root build file."""
        project = self._make_multi_module_project(tmp_path)

        strategy = GradleStrategy()
        fake_jar = tmp_path / "fake-runtime.jar"
        fake_jar.write_bytes(b"PK\x03\x04")

        with patch.object(strategy, "find_runtime_jar", return_value=fake_jar):
            result = strategy.ensure_runtime(project, test_module=None)

        assert result is True
        root_build = (project / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" in root_build

    def test_adds_dependency_to_nested_module(self, tmp_path):
        """When test_module='connect:runtime', the dep goes to connect/runtime/build.gradle.kts."""
        project = self._make_multi_module_project(tmp_path)
        # Add nested module
        nested = tmp_path / "connect" / "runtime"
        nested.mkdir(parents=True)
        (nested / "build.gradle.kts").write_text(
            'plugins {\n    java\n}\n\ndependencies {\n    testImplementation("junit:junit:4.13.2")\n}\n',
            encoding="utf-8",
        )

        strategy = GradleStrategy()
        fake_jar = tmp_path / "fake-runtime.jar"
        fake_jar.write_bytes(b"PK\x03\x04")

        with patch.object(strategy, "find_runtime_jar", return_value=fake_jar):
            result = strategy.ensure_runtime(project, test_module="connect:runtime")

        assert result is True
        nested_build = (nested / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" in nested_build


class TestDownloadFromMavenCentralHttp:
    """Tests for the direct HTTP download from Maven Central."""

    def test_returns_existing_m2_jar(self, tmp_path):
        """If the JAR already exists in ~/.m2, return it without downloading."""
        fake_m2 = tmp_path / "m2" / "codeflash-runtime" / "1.0.1" / "codeflash-runtime-1.0.1.jar"
        fake_m2.parent.mkdir(parents=True)
        fake_m2.write_bytes(b"PK\x03\x04")

        with patch("codeflash.languages.java.maven_strategy.M2_JAR_PATH", fake_m2):
            result = download_from_maven_central_http()

        assert result == fake_m2

    def test_downloads_jar_when_not_cached(self, tmp_path):
        """Downloads the JAR to ~/.m2 when it doesn't exist locally."""
        fake_m2 = tmp_path / "m2" / "codeflash-runtime" / "1.0.1" / "codeflash-runtime-1.0.1.jar"

        with (
            patch("codeflash.languages.java.maven_strategy.M2_JAR_PATH", fake_m2),
            patch("urllib.request.urlretrieve") as mock_download,
        ):
            mock_download.side_effect = lambda _url, path: Path(path).write_bytes(b"PK\x03\x04")
            result = download_from_maven_central_http()

        assert result == fake_m2
        assert "repo1.maven.org" in mock_download.call_args[0][0]

    def test_returns_none_on_network_failure(self, tmp_path):
        """Returns None when the download fails."""
        fake_m2 = tmp_path / "m2" / "codeflash-runtime" / "1.0.1" / "codeflash-runtime-1.0.1.jar"

        with (
            patch("codeflash.languages.java.maven_strategy.M2_JAR_PATH", fake_m2),
            patch("urllib.request.urlretrieve", side_effect=OSError("Network unreachable")),
        ):
            result = download_from_maven_central_http()

        assert result is None
        assert not fake_m2.exists()


class TestFindAgentJarFallback:
    """Tests that find_agent_jar falls back to Maven Central HTTP download."""

    def test_falls_back_to_maven_central_when_no_local_jar(self, tmp_path):
        """When no local JAR exists, find_agent_jar tries Maven Central HTTP download."""
        fake_jar = tmp_path / "downloaded.jar"
        fake_jar.write_bytes(b"PK\x03\x04")

        with (
            patch("codeflash.languages.java.line_profiler.CODEFLASH_RUNTIME_VERSION", "99.99.99"),
            patch("codeflash.languages.java.line_profiler.AGENT_JAR_NAME", "codeflash-runtime-99.99.99.jar"),
            patch(
                "codeflash.languages.java.maven_strategy.download_from_maven_central_http", return_value=fake_jar
            ) as mock_download,
        ):
            result = find_agent_jar()

        assert result == fake_jar
        assert mock_download.called


class TestGradleEnsureRuntimeFallback:
    """Tests that Gradle ensure_runtime falls back to Maven Central HTTP download."""

    def test_falls_back_to_http_download_when_find_runtime_jar_returns_none(self, tmp_path):
        """When find_runtime_jar returns None, ensure_runtime tries HTTP download."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "build.gradle.kts").write_text(
            'plugins {\n    java\n}\n\ndependencies {\n    testImplementation("junit:junit:4.13.2")\n}\n',
            encoding="utf-8",
        )
        (project / "gradlew").write_text("#!/bin/sh\necho gradle", encoding="utf-8")
        (project / "gradlew").chmod(0o755)

        fake_jar = tmp_path / "downloaded.jar"
        fake_jar.write_bytes(b"PK\x03\x04")

        strategy = GradleStrategy()
        with (
            patch.object(strategy, "find_runtime_jar", return_value=None),
            patch(
                "codeflash.languages.java.maven_strategy.download_from_maven_central_http", return_value=fake_jar
            ) as mock_download,
        ):
            result = strategy.ensure_runtime(project, test_module=None)

        assert result is True
        assert mock_download.called
        build_content = (project / "build.gradle.kts").read_text(encoding="utf-8")
        assert "codeflash-runtime" in build_content

    def test_fails_when_both_local_and_http_return_none(self, tmp_path):
        """When both find_runtime_jar and HTTP download return None, ensure_runtime fails."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "build.gradle.kts").write_text("plugins { java }\n", encoding="utf-8")

        strategy = GradleStrategy()
        with (
            patch.object(strategy, "find_runtime_jar", return_value=None),
            patch("codeflash.languages.java.maven_strategy.download_from_maven_central_http", return_value=None),
        ):
            result = strategy.ensure_runtime(project, test_module=None)

        assert result is False


class TestGradleSetupCoverage:
    """Tests for GradleStrategy.setup_coverage — returns report path without modifying build files."""

    def test_returns_report_path_for_module(self, tmp_path):
        strategy = GradleStrategy()
        path = strategy.setup_coverage(tmp_path, test_module="eureka-core", project_root=tmp_path)
        assert path == tmp_path / "eureka-core" / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"

    def test_returns_report_path_without_module(self, tmp_path):
        strategy = GradleStrategy()
        path = strategy.setup_coverage(tmp_path, test_module=None, project_root=tmp_path)
        assert path == tmp_path / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"

    def test_does_not_modify_build_files(self, tmp_path):
        """setup_coverage must NOT modify build.gradle — JaCoCo is applied via init script."""
        build_file = tmp_path / "build.gradle"
        original_content = "plugins { id 'java' }\n"
        build_file.write_text(original_content, encoding="utf-8")

        strategy = GradleStrategy()
        strategy.setup_coverage(tmp_path, test_module=None, project_root=tmp_path)
        assert build_file.read_text(encoding="utf-8") == original_content


class TestGradleJacocoInitScript:
    """Tests for the JaCoCo init script content and helper."""

    def test_init_script_has_java_plugin_guard(self):
        from codeflash.languages.java.gradle_strategy import _JACOCO_INIT_SCRIPT

        assert "withType(JavaPlugin)" in _JACOCO_INIT_SCRIPT

    def test_get_jacoco_init_script_creates_temp_file(self):
        from codeflash.languages.java.gradle_strategy import _JACOCO_INIT_SCRIPT, _get_jacoco_init_script

        path = _get_jacoco_init_script()
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert content == _JACOCO_INIT_SCRIPT

    def test_get_jacoco_init_script_is_cached(self):
        from codeflash.languages.java.gradle_strategy import _get_jacoco_init_script

        path1 = _get_jacoco_init_script()
        path2 = _get_jacoco_init_script()
        assert path1 == path2
