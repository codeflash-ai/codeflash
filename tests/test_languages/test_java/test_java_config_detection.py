"""Tests for Java project auto-detection from Maven/Gradle build files.

Tests that codeflash can detect Java projects and infer module-root,
tests-root, and other config from pom.xml / build.gradle / gradle.properties
without requiring a standalone codeflash.toml config file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.java.build_tools import (
    BuildTool,
    detect_build_tool,
    find_source_root,
    find_test_root,
    parse_java_project_config,
)


# ---------------------------------------------------------------------------
# Build tool detection
# ---------------------------------------------------------------------------


class TestDetectBuildTool:
    def test_detect_maven(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        assert detect_build_tool(tmp_path) == BuildTool.MAVEN

    def test_detect_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        assert detect_build_tool(tmp_path) == BuildTool.GRADLE

    def test_detect_gradle_kts(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle.kts").write_text("", encoding="utf-8")
        assert detect_build_tool(tmp_path) == BuildTool.GRADLE

    def test_maven_takes_priority_over_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        assert detect_build_tool(tmp_path) == BuildTool.MAVEN

    def test_unknown_when_no_build_file(self, tmp_path: Path) -> None:
        assert detect_build_tool(tmp_path) == BuildTool.UNKNOWN

    def test_detect_maven_in_parent(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        child = tmp_path / "module"
        child.mkdir()
        assert detect_build_tool(child) == BuildTool.MAVEN


# ---------------------------------------------------------------------------
# Source / test root detection (standard layouts)
# ---------------------------------------------------------------------------


class TestFindSourceRoot:
    def test_standard_maven_layout(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        assert find_source_root(tmp_path) == src

    def test_fallback_to_src_with_java_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "App.java").write_text("class App {}", encoding="utf-8")
        assert find_source_root(tmp_path) == src

    def test_returns_none_when_no_source(self, tmp_path: Path) -> None:
        assert find_source_root(tmp_path) is None


class TestFindTestRoot:
    def test_standard_maven_layout(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        test = tmp_path / "src" / "test" / "java"
        test.mkdir(parents=True)
        assert find_test_root(tmp_path) == test

    def test_fallback_to_test_dir(self, tmp_path: Path) -> None:
        test = tmp_path / "test"
        test.mkdir()
        assert find_test_root(tmp_path) == test

    def test_fallback_to_tests_dir(self, tmp_path: Path) -> None:
        tests = tmp_path / "tests"
        tests.mkdir()
        assert find_test_root(tmp_path) == tests

    def test_returns_none_when_no_test_dir(self, tmp_path: Path) -> None:
        assert find_test_root(tmp_path) is None


# ---------------------------------------------------------------------------
# parse_java_project_config — standard layouts
# ---------------------------------------------------------------------------


class TestParseJavaProjectConfigStandard:
    def test_standard_maven_project(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        test = tmp_path / "src" / "test" / "java"
        test.mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["language"] == "java"
        assert config["module_root"] == str(src)
        assert config["tests_root"] == str(test)

    def test_standard_gradle_project(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        test = tmp_path / "src" / "test" / "java"
        test.mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["language"] == "java"
        assert config["module_root"] == str(src)
        assert config["tests_root"] == str(test)

    def test_returns_none_for_non_java_project(self, tmp_path: Path) -> None:
        assert parse_java_project_config(tmp_path) is None

    def test_defaults_when_dirs_missing(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        config = parse_java_project_config(tmp_path)
        assert config is not None
        # Falls back to default paths even if they don't exist
        assert str(tmp_path / "src" / "main" / "java") == config["module_root"]
        assert config["language"] == "java"


# ---------------------------------------------------------------------------
# parse_java_project_config — Maven properties (codeflash.*)
# ---------------------------------------------------------------------------

MAVEN_POM_WITH_PROPERTIES = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test</artifactId>
  <version>1.0</version>
  <properties>
    <codeflash.moduleRoot>custom/src</codeflash.moduleRoot>
    <codeflash.testsRoot>custom/test</codeflash.testsRoot>
    <codeflash.disableTelemetry>true</codeflash.disableTelemetry>
    <codeflash.gitRemote>upstream</codeflash.gitRemote>
    <codeflash.ignorePaths>gen/,build/</codeflash.ignorePaths>
  </properties>
</project>
"""


class TestMavenCodeflashProperties:
    def test_reads_custom_properties(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(MAVEN_POM_WITH_PROPERTIES, encoding="utf-8")
        (tmp_path / "custom" / "src").mkdir(parents=True)
        (tmp_path / "custom" / "test").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["module_root"] == str((tmp_path / "custom" / "src").resolve())
        assert config["tests_root"] == str((tmp_path / "custom" / "test").resolve())
        assert config["disable_telemetry"] is True
        assert config["git_remote"] == "upstream"
        assert len(config["ignore_paths"]) == 2

    def test_properties_override_auto_detection(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(MAVEN_POM_WITH_PROPERTIES, encoding="utf-8")
        # Create standard dirs AND custom dirs
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "custom" / "src").mkdir(parents=True)
        (tmp_path / "custom" / "test").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        # Should use custom paths from properties, not auto-detected standard paths
        assert config["module_root"] == str((tmp_path / "custom" / "src").resolve())

    def test_no_properties_uses_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion></project>',
            encoding="utf-8",
        )
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["disable_telemetry"] is False
        assert config["git_remote"] == "origin"


# ---------------------------------------------------------------------------
# parse_java_project_config — Gradle properties
# ---------------------------------------------------------------------------


class TestGradleCodeflashProperties:
    def test_reads_gradle_properties(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        (tmp_path / "gradle.properties").write_text(
            "codeflash.moduleRoot=lib/src\ncodeflash.testsRoot=lib/test\ncodeflash.disableTelemetry=true\n",
            encoding="utf-8",
        )
        (tmp_path / "lib" / "src").mkdir(parents=True)
        (tmp_path / "lib" / "test").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["module_root"] == str((tmp_path / "lib" / "src").resolve())
        assert config["tests_root"] == str((tmp_path / "lib" / "test").resolve())
        assert config["disable_telemetry"] is True

    def test_ignores_non_codeflash_properties(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        (tmp_path / "gradle.properties").write_text(
            "org.gradle.jvmargs=-Xmx2g\ncodeflash.gitRemote=upstream\n",
            encoding="utf-8",
        )

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["git_remote"] == "upstream"

    def test_no_gradle_properties_uses_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["git_remote"] == "origin"
        assert config["disable_telemetry"] is False


# ---------------------------------------------------------------------------
# Multi-module Maven projects
# ---------------------------------------------------------------------------

PARENT_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules>
    <module>client</module>
    <module>test</module>
    <module>examples</module>
  </modules>
</project>
"""

CLIENT_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0</version>
  </parent>
  <artifactId>client</artifactId>
  <build>
    <sourceDirectory>${project.basedir}/src</sourceDirectory>
  </build>
</project>
"""

TEST_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0</version>
  </parent>
  <artifactId>test</artifactId>
  <build>
    <testSourceDirectory>${project.basedir}/src</testSourceDirectory>
  </build>
</project>
"""

EXAMPLES_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0</version>
  </parent>
  <artifactId>examples</artifactId>
  <build>
    <sourceDirectory>${project.basedir}/src</sourceDirectory>
  </build>
</project>
"""


class TestMultiModuleMaven:
    @pytest.fixture
    def multi_module_project(self, tmp_path: Path) -> Path:
        """Create a multi-module Maven project mimicking aerospike's layout."""
        (tmp_path / "pom.xml").write_text(PARENT_POM, encoding="utf-8")

        # Client module — main library with the most Java files
        client = tmp_path / "client"
        client.mkdir()
        (client / "pom.xml").write_text(CLIENT_POM, encoding="utf-8")
        client_src = client / "src" / "com" / "example" / "client"
        client_src.mkdir(parents=True)
        for i in range(10):
            (client_src / f"Class{i}.java").write_text(f"class Class{i} {{}}", encoding="utf-8")

        # Test module — test code
        test = tmp_path / "test"
        test.mkdir()
        (test / "pom.xml").write_text(TEST_POM, encoding="utf-8")
        test_src = test / "src" / "com" / "example" / "test"
        test_src.mkdir(parents=True)
        (test_src / "ClientTest.java").write_text("class ClientTest {}", encoding="utf-8")

        # Examples module — should be skipped
        examples = tmp_path / "examples"
        examples.mkdir()
        (examples / "pom.xml").write_text(EXAMPLES_POM, encoding="utf-8")
        examples_src = examples / "src" / "com" / "example"
        examples_src.mkdir(parents=True)
        (examples_src / "Example.java").write_text("class Example {}", encoding="utf-8")

        return tmp_path

    def test_detects_client_as_source_root(self, multi_module_project: Path) -> None:
        config = parse_java_project_config(multi_module_project)
        assert config is not None
        assert config["module_root"] == str(multi_module_project / "client" / "src")

    def test_detects_test_module_as_test_root(self, multi_module_project: Path) -> None:
        config = parse_java_project_config(multi_module_project)
        assert config is not None
        assert config["tests_root"] == str(multi_module_project / "test" / "src")

    def test_skips_examples_module(self, multi_module_project: Path) -> None:
        config = parse_java_project_config(multi_module_project)
        assert config is not None
        # The module_root should be client/src, not examples/src
        assert config["module_root"] == str(multi_module_project / "client" / "src")

    def test_picks_module_with_most_java_files(self, multi_module_project: Path) -> None:
        """Client has 10 .java files, examples has 1 — client should win."""
        config = parse_java_project_config(multi_module_project)
        assert config is not None
        assert "client" in config["module_root"]


# ---------------------------------------------------------------------------
# Language detection from config_parser
# ---------------------------------------------------------------------------


class TestLanguageDetectionViaConfigParser:
    def test_java_detected_from_pom_xml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        from codeflash.code_utils.config_parser import _try_parse_java_build_config

        result = _try_parse_java_build_config()
        assert result is not None
        config, project_root = result
        assert config["language"] == "java"
        assert project_root == tmp_path

    def test_java_detected_from_build_gradle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        from codeflash.code_utils.config_parser import _try_parse_java_build_config

        result = _try_parse_java_build_config()
        assert result is not None
        config, _ = result
        assert config["language"] == "java"

    def test_no_java_detected_for_python_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.codeflash]\nmodule-root='src'\ntests-root='tests'\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        from codeflash.code_utils.config_parser import _try_parse_java_build_config

        result = _try_parse_java_build_config()
        assert result is None


# ---------------------------------------------------------------------------
# Language detection from tracer
# ---------------------------------------------------------------------------


class TestTracerLanguageDetection:
    def test_detects_java_from_build_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        from codeflash.languages.base import Language
        from codeflash.tracer import _detect_non_python_language

        result = _detect_non_python_language(None)
        assert result == Language.JAVA

    def test_no_detection_without_build_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        from codeflash.tracer import _detect_non_python_language

        result = _detect_non_python_language(None)
        assert result is None

    def test_detects_java_from_file_extension(self, tmp_path: Path) -> None:
        java_file = tmp_path / "App.java"
        java_file.write_text("class App {}", encoding="utf-8")

        from argparse import Namespace

        from codeflash.languages.base import Language
        from codeflash.tracer import _detect_non_python_language

        args = Namespace(file=str(java_file))
        result = _detect_non_python_language(args)
        assert result == Language.JAVA
