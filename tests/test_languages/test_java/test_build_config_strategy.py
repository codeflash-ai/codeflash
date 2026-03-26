"""Tests for BuildConfigStrategy — Maven (lxml) and Gradle config read/write/remove."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.java.build_config_strategy import (
    GradleConfigStrategy,
    MavenConfigStrategy,
    get_config_strategy,
    parse_java_project_config,
)


# ---------------------------------------------------------------------------
# MavenConfigStrategy — read
# ---------------------------------------------------------------------------


class TestMavenRead:
    def test_reads_codeflash_properties_with_namespace(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "    <codeflash.moduleRoot>custom/src</codeflash.moduleRoot>\n"
            "    <codeflash.testsRoot>custom/test</codeflash.testsRoot>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        result = MavenConfigStrategy().read_codeflash_properties(tmp_path)
        assert result == {"moduleRoot": "custom/src", "testsRoot": "custom/test"}

    def test_reads_codeflash_properties_without_namespace(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n"
            "  <properties>\n"
            "    <codeflash.gitRemote>upstream</codeflash.gitRemote>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        result = MavenConfigStrategy().read_codeflash_properties(tmp_path)
        assert result == {"gitRemote": "upstream"}

    def test_returns_empty_when_no_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text("<project></project>\n", encoding="utf-8")
        assert MavenConfigStrategy().read_codeflash_properties(tmp_path) == {}

    def test_returns_empty_when_no_pom(self, tmp_path: Path) -> None:
        assert MavenConfigStrategy().read_codeflash_properties(tmp_path) == {}

    def test_ignores_non_codeflash_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n"
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "    <codeflash.moduleRoot>src</codeflash.moduleRoot>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        result = MavenConfigStrategy().read_codeflash_properties(tmp_path)
        assert "maven.compiler.source" not in result
        assert result == {"moduleRoot": "src"}


# ---------------------------------------------------------------------------
# MavenConfigStrategy — write
# ---------------------------------------------------------------------------


class TestMavenWrite:
    def test_preserves_comments(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<project>\n"
            "  <!-- Important comment -->\n"
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<!-- Important comment -->" in result
        assert "codeflash.moduleRoot" in result

    def test_preserves_namespace(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert 'xmlns="http://maven.apache.org/POM/4.0.0"' in result
        # Must NOT have ns0: prefix (ElementTree bug — lxml avoids this)
        assert "ns0:" not in result

    def test_preserves_existing_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n"
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "src"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<maven.compiler.source>17</maven.compiler.source>" in result
        assert "codeflash.moduleRoot" in result

    def test_updates_existing_codeflash_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n"
            "  <properties>\n"
            "    <codeflash.moduleRoot>old/path</codeflash.moduleRoot>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "new/path"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "old/path" not in result
        assert "<codeflash.moduleRoot>new/path</codeflash.moduleRoot>" in result

    def test_creates_properties_section(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text("<project>\n  <modelVersion>4.0.0</modelVersion>\n</project>\n", encoding="utf-8")

        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "properties" in result
        assert "codeflash.moduleRoot" in result

    def test_converts_kebab_to_camelcase(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text("<project>\n  <properties>\n  </properties>\n</project>\n", encoding="utf-8")

        ok, _ = MavenConfigStrategy().write_codeflash_properties(
            tmp_path, {"ignore-paths": ["target", "build"]}
        )
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<codeflash.ignorePaths>target,build</codeflash.ignorePaths>" in result

    def test_handles_boolean_values(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text("<project>\n  <properties>\n  </properties>\n</project>\n", encoding="utf-8")

        ok, _ = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"disable-telemetry": True})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<codeflash.disableTelemetry>true</codeflash.disableTelemetry>" in result

    def test_returns_error_when_no_pom(self, tmp_path: Path) -> None:
        ok, msg = MavenConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "src"})
        assert not ok
        assert "No pom.xml" in msg


# ---------------------------------------------------------------------------
# MavenConfigStrategy — remove
# ---------------------------------------------------------------------------


class TestMavenRemove:
    def test_removes_only_codeflash_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n"
            "  <!-- Keep me -->\n"
            "  <properties>\n"
            "    <maven.compiler.source>17</maven.compiler.source>\n"
            "    <codeflash.moduleRoot>src/main/java</codeflash.moduleRoot>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().remove_codeflash_properties(tmp_path)
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<!-- Keep me -->" in result
        assert "<maven.compiler.source>17</maven.compiler.source>" in result
        assert "codeflash.moduleRoot" not in result

    def test_preserves_comments_after_removal(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<project>\n"
            "  <!-- Project comment -->\n"
            "  <properties>\n"
            "    <!-- Property comment -->\n"
            "    <codeflash.moduleRoot>src</codeflash.moduleRoot>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().remove_codeflash_properties(tmp_path)
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<!-- Project comment -->" in result
        assert "<!-- Property comment -->" in result
        assert "codeflash" not in result

    def test_noop_when_no_codeflash_properties(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n  <properties>\n    <foo>bar</foo>\n  </properties>\n</project>\n",
            encoding="utf-8",
        )
        ok, _ = MavenConfigStrategy().remove_codeflash_properties(tmp_path)
        assert ok


# ---------------------------------------------------------------------------
# MavenConfigStrategy — roundtrip
# ---------------------------------------------------------------------------


class TestMavenRoundtrip:
    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n  <properties>\n  </properties>\n</project>\n",
            encoding="utf-8",
        )
        strategy = MavenConfigStrategy()
        strategy.write_codeflash_properties(
            tmp_path,
            {"module-root": "client/src", "git-remote": "upstream", "disable-telemetry": True},
        )
        result = strategy.read_codeflash_properties(tmp_path)
        assert result["moduleRoot"] == "client/src"
        assert result["gitRemote"] == "upstream"
        assert result["disableTelemetry"] == "true"


# ---------------------------------------------------------------------------
# GradleConfigStrategy
# ---------------------------------------------------------------------------


class TestGradleRead:
    def test_reads_gradle_properties(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        (tmp_path / "gradle.properties").write_text(
            "org.gradle.jvmargs=-Xmx2g\ncodeflash.moduleRoot=lib/src\ncodeflash.disableTelemetry=true\n",
            encoding="utf-8",
        )
        result = GradleConfigStrategy().read_codeflash_properties(tmp_path)
        assert result == {"moduleRoot": "lib/src", "disableTelemetry": "true"}

    def test_ignores_non_codeflash(self, tmp_path: Path) -> None:
        (tmp_path / "gradle.properties").write_text(
            "org.gradle.jvmargs=-Xmx2g\ncodeflash.gitRemote=upstream\n",
            encoding="utf-8",
        )
        result = GradleConfigStrategy().read_codeflash_properties(tmp_path)
        assert "org.gradle.jvmargs" not in result
        assert result == {"gitRemote": "upstream"}

    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        assert GradleConfigStrategy().read_codeflash_properties(tmp_path) == {}


class TestGradleWrite:
    def test_writes_gradle_properties(self, tmp_path: Path) -> None:
        (tmp_path / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx2g\n", encoding="utf-8")
        ok, _ = GradleConfigStrategy().write_codeflash_properties(
            tmp_path, {"module-root": "lib/src", "disable-telemetry": True}
        )
        result = (tmp_path / "gradle.properties").read_text(encoding="utf-8")

        assert ok
        assert "org.gradle.jvmargs=-Xmx2g" in result
        assert "codeflash.moduleRoot=lib/src" in result
        assert "codeflash.disableTelemetry=true" in result

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        ok, _ = GradleConfigStrategy().write_codeflash_properties(tmp_path, {"git-remote": "upstream"})
        result = (tmp_path / "gradle.properties").read_text(encoding="utf-8")

        assert ok
        assert "codeflash.gitRemote=upstream" in result

    def test_updates_existing_codeflash_properties(self, tmp_path: Path) -> None:
        (tmp_path / "gradle.properties").write_text(
            "codeflash.moduleRoot=old\ncodeflash.gitRemote=origin\n",
            encoding="utf-8",
        )
        ok, _ = GradleConfigStrategy().write_codeflash_properties(tmp_path, {"module-root": "new"})
        result = (tmp_path / "gradle.properties").read_text(encoding="utf-8")

        assert ok
        assert "codeflash.moduleRoot=new" in result
        assert "old" not in result


class TestGradleRemove:
    def test_removes_codeflash_from_gradle_properties(self, tmp_path: Path) -> None:
        (tmp_path / "gradle.properties").write_text(
            "org.gradle.jvmargs=-Xmx2g\n"
            "# Codeflash configuration \u2014 https://docs.codeflash.ai\n"
            "codeflash.moduleRoot=src/main/java\n",
            encoding="utf-8",
        )
        ok, _ = GradleConfigStrategy().remove_codeflash_properties(tmp_path)
        result = (tmp_path / "gradle.properties").read_text(encoding="utf-8")

        assert ok
        assert "org.gradle.jvmargs=-Xmx2g" in result
        assert "codeflash." not in result

    def test_noop_when_no_file(self, tmp_path: Path) -> None:
        ok, _ = GradleConfigStrategy().remove_codeflash_properties(tmp_path)
        assert ok


class TestGradleRoundtrip:
    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        strategy = GradleConfigStrategy()
        strategy.write_codeflash_properties(tmp_path, {"module-root": "lib/src", "git-remote": "upstream"})
        result = strategy.read_codeflash_properties(tmp_path)
        assert result["moduleRoot"] == "lib/src"
        assert result["gitRemote"] == "upstream"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetConfigStrategy:
    def test_returns_maven_for_pom(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        assert isinstance(get_config_strategy(tmp_path), MavenConfigStrategy)

    def test_returns_gradle_for_build_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("", encoding="utf-8")
        assert isinstance(get_config_strategy(tmp_path), GradleConfigStrategy)

    def test_raises_for_unknown(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No supported Java build tool"):
            get_config_strategy(tmp_path)


# ---------------------------------------------------------------------------
# parse_java_project_config
# ---------------------------------------------------------------------------


class TestParseJavaProjectConfig:
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

    def test_returns_none_for_non_java(self, tmp_path: Path) -> None:
        assert parse_java_project_config(tmp_path) is None

    def test_maven_with_custom_properties(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "  <properties>\n"
            "    <codeflash.moduleRoot>custom/src</codeflash.moduleRoot>\n"
            "    <codeflash.testsRoot>custom/test</codeflash.testsRoot>\n"
            "    <codeflash.disableTelemetry>true</codeflash.disableTelemetry>\n"
            "  </properties>\n"
            "</project>\n",
            encoding="utf-8",
        )
        (tmp_path / "custom" / "src").mkdir(parents=True)
        (tmp_path / "custom" / "test").mkdir(parents=True)

        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["module_root"] == str((tmp_path / "custom" / "src").resolve())
        assert config["tests_root"] == str((tmp_path / "custom" / "test").resolve())
        assert config["disable_telemetry"] is True

    def test_defaults_when_dirs_missing(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        config = parse_java_project_config(tmp_path)
        assert config is not None
        assert config["module_root"] == str(tmp_path / "src" / "main" / "java")
