"""Tests for config_writer module — Java pom.xml formatting preservation."""

from pathlib import Path


class TestWriteMavenProperties:
    """Tests for _write_maven_properties — text-based pom.xml editing."""

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

        from codeflash.setup.config_writer import _write_maven_properties

        ok, _ = _write_maven_properties(pom, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<!-- Important comment -->" in result
        assert "<codeflash.moduleRoot>src/main/java</codeflash.moduleRoot>" in result

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

        from codeflash.setup.config_writer import _write_maven_properties

        ok, _ = _write_maven_properties(pom, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert 'xmlns="http://maven.apache.org/POM/4.0.0"' in result
        # Must NOT have ns0: prefix (ElementTree bug)
        assert "ns0:" not in result

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

        from codeflash.setup.config_writer import _write_maven_properties

        ok, _ = _write_maven_properties(pom, {"module-root": "new/path"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "old/path" not in result
        assert "<codeflash.moduleRoot>new/path</codeflash.moduleRoot>" in result

    def test_creates_properties_section(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n" "  <modelVersion>4.0.0</modelVersion>\n" "</project>\n",
            encoding="utf-8",
        )

        from codeflash.setup.config_writer import _write_maven_properties

        ok, _ = _write_maven_properties(pom, {"module-root": "src/main/java"})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<properties>" in result
        assert "<codeflash.moduleRoot>src/main/java</codeflash.moduleRoot>" in result

    def test_converts_kebab_to_camelcase(self, tmp_path: Path) -> None:
        pom = tmp_path / "pom.xml"
        pom.write_text(
            "<project>\n  <properties>\n  </properties>\n</project>\n",
            encoding="utf-8",
        )

        from codeflash.setup.config_writer import _write_maven_properties

        ok, _ = _write_maven_properties(pom, {"ignore-paths": ["target", "build"]})
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<codeflash.ignorePaths>target,build</codeflash.ignorePaths>" in result


class TestRemoveJavaBuildConfig:
    """Tests for _remove_java_build_config — preserves formatting during removal."""

    def test_removes_codeflash_from_pom_preserving_others(self, tmp_path: Path) -> None:
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

        from codeflash.setup.config_writer import _remove_java_build_config

        ok, _ = _remove_java_build_config(tmp_path)
        result = pom.read_text(encoding="utf-8")

        assert ok
        assert "<!-- Keep me -->" in result
        assert "<maven.compiler.source>17</maven.compiler.source>" in result
        assert "codeflash.moduleRoot" not in result

    def test_removes_codeflash_from_gradle_properties(self, tmp_path: Path) -> None:
        gradle = tmp_path / "gradle.properties"
        gradle.write_text(
            "org.gradle.jvmargs=-Xmx2g\n"
            "# Codeflash configuration \u2014 https://docs.codeflash.ai\n"
            "codeflash.moduleRoot=src/main/java\n"
            "codeflash.testsRoot=src/test/java\n",
            encoding="utf-8",
        )

        from codeflash.setup.config_writer import _remove_java_build_config

        ok, _ = _remove_java_build_config(tmp_path)
        result = gradle.read_text(encoding="utf-8")

        assert ok
        assert "org.gradle.jvmargs=-Xmx2g" in result
        assert "codeflash." not in result
