"""Test that spurious Java configs (like codeflash-java-runtime/) don't crash --file optimizations.

Reproduces the bug where running:
  codeflash --file tests/.../Calculator.java --module-root tests/.../src/main/java --tests-root tests/.../src/test/java

from a repo that contains codeflash-java-runtime/ (which has a pom.xml) crashes with:
  ValueError: File .../Calculator.java is not within the project root .../codeflash-java-runtime
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import tomlkit

from codeflash.code_utils.config_parser import LanguageConfig, find_all_config_files
from codeflash.languages.language_enum import Language


def write_toml(path: Path, data: dict) -> None:
    path.write_text(tomlkit.dumps(data), encoding="utf-8")


class TestSpuriousJavaConfigDiscovery:
    def test_subdirectory_with_pom_picked_up_as_java_config(self, tmp_path: Path, monkeypatch) -> None:
        """Verify the bug scenario: a subdir with pom.xml gets picked up as Java config."""
        # Root has a pyproject.toml (Python project, like codeflash itself)
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        (tmp_path / "src").mkdir()

        # Subdirectory mimicking codeflash-java-runtime (has pom.xml)
        runtime_dir = tmp_path / "codeflash-java-runtime"
        runtime_dir.mkdir()
        (runtime_dir / "pom.xml").write_text("<project/>", encoding="utf-8")

        java_config = {"language": "java", "module_root": str(runtime_dir / "src/main/java")}
        monkeypatch.chdir(tmp_path)

        with patch("codeflash.code_utils.config_parser._parse_java_config_for_dir", return_value=java_config):
            result = find_all_config_files()

        # This demonstrates the bug: codeflash-java-runtime gets picked up
        java_configs = [r for r in result if r.language == Language.JAVA]
        assert len(java_configs) == 1
        assert java_configs[0].config_path == runtime_dir

    def test_file_flag_with_spurious_java_config_should_not_crash(self, tmp_path: Path, monkeypatch) -> None:
        """The actual bug: --file Calculator.java crashes because project_root points to codeflash-java-runtime."""
        # Setup: Python project at root with codeflash-java-runtime subdir
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "codeflash"}}})
        (tmp_path / "codeflash").mkdir()

        # Spurious Java subdir (like codeflash-java-runtime)
        runtime_dir = tmp_path / "codeflash-java-runtime"
        runtime_dir.mkdir()
        (runtime_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        (runtime_dir / "src" / "main" / "java").mkdir(parents=True)

        # Actual target: Java fixture in a completely different location
        fixture_dir = tmp_path / "tests" / "fixtures" / "java_maven"
        (fixture_dir / "src" / "main" / "java" / "com" / "example").mkdir(parents=True)
        (fixture_dir / "src" / "test" / "java").mkdir(parents=True)
        target_file = fixture_dir / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        target_file.write_text(
            "public class Calculator { public int add(int a, int b) { return a + b; } }", encoding="utf-8"
        )

        monkeypatch.chdir(tmp_path)

        runtime_java_config = {"language": "java", "module_root": str(runtime_dir / "src" / "main" / "java")}

        from codeflash.cli_cmds.cli import apply_language_config

        # Simulate what main() does: discover configs, filter by language, apply config
        with patch("codeflash.code_utils.config_parser._parse_java_config_for_dir", return_value=runtime_java_config):
            configs = find_all_config_files()

        java_configs = [c for c in configs if c.language == Language.JAVA]
        assert len(java_configs) == 1

        # Now simulate what happens: user provided --file and --module-root explicitly
        from tests.test_multi_language_orchestration import make_base_args

        args = make_base_args(
            file=str(target_file),
            module_root=str(fixture_dir / "src" / "main" / "java"),
            tests_root=str(fixture_dir / "src" / "test" / "java"),
        )

        # This is where it crashes: apply_language_config sets project_root to
        # codeflash-java-runtime/ (config_path), then later module_name_from_file_path
        # fails because Calculator.java is not within codeflash-java-runtime/
        result = apply_language_config(args, java_configs[0])

        # The bug: project_root is set to the spurious config path, not the user's target
        # After the fix, the file should be within project_root
        resolved_file = Path(args.file).resolve()
        assert resolved_file.is_relative_to(result.project_root), (
            f"File {resolved_file} is not within project_root {result.project_root}"
        )


class TestFileNotWithinDiscoveredProjectRoot:
    def test_orchestrator_skips_config_when_file_outside_project_root(self, tmp_path: Path, monkeypatch) -> None:
        """When --file points to a file outside a discovered config's project root, skip that config."""
        # Two Java configs: one correct, one spurious
        correct_dir = tmp_path / "my-java-project"
        (correct_dir / "src" / "main" / "java").mkdir(parents=True)
        (correct_dir / "src" / "test" / "java").mkdir(parents=True)
        target_file = correct_dir / "src" / "main" / "java" / "Foo.java"
        target_file.write_text("public class Foo {}", encoding="utf-8")

        spurious_dir = tmp_path / "runtime-lib"
        (spurious_dir / "src" / "main" / "java").mkdir(parents=True)
        (spurious_dir / "src" / "test" / "java").mkdir(parents=True)

        correct_config = LanguageConfig(
            config={
                "module_root": str(correct_dir / "src/main/java"),
                "tests_root": str(correct_dir / "src/test/java"),
            },
            config_path=correct_dir,
            language=Language.JAVA,
        )
        spurious_config = LanguageConfig(
            config={
                "module_root": str(spurious_dir / "src/main/java"),
                "tests_root": str(spurious_dir / "src/test/java"),
            },
            config_path=spurious_dir,
            language=Language.JAVA,
        )

        monkeypatch.chdir(tmp_path)

        from codeflash.main import filter_configs_for_file

        # After the fix, this function should exist and filter out spurious configs
        filtered = filter_configs_for_file([spurious_config, correct_config], str(target_file))
        assert len(filtered) == 1
        assert filtered[0].config_path == correct_dir
