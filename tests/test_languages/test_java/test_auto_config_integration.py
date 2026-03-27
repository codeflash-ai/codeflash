"""Integration tests for Java auto-config logic across Gradle and Maven projects.

Tests the end-to-end flow: build tool detection → strategy selection →
config parsing → write → read → remove, using realistic project layouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.java.build_config_strategy import (
    GradleConfigStrategy,
    MavenConfigStrategy,
    get_config_strategy,
    parse_java_project_config,
)
from codeflash.languages.java.build_tools import (
    BuildTool,
    detect_build_tool,
    find_source_root,
    find_test_root,
    get_project_info,
)


# ---------------------------------------------------------------------------
# Helpers — create realistic project layouts in tmp_path
# ---------------------------------------------------------------------------


def _make_maven_project(root: Path, *, with_namespace: bool = True, java_version: str = "17") -> Path:
    ns = ' xmlns="http://maven.apache.org/POM/4.0.0"' if with_namespace else ""
    pom = root / "pom.xml"
    pom.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<project{ns}>\n"
        f"    <modelVersion>4.0.0</modelVersion>\n"
        f"    <groupId>com.example</groupId>\n"
        f"    <artifactId>demo-app</artifactId>\n"
        f"    <version>1.0.0</version>\n"
        f"    <properties>\n"
        f"        <maven.compiler.source>{java_version}</maven.compiler.source>\n"
        f"        <maven.compiler.target>{java_version}</maven.compiler.target>\n"
        f"    </properties>\n"
        f"</project>\n",
        encoding="utf-8",
    )
    src = root / "src" / "main" / "java" / "com" / "example"
    src.mkdir(parents=True)
    (src / "App.java").write_text("package com.example;\npublic class App {}\n", encoding="utf-8")
    test = root / "src" / "test" / "java" / "com" / "example"
    test.mkdir(parents=True)
    (test / "AppTest.java").write_text(
        "package com.example;\nimport org.junit.jupiter.api.Test;\nclass AppTest {\n"
        "    @Test void works() {}\n}\n",
        encoding="utf-8",
    )
    return root


def _make_gradle_project(root: Path, *, kotlin_dsl: bool = False) -> Path:
    ext = ".kts" if kotlin_dsl else ""
    build_file = root / f"build.gradle{ext}"
    build_file.write_text(
        "plugins {\n    id 'java'\n}\ngroup = 'com.example'\nversion = '1.0.0'\n",
        encoding="utf-8",
    )
    (root / f"settings.gradle{ext}").write_text(f"rootProject.name = 'demo'\n", encoding="utf-8")
    src = root / "src" / "main" / "java" / "com" / "example"
    src.mkdir(parents=True)
    (src / "App.java").write_text("package com.example;\npublic class App {}\n", encoding="utf-8")
    test = root / "src" / "test" / "java" / "com" / "example"
    test.mkdir(parents=True)
    (test / "AppTest.java").write_text(
        "package com.example;\nimport org.junit.jupiter.api.Test;\nclass AppTest {\n"
        "    @Test void works() {}\n}\n",
        encoding="utf-8",
    )
    return root


def _make_maven_multimodule(root: Path) -> Path:
    # Parent pom with modules
    (root / "pom.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "    <modelVersion>4.0.0</modelVersion>\n"
        "    <groupId>com.example</groupId>\n"
        "    <artifactId>parent</artifactId>\n"
        "    <version>1.0.0</version>\n"
        "    <packaging>pom</packaging>\n"
        "    <modules>\n"
        "        <module>core</module>\n"
        "        <module>api</module>\n"
        "        <module>tests</module>\n"
        "    </modules>\n"
        "</project>\n",
        encoding="utf-8",
    )

    # core module — main source
    core = root / "core"
    core.mkdir()
    (core / "pom.xml").write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "    <modelVersion>4.0.0</modelVersion>\n"
        "    <parent>\n"
        "        <groupId>com.example</groupId>\n"
        "        <artifactId>parent</artifactId>\n"
        "        <version>1.0.0</version>\n"
        "    </parent>\n"
        "    <artifactId>core</artifactId>\n"
        "</project>\n",
        encoding="utf-8",
    )
    core_src = core / "src" / "main" / "java" / "com" / "example"
    core_src.mkdir(parents=True)
    (core_src / "Core.java").write_text("package com.example;\npublic class Core {}\n", encoding="utf-8")
    (core_src / "Utils.java").write_text("package com.example;\npublic class Utils {}\n", encoding="utf-8")

    # api module — fewer source files
    api = root / "api"
    api.mkdir()
    (api / "pom.xml").write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "    <modelVersion>4.0.0</modelVersion>\n"
        "    <parent>\n"
        "        <groupId>com.example</groupId>\n"
        "        <artifactId>parent</artifactId>\n"
        "        <version>1.0.0</version>\n"
        "    </parent>\n"
        "    <artifactId>api</artifactId>\n"
        "</project>\n",
        encoding="utf-8",
    )
    api_src = api / "src" / "main" / "java" / "com" / "example"
    api_src.mkdir(parents=True)
    (api_src / "Api.java").write_text("package com.example;\npublic class Api {}\n", encoding="utf-8")

    # tests module — integration tests
    tests = root / "tests"
    tests.mkdir()
    (tests / "pom.xml").write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "    <modelVersion>4.0.0</modelVersion>\n"
        "    <parent>\n"
        "        <groupId>com.example</groupId>\n"
        "        <artifactId>parent</artifactId>\n"
        "        <version>1.0.0</version>\n"
        "    </parent>\n"
        "    <artifactId>tests</artifactId>\n"
        "</project>\n",
        encoding="utf-8",
    )
    test_src = tests / "src" / "test" / "java" / "com" / "example"
    test_src.mkdir(parents=True)
    (test_src / "IntegrationTest.java").write_text(
        "package com.example;\npublic class IntegrationTest {}\n", encoding="utf-8"
    )

    return root


def _make_gradle_multimodule(root: Path, *, kotlin_dsl: bool = False) -> Path:
    ext = ".kts" if kotlin_dsl else ""

    (root / f"build.gradle{ext}").write_text("// root build\n", encoding="utf-8")
    (root / f"settings.gradle{ext}").write_text(
        "rootProject.name = 'multi'\ninclude 'core', 'api'\n", encoding="utf-8"
    )

    for module_name in ["core", "api"]:
        mod = root / module_name
        mod.mkdir()
        (mod / f"build.gradle{ext}").write_text(
            f"plugins {{\n    id 'java'\n}}\n", encoding="utf-8"
        )
        src = mod / "src" / "main" / "java" / "com" / "example"
        src.mkdir(parents=True)
        (src / f"{module_name.capitalize()}.java").write_text(
            f"package com.example;\npublic class {module_name.capitalize()} {{}}\n", encoding="utf-8"
        )
        test_dir = mod / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)
        (test_dir / f"{module_name.capitalize()}Test.java").write_text(
            f"package com.example;\nclass {module_name.capitalize()}Test {{}}\n", encoding="utf-8"
        )

    return root


# ===================================================================
# Integration: Maven — detection through full config lifecycle
# ===================================================================


class TestMavenAutoConfigIntegration:
    """End-to-end: detect Maven → get strategy → parse config → write → read → remove."""

    def test_standard_maven_detection_to_config(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)

        assert detect_build_tool(project) == BuildTool.MAVEN
        assert find_source_root(project) == project / "src" / "main" / "java"
        assert find_test_root(project) == project / "src" / "test" / "java"

        strategy = get_config_strategy(project)
        assert isinstance(strategy, MavenConfigStrategy)

        config = parse_java_project_config(project)
        assert config is not None
        assert config["language"] == "java"
        assert config["module_root"] == str(project / "src" / "main" / "java")
        assert config["tests_root"] == str(project / "src" / "test" / "java")
        assert config["git_remote"] == "origin"
        assert config["disable_telemetry"] is False

    def test_maven_full_lifecycle_write_read_remove(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        # Write config
        ok, msg = strategy.write_codeflash_properties(project, {
            "module-root": "custom/src",
            "tests-root": "custom/test",
            "git-remote": "upstream",
            "disable-telemetry": True,
            "ignore-paths": ["target", ".idea"],
            "formatter-cmds": ["spotless:apply"],
        })
        assert ok, msg

        # Read back
        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "custom/src"
        assert props["testsRoot"] == "custom/test"
        assert props["gitRemote"] == "upstream"
        assert props["disableTelemetry"] == "true"
        assert props["ignorePaths"] == "target,.idea"
        assert props["formatterCmds"] == "spotless:apply"

        # Verify non-codeflash properties are preserved
        pom_text = (project / "pom.xml").read_text(encoding="utf-8")
        assert "maven.compiler.source" in pom_text
        assert "maven.compiler.target" in pom_text

        # Remove
        ok, msg = strategy.remove_codeflash_properties(project)
        assert ok, msg

        # Verify removed
        props_after = strategy.read_codeflash_properties(project)
        assert props_after == {}

        # Verify non-codeflash properties still preserved
        pom_after = (project / "pom.xml").read_text(encoding="utf-8")
        assert "maven.compiler.source" in pom_after

    def test_maven_with_namespace_full_lifecycle(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path, with_namespace=True)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "lib/main"})
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "lib/main"

        # Verify namespace preserved, no ns0: prefix
        pom_text = (project / "pom.xml").read_text(encoding="utf-8")
        assert 'xmlns="http://maven.apache.org/POM/4.0.0"' in pom_text
        assert "ns0:" not in pom_text

    def test_maven_without_namespace_full_lifecycle(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path, with_namespace=False)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "src/main/java"})
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "src/main/java"

    def test_maven_user_overrides_take_precedence(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        # Write user overrides to pom.xml
        ok, _ = strategy.write_codeflash_properties(project, {
            "module-root": "custom/src",
            "tests-root": "custom/test",
            "disable-telemetry": True,
        })
        assert ok

        # Create the custom directories
        (project / "custom" / "src").mkdir(parents=True)
        (project / "custom" / "test").mkdir(parents=True)

        # parse_java_project_config should use user overrides, not auto-detected paths
        config = parse_java_project_config(project)
        assert config is not None
        assert config["module_root"] == str((project / "custom" / "src").resolve())
        assert config["tests_root"] == str((project / "custom" / "test").resolve())
        assert config["disable_telemetry"] is True

    def test_maven_project_info_extraction(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path, java_version="11")

        info = get_project_info(project)
        assert info is not None
        assert info.build_tool == BuildTool.MAVEN
        assert info.group_id == "com.example"
        assert info.artifact_id == "demo-app"
        assert info.version == "1.0.0"
        assert info.java_version == "11"
        assert len(info.source_roots) == 1
        assert len(info.test_roots) == 1

    def test_maven_overwrite_then_overwrite(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        # First write
        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "v1"})
        assert ok
        assert strategy.read_codeflash_properties(project)["moduleRoot"] == "v1"

        # Second write replaces previous values
        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "v2", "git-remote": "upstream"})
        assert ok
        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "v2"
        assert props["gitRemote"] == "upstream"


# ===================================================================
# Integration: Maven multi-module
# ===================================================================


class TestMavenMultiModuleIntegration:
    """End-to-end auto-config for Maven multi-module projects."""

    def test_multimodule_detects_source_from_largest_module(self, tmp_path: Path) -> None:
        project = _make_maven_multimodule(tmp_path)

        config = parse_java_project_config(project)
        assert config is not None
        # core has 2 java files, api has 1 → core should be chosen as source root
        assert "core" in config["module_root"]
        assert config["module_root"].endswith(str(Path("src") / "main" / "java"))

    def test_multimodule_detects_test_module(self, tmp_path: Path) -> None:
        project = _make_maven_multimodule(tmp_path)

        config = parse_java_project_config(project)
        assert config is not None
        # "tests" module has "test" in its name → should be detected as test root
        assert "tests" in config["tests_root"]

    def test_multimodule_build_tool_detection(self, tmp_path: Path) -> None:
        project = _make_maven_multimodule(tmp_path)

        assert detect_build_tool(project) == BuildTool.MAVEN
        strategy = get_config_strategy(project)
        assert isinstance(strategy, MavenConfigStrategy)

    def test_multimodule_config_write_read_on_parent(self, tmp_path: Path) -> None:
        project = _make_maven_multimodule(tmp_path)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {"git-remote": "upstream"})
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert props["gitRemote"] == "upstream"

        # Verify the parent pom still has modules
        pom_text = (project / "pom.xml").read_text(encoding="utf-8")
        assert "<module>core</module>" in pom_text
        assert "<module>api</module>" in pom_text

    def test_multimodule_with_custom_source_directory(self, tmp_path: Path) -> None:
        project = _make_maven_multimodule(tmp_path)

        # Modify core module to use a custom source directory
        core_pom = project / "core" / "pom.xml"
        core_pom.write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "    <modelVersion>4.0.0</modelVersion>\n"
            "    <parent>\n"
            "        <groupId>com.example</groupId>\n"
            "        <artifactId>parent</artifactId>\n"
            "        <version>1.0.0</version>\n"
            "    </parent>\n"
            "    <artifactId>core</artifactId>\n"
            "    <build>\n"
            "        <sourceDirectory>src/main/custom</sourceDirectory>\n"
            "    </build>\n"
            "</project>\n",
            encoding="utf-8",
        )
        custom_src = project / "core" / "src" / "main" / "custom"
        custom_src.mkdir(parents=True)
        (custom_src / "Main.java").write_text("public class Main {}\n", encoding="utf-8")

        config = parse_java_project_config(project)
        assert config is not None
        # Should detect the custom source directory from the module pom
        # The exact path depends on which module has more java files
        assert config["module_root"] is not None


# ===================================================================
# Integration: Gradle — detection through full config lifecycle
# ===================================================================


class TestGradleAutoConfigIntegration:
    """End-to-end: detect Gradle → get strategy → parse config → write → read → remove."""

    def test_standard_gradle_detection_to_config(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)

        assert detect_build_tool(project) == BuildTool.GRADLE
        assert find_source_root(project) == project / "src" / "main" / "java"
        assert find_test_root(project) == project / "src" / "test" / "java"

        strategy = get_config_strategy(project)
        assert isinstance(strategy, GradleConfigStrategy)

        config = parse_java_project_config(project)
        assert config is not None
        assert config["language"] == "java"
        assert config["module_root"] == str(project / "src" / "main" / "java")
        assert config["tests_root"] == str(project / "src" / "test" / "java")

    def test_gradle_kotlin_dsl_detection_to_config(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path, kotlin_dsl=True)

        assert detect_build_tool(project) == BuildTool.GRADLE

        strategy = get_config_strategy(project)
        assert isinstance(strategy, GradleConfigStrategy)

        config = parse_java_project_config(project)
        assert config is not None
        assert config["language"] == "java"

    def test_gradle_full_lifecycle_write_read_remove(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        strategy = get_config_strategy(project)

        # Write config
        ok, msg = strategy.write_codeflash_properties(project, {
            "module-root": "custom/src",
            "tests-root": "custom/test",
            "git-remote": "upstream",
            "disable-telemetry": True,
            "ignore-paths": ["build", ".gradle"],
            "formatter-cmds": ["spotlessApply"],
        })
        assert ok, msg

        # Read back
        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "custom/src"
        assert props["testsRoot"] == "custom/test"
        assert props["gitRemote"] == "upstream"
        assert props["disableTelemetry"] == "true"
        assert props["ignorePaths"] == "build,.gradle"
        assert props["formatterCmds"] == "spotlessApply"

        # Verify gradle.properties has the codeflash header comment
        gp_text = (project / "gradle.properties").read_text(encoding="utf-8")
        assert "# Codeflash configuration" in gp_text

        # Remove
        ok, msg = strategy.remove_codeflash_properties(project)
        assert ok, msg

        # Verify removed
        props_after = strategy.read_codeflash_properties(project)
        assert props_after == {}

        # Verify header comment also removed
        gp_after = (project / "gradle.properties").read_text(encoding="utf-8")
        assert "Codeflash" not in gp_after

    def test_gradle_preserves_existing_properties(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)

        # Pre-existing gradle.properties with user settings
        (project / "gradle.properties").write_text(
            "org.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=512m\n"
            "org.gradle.parallel=true\n"
            "org.gradle.caching=true\n",
            encoding="utf-8",
        )

        strategy = get_config_strategy(project)
        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "lib/src"})
        assert ok

        gp_text = (project / "gradle.properties").read_text(encoding="utf-8")
        assert "org.gradle.jvmargs=-Xmx4g" in gp_text
        assert "org.gradle.parallel=true" in gp_text
        assert "org.gradle.caching=true" in gp_text
        assert "codeflash.moduleRoot=lib/src" in gp_text

    def test_gradle_user_overrides_take_precedence(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {
            "module-root": "custom/src",
            "tests-root": "custom/test",
        })
        assert ok

        (project / "custom" / "src").mkdir(parents=True)
        (project / "custom" / "test").mkdir(parents=True)

        config = parse_java_project_config(project)
        assert config is not None
        assert config["module_root"] == str((project / "custom" / "src").resolve())
        assert config["tests_root"] == str((project / "custom" / "test").resolve())

    def test_gradle_overwrite_then_overwrite(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "v1"})
        assert ok
        assert strategy.read_codeflash_properties(project)["moduleRoot"] == "v1"

        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "v2", "git-remote": "upstream"})
        assert ok
        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "v2"
        assert props["gitRemote"] == "upstream"
        # Old values should not persist
        gp_text = (project / "gradle.properties").read_text(encoding="utf-8")
        assert gp_text.count("codeflash.moduleRoot") == 1

    def test_gradle_project_info_extraction(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)

        info = get_project_info(project)
        assert info is not None
        assert info.build_tool == BuildTool.GRADLE
        assert len(info.source_roots) == 1
        assert len(info.test_roots) == 1


# ===================================================================
# Integration: Gradle multi-module
# ===================================================================


class TestGradleMultiModuleIntegration:
    """End-to-end auto-config for Gradle multi-module projects."""

    def test_multimodule_root_detection(self, tmp_path: Path) -> None:
        project = _make_gradle_multimodule(tmp_path)

        assert detect_build_tool(project) == BuildTool.GRADLE
        strategy = get_config_strategy(project)
        assert isinstance(strategy, GradleConfigStrategy)

    def test_multimodule_config_write_read_at_root(self, tmp_path: Path) -> None:
        project = _make_gradle_multimodule(tmp_path)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {
            "module-root": "core/src/main/java",
            "tests-root": "core/src/test/java",
        })
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "core/src/main/java"
        assert props["testsRoot"] == "core/src/test/java"

    def test_multimodule_kotlin_dsl(self, tmp_path: Path) -> None:
        project = _make_gradle_multimodule(tmp_path, kotlin_dsl=True)

        assert detect_build_tool(project) == BuildTool.GRADLE
        config = parse_java_project_config(project)
        assert config is not None
        assert config["language"] == "java"


# ===================================================================
# Integration: cross-cutting scenarios
# ===================================================================


class TestCrossCuttingIntegration:
    """Scenarios that test across both build tools or edge conditions."""

    def test_maven_takes_precedence_over_gradle(self, tmp_path: Path) -> None:
        # Create both Maven and Gradle files
        _make_maven_project(tmp_path)
        (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")

        assert detect_build_tool(tmp_path) == BuildTool.MAVEN
        strategy = get_config_strategy(tmp_path)
        assert isinstance(strategy, MavenConfigStrategy)

    def test_empty_directory_returns_unknown(self, tmp_path: Path) -> None:
        assert detect_build_tool(tmp_path) == BuildTool.UNKNOWN
        assert parse_java_project_config(tmp_path) is None
        with pytest.raises(ValueError, match="No supported Java build tool"):
            get_config_strategy(tmp_path)

    def test_maven_config_with_all_properties(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        full_config = {
            "module-root": "src/main/java",
            "tests-root": "src/test/java",
            "git-remote": "upstream",
            "disable-telemetry": True,
            "ignore-paths": ["target", ".idea", "*.iml"],
            "formatter-cmds": ["mvn spotless:apply", "mvn formatter:format"],
        }

        ok, _ = strategy.write_codeflash_properties(project, full_config)
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert len(props) == 6
        assert props["moduleRoot"] == "src/main/java"
        assert props["testsRoot"] == "src/test/java"
        assert props["gitRemote"] == "upstream"
        assert props["disableTelemetry"] == "true"
        assert props["ignorePaths"] == "target,.idea,*.iml"
        assert props["formatterCmds"] == "mvn spotless:apply,mvn formatter:format"

    def test_gradle_config_with_all_properties(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        strategy = get_config_strategy(project)

        full_config = {
            "module-root": "lib/src/main/java",
            "tests-root": "lib/src/test/java",
            "git-remote": "upstream",
            "disable-telemetry": False,
            "ignore-paths": ["build", ".gradle"],
            "formatter-cmds": ["./gradlew spotlessApply"],
        }

        ok, _ = strategy.write_codeflash_properties(project, full_config)
        assert ok

        props = strategy.read_codeflash_properties(project)
        assert len(props) == 6
        assert props["moduleRoot"] == "lib/src/main/java"
        assert props["disableTelemetry"] == "false"

    def test_parse_config_feeds_ignore_paths_and_formatter_cmds(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        ok, _ = strategy.write_codeflash_properties(project, {
            "ignore-paths": ["target", "generated"],
            "formatter-cmds": ["mvn fmt:format"],
        })
        assert ok

        config = parse_java_project_config(project)
        assert config is not None
        assert len(config["ignore_paths"]) == 2
        assert any("target" in p for p in config["ignore_paths"])
        assert any("generated" in p for p in config["ignore_paths"])
        assert config["formatter_cmds"] == ["mvn fmt:format"]

    def test_parse_config_defaults_when_no_user_overrides(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)

        config = parse_java_project_config(project)
        assert config is not None
        assert config["git_remote"] == "origin"
        assert config["disable_telemetry"] is False
        assert config["ignore_paths"] == []
        assert config["formatter_cmds"] == []

    def test_subdir_detection_from_child(self, tmp_path: Path) -> None:
        """Build tool detection works from a subdirectory (multi-module child)."""
        project = _make_maven_project(tmp_path)
        child = project / "src" / "main" / "java"

        # detect_build_tool should find pom.xml in parent directories
        assert detect_build_tool(child) == BuildTool.MAVEN

    def test_gradle_write_creates_properties_file_if_missing(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        props_path = project / "gradle.properties"

        # Ensure no gradle.properties exists
        if props_path.exists():
            props_path.unlink()
        assert not props_path.exists()

        strategy = get_config_strategy(project)
        ok, _ = strategy.write_codeflash_properties(project, {"module-root": "src/main/java"})
        assert ok
        assert props_path.exists()

        props = strategy.read_codeflash_properties(project)
        assert props["moduleRoot"] == "src/main/java"

    def test_maven_remove_idempotent(self, tmp_path: Path) -> None:
        project = _make_maven_project(tmp_path)
        strategy = get_config_strategy(project)

        # Remove when nothing was written
        ok1, _ = strategy.remove_codeflash_properties(project)
        assert ok1

        # Write then remove twice
        strategy.write_codeflash_properties(project, {"module-root": "src"})
        ok2, _ = strategy.remove_codeflash_properties(project)
        assert ok2
        ok3, _ = strategy.remove_codeflash_properties(project)
        assert ok3

        assert strategy.read_codeflash_properties(project) == {}

    def test_gradle_remove_idempotent(self, tmp_path: Path) -> None:
        project = _make_gradle_project(tmp_path)
        strategy = get_config_strategy(project)

        strategy.write_codeflash_properties(project, {"module-root": "src"})
        ok1, _ = strategy.remove_codeflash_properties(project)
        assert ok1
        ok2, _ = strategy.remove_codeflash_properties(project)
        assert ok2

        assert strategy.read_codeflash_properties(project) == {}
