"""Tests for GradleStrategy._resolve_project_classpath and _compile_dependency_modules."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.gradle_strategy import GradleStrategy


@pytest.fixture()
def strategy():
    return GradleStrategy()


@pytest.fixture()
def build_root(tmp_path):
    """Create a multi-module Gradle project layout with missing JARs."""
    root = (tmp_path / "project").resolve()
    root.mkdir()

    # Module A: compiled (has classes but no JAR)
    mod_a = root / "module-a"
    (mod_a / "build" / "classes" / "java" / "main" / "com" / "example").mkdir(parents=True)
    (mod_a / "build" / "classes" / "java" / "main" / "com" / "example" / "A.class").write_bytes(b"")
    (mod_a / "build" / "resources" / "main" / "META-INF").mkdir(parents=True)
    (mod_a / "build" / "resources" / "main" / "META-INF" / "services.txt").write_bytes(b"")

    # Module B: compiled with Kotlin (has kotlin classes but no JAR)
    mod_b = root / "module-b"
    (mod_b / "build" / "classes" / "kotlin" / "main" / "com" / "example").mkdir(parents=True)
    (mod_b / "build" / "classes" / "kotlin" / "main" / "com" / "example" / "B.class").write_bytes(b"")
    (mod_b / "build" / "classes" / "java" / "main" / "com" / "example").mkdir(parents=True)
    (mod_b / "build" / "classes" / "java" / "main" / "com" / "example" / "BHelper.class").write_bytes(b"")

    # Module C: uncompiled (no build/classes at all — testRuntimeOnly dep)
    mod_c = root / "module-c"
    mod_c.mkdir()

    # External dependency JAR (exists)
    ext_dir = tmp_path / "gradle-cache"
    ext_dir.mkdir()
    ext_jar = ext_dir / "some-lib-1.0.jar"
    ext_jar.write_bytes(b"")

    return root


def _make_classpath(build_root: Path, tmp_path: Path) -> str:
    """Build a classpath string mimicking Gradle's testRuntimeClasspath output."""
    sep = os.pathsep
    ext_jar = str(tmp_path / "gradle-cache" / "some-lib-1.0.jar")
    return sep.join([
        str(build_root / "module-a" / "build" / "libs" / "module-a-1.0.jar"),
        ext_jar,
        str(build_root / "module-b" / "build" / "libs" / "module-b-1.0.jar"),
        str(build_root / "module-c" / "build" / "libs" / "module-c-1.0.jar"),
    ])


def test_replaces_missing_jars_with_class_dirs(strategy, build_root, tmp_path):
    """Missing project JARs should be replaced with class/resource directories."""
    classpath = _make_classpath(build_root, tmp_path)

    with (
        patch.object(GradleStrategy, "find_executable", return_value="gradle"),
        patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout") as mock_run,
    ):
        # Mock the compilation of module-c
        mock_run.return_value = subprocess.CompletedProcess(args=["gradle"], returncode=0, stdout="", stderr="")
        # Simulate that compilation creates the class directory
        (build_root / "module-c" / "build" / "classes" / "java" / "main").mkdir(parents=True)

        result = strategy._resolve_project_classpath(classpath, build_root, {}, timeout=60)

    entries = result.split(os.pathsep)

    ext_jar = str(tmp_path / "gradle-cache" / "some-lib-1.0.jar")
    mod_a_java = str(build_root / "module-a" / "build" / "classes" / "java" / "main")
    mod_a_resources = str(build_root / "module-a" / "build" / "resources" / "main")
    mod_b_kotlin = str(build_root / "module-b" / "build" / "classes" / "kotlin" / "main")
    mod_b_java = str(build_root / "module-b" / "build" / "classes" / "java" / "main")
    mod_c_java = str(build_root / "module-c" / "build" / "classes" / "java" / "main")

    # Full equality: module-a JAR → java/main + resources/main,
    # external JAR preserved, module-b JAR → kotlin/main + java/main,
    # module-c JAR → java/main (compiled by mock)
    assert entries == [
        mod_a_java,
        mod_a_resources,
        ext_jar,
        mod_b_kotlin,
        mod_b_java,
        mod_c_java,
    ]


def test_compiles_uncompiled_modules(strategy, build_root, tmp_path):
    """Modules with no compiled classes should trigger a Gradle compilation."""
    classpath = _make_classpath(build_root, tmp_path)

    with (
        patch.object(GradleStrategy, "find_executable", return_value="gradle"),
        patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(args=["gradle"], returncode=0, stdout="", stderr="")

        strategy._resolve_project_classpath(classpath, build_root, {"JAVA_HOME": "/jdk"}, timeout=120)

    # Should have been called once to compile module-c
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "gradle"
    assert ":module-c:classes" in cmd
    assert "--no-daemon" in cmd


def test_no_compilation_when_all_compiled(strategy, build_root, tmp_path):
    """When all modules have compiled classes, no compilation should be triggered."""
    # Give module-c some compiled classes
    (build_root / "module-c" / "build" / "classes" / "java" / "main").mkdir(parents=True)
    (build_root / "module-c" / "build" / "classes" / "java" / "main" / "C.class").write_bytes(b"")

    classpath = _make_classpath(build_root, tmp_path)

    with (
        patch.object(GradleStrategy, "find_executable", return_value="gradle"),
        patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout") as mock_run,
    ):
        strategy._resolve_project_classpath(classpath, build_root, {}, timeout=60)

    # No Gradle call should have been made
    mock_run.assert_not_called()


def test_noop_when_no_missing_jars(strategy, build_root, tmp_path):
    """When all JARs exist, the classpath should be returned unchanged."""
    # Create all the JAR files
    for mod in ["module-a", "module-b", "module-c"]:
        jar_dir = build_root / mod / "build" / "libs"
        jar_dir.mkdir(parents=True, exist_ok=True)
        (jar_dir / f"{mod}-1.0.jar").write_bytes(b"")

    classpath = _make_classpath(build_root, tmp_path)
    result = strategy._resolve_project_classpath(classpath, build_root, {}, timeout=60)
    assert result == classpath


def test_external_missing_jar_preserved(strategy, tmp_path):
    """Missing external JARs (not under build_root) should be kept as-is."""
    root = (tmp_path / "project").resolve()
    root.mkdir()

    external_jar = "/some/external/path/lib.jar"
    classpath = external_jar

    result = strategy._resolve_project_classpath(classpath, root, {}, timeout=60)
    assert result == external_jar


def test_nested_gradle_module(strategy, tmp_path):
    """Nested Gradle modules (connect/runtime) should be handled correctly."""
    root = (tmp_path / "project").resolve()
    root.mkdir()

    # Nested module: connect/runtime
    nested = root / "connect" / "runtime"
    (nested / "build" / "classes" / "java" / "main").mkdir(parents=True)
    (nested / "build" / "classes" / "java" / "main" / "R.class").write_bytes(b"")

    jar_path = str(root / "connect" / "runtime" / "build" / "libs" / "runtime-1.0.jar")
    classpath = jar_path

    result = strategy._resolve_project_classpath(classpath, root, {}, timeout=60)
    entries = result.split(os.pathsep)

    assert str(root / "connect" / "runtime" / "build" / "classes" / "java" / "main") in entries
    assert jar_path not in entries


def test_compile_dependency_modules_single_call(strategy, tmp_path):
    """Multiple uncompiled modules should be compiled in a single Gradle invocation."""
    root = (tmp_path / "project").resolve()
    root.mkdir()

    with (
        patch.object(GradleStrategy, "find_executable", return_value="gradle"),
        patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(args=["gradle"], returncode=0, stdout="", stderr="")

        strategy._compile_dependency_modules(root, {}, ["module-a", "module-b", "connect:runtime"], timeout=120)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert ":module-a:classes" in cmd
    assert ":module-b:classes" in cmd
    assert ":connect:runtime:classes" in cmd


def test_compile_dependency_modules_gradle_not_found(strategy, tmp_path):
    """Should not crash when Gradle executable is not found."""
    root = (tmp_path / "project").resolve()
    root.mkdir()

    with patch.object(GradleStrategy, "find_executable", return_value=None):
        # Should not raise
        strategy._compile_dependency_modules(root, {}, ["module-a"], timeout=60)
