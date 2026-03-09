"""Maven build tool strategy for Java projects.

Implements BuildToolStrategy for Maven-based projects, handling compilation,
classpath extraction, test execution via Surefire, and JaCoCo coverage.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from codeflash.languages.java.build_tool_strategy import BuildToolStrategy
from codeflash.languages.java.build_tools import (
    add_codeflash_dependency_to_pom,
    add_jacoco_plugin_to_pom,
    find_maven_executable,
    get_jacoco_xml_path,
    install_codeflash_runtime,
    is_jacoco_configured,
)

logger = logging.getLogger(__name__)

# Skip validation/analysis plugins that reject generated instrumented files
_MAVEN_VALIDATION_SKIP_FLAGS = [
    "-Drat.skip=true",
    "-Dcheckstyle.skip=true",
    "-Dspotbugs.skip=true",
    "-Dpmd.skip=true",
    "-Denforcer.skip=true",
    "-Djapicmp.skip=true",
]

# Cache for classpath strings — keyed on (maven_root, test_module).
_classpath_cache: dict[tuple[Path, str | None], str] = {}

# Cache for multi-module dependency installs — keyed on (maven_root, test_module).
_multimodule_deps_installed: set[tuple[Path, str]] = set()


class MavenStrategy(BuildToolStrategy):
    """Maven-specific build tool operations."""

    @property
    def name(self) -> str:
        return "Maven"

    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        from codeflash.languages.java.test_runner import _find_runtime_jar

        runtime_jar = _find_runtime_jar()
        if runtime_jar is None:
            logger.error("codeflash-runtime JAR not found. Generated tests will fail to compile.")
            return False

        m2_jar = (
            Path.home()
            / ".m2"
            / "repository"
            / "com"
            / "codeflash"
            / "codeflash-runtime"
            / "1.0.0"
            / "codeflash-runtime-1.0.0.jar"
        )
        if not m2_jar.exists():
            logger.info("Installing codeflash-runtime JAR to local Maven repository")
            if not install_codeflash_runtime(build_root, runtime_jar):
                logger.error("Failed to install codeflash-runtime to local Maven repository")
                return False

        if test_module:
            pom_path = build_root / test_module / "pom.xml"
        else:
            pom_path = build_root / "pom.xml"

        if pom_path.exists():
            if not add_codeflash_dependency_to_pom(pom_path):
                logger.error("Failed to add codeflash-runtime dependency to %s", pom_path)
                return False
        else:
            logger.warning("pom.xml not found at %s, cannot add codeflash-runtime dependency", pom_path)
            return False

        return True

    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> bool:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        if not test_module:
            return True

        cache_key = (build_root, test_module)
        if cache_key in _multimodule_deps_installed:
            logger.debug("Multi-module deps already installed for %s:%s, skipping", build_root, test_module)
            return True

        mvn = find_maven_executable()
        if not mvn:
            logger.error("Maven not found — cannot pre-install multi-module dependencies")
            return False

        cmd = [mvn, "install", "-DskipTests", "-B", "-pl", test_module, "-am"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        logger.info("Pre-installing multi-module dependencies: %s (module: %s)", build_root, test_module)
        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=300)
            if result.returncode != 0:
                logger.error(
                    "Failed to pre-install multi-module deps (exit %d).\nstdout: %s\nstderr: %s",
                    result.returncode,
                    result.stdout[-2000:] if result.stdout else "",
                    result.stderr[-2000:] if result.stderr else "",
                )
                return False
        except Exception:
            logger.exception("Exception during multi-module dependency install")
            return False

        _multimodule_deps_installed.add(cache_key)
        logger.info("Multi-module dependencies installed successfully for %s:%s", build_root, test_module)
        return True

    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        mvn = find_maven_executable()
        if not mvn:
            logger.error("Maven not found")
            return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

        cmd = [mvn, "test-compile", "-e", "-B"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        if test_module:
            cmd.extend(["-pl", test_module])

        logger.debug("Compiling tests: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Maven compilation failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def get_classpath(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        key = (build_root, test_module)
        cached = _classpath_cache.get(key)
        if cached is not None:
            logger.debug("Using cached classpath for (%s, %s)", build_root, test_module)
            return cached
        result = self._get_classpath_uncached(build_root, env, test_module, timeout)
        if result is not None:
            _classpath_cache[key] = result
        return result

    def _get_classpath_uncached(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        from codeflash.languages.java.test_runner import _find_junit_console_standalone, _run_cmd_kill_pg_on_timeout

        mvn = find_maven_executable()
        if not mvn:
            return None

        cp_file = build_root / ".codeflash_classpath.txt"

        cmd = [mvn, "dependency:build-classpath", "-DincludeScope=test", f"-Dmdep.outputFile={cp_file}", "-q", "-B"]

        if test_module:
            cmd.extend(["-pl", test_module])

        logger.debug("Getting classpath: %s", " ".join(cmd))

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
                logger.error("Failed to get classpath: %s", result.stderr)
                return None

            if not cp_file.exists():
                logger.error("Classpath file not created")
                return None

            classpath = cp_file.read_text(encoding="utf-8").strip()

            if test_module:
                module_path = build_root / test_module
            else:
                module_path = build_root

            test_classes = module_path / "target" / "test-classes"
            main_classes = module_path / "target" / "classes"

            cp_parts = [classpath]
            if test_classes.exists():
                cp_parts.append(str(test_classes))
            if main_classes.exists():
                cp_parts.append(str(main_classes))

            if test_module:
                for module_dir in build_root.iterdir():
                    if module_dir.is_dir() and module_dir.name != test_module:
                        module_classes = module_dir / "target" / "classes"
                        if module_classes.exists():
                            logger.debug("Adding multi-module classpath: %s", module_classes)
                            cp_parts.append(str(module_classes))

            if "console-standalone" not in classpath and "ConsoleLauncher" not in classpath:
                console_jar = _find_junit_console_standalone()
                if console_jar:
                    logger.debug("Adding JUnit Console Standalone to classpath: %s", console_jar)
                    cp_parts.append(str(console_jar))

            return os.pathsep.join(cp_parts)

        except Exception as e:
            logger.exception("Failed to get classpath: %s", e)
            return None
        finally:
            if cp_file.exists():
                cp_file.unlink()

    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        target_dir = self.get_build_output_dir(build_root, test_module)
        return target_dir / "surefire-reports"

    def get_build_output_dir(self, build_root: Path, test_module: str | None) -> Path:
        if test_module:
            return build_root / test_module / "target"
        return build_root / "target"

    def run_tests_via_build_tool(
        self,
        build_root: Path,
        test_paths: Any,
        env: dict[str, str],
        timeout: int,
        mode: str,
        test_module: str | None,
        javaagent_arg: str | None = None,
        enable_coverage: bool = False,
    ) -> subprocess.CompletedProcess:
        from codeflash.languages.java.test_runner import (
            _build_test_filter,
            _run_cmd_kill_pg_on_timeout,
            _validate_test_filter,
        )

        mvn = find_maven_executable()
        if not mvn:
            logger.error("Maven not found")
            return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

        test_filter = _build_test_filter(test_paths, mode=mode)
        logger.debug("Built test filter for mode=%s: '%s' (empty=%s)", mode, test_filter, not test_filter)

        maven_goal = "verify" if enable_coverage else "test"
        cmd = [mvn, maven_goal, "-fae", "-B"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        add_opens_flags = (
            "--add-opens java.base/java.util=ALL-UNNAMED"
            " --add-opens java.base/java.lang=ALL-UNNAMED"
            " --add-opens java.base/java.lang.reflect=ALL-UNNAMED"
            " --add-opens java.base/java.io=ALL-UNNAMED"
            " --add-opens java.base/java.math=ALL-UNNAMED"
            " --add-opens java.base/java.net=ALL-UNNAMED"
            " --add-opens java.base/java.util.zip=ALL-UNNAMED"
        )
        if javaagent_arg:
            cmd.append(f"-DargLine={javaagent_arg} {add_opens_flags}")
        else:
            cmd.append(f"-DargLine={add_opens_flags}")

        if mode == "performance":
            cmd.append("-Dsurefire.useFile=false")

        if enable_coverage:
            cmd.append("-Dmaven.test.failure.ignore=true")

        if test_module:
            cmd.extend(
                [
                    "-pl",
                    test_module,
                    "-DfailIfNoTests=false",
                    "-Dsurefire.failIfNoSpecifiedTests=false",
                    "-DskipTests=false",
                ]
            )

        if test_filter:
            validated_filter = _validate_test_filter(test_filter)
            cmd.append(f"-Dtest={validated_filter}")
            logger.debug("Added -Dtest=%s to Maven command", validated_filter)
        else:
            error_msg = (
                f"Test filter is EMPTY for mode={mode}! "
                f"Maven will run ALL tests instead of the specified tests. "
                f"This indicates a problem with test file instrumentation or path resolution."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug("Running Maven command: %s in %s", " ".join(cmd), build_root)

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
                compilation_error_indicators = [
                    "[ERROR] COMPILATION ERROR",
                    "[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin",
                    "compilation failure",
                    "cannot find symbol",
                    "package .* does not exist",
                ]
                combined_output = (result.stdout or "") + (result.stderr or "")
                has_compilation_error = any(
                    indicator.lower() in combined_output.lower() for indicator in compilation_error_indicators
                )
                if has_compilation_error:
                    logger.error(
                        "Maven compilation failed for %s tests. "
                        "Check that generated test code is syntactically valid Java. "
                        "Return code: %s",
                        mode,
                        result.returncode,
                    )
                    output_lines = combined_output.split("\n")
                    error_context = "\n".join(output_lines[:50]) if len(output_lines) > 50 else combined_output
                    logger.error("Maven compilation error output:\n%s", error_context)

            return result

        except Exception as e:
            logger.exception("Maven test execution failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def run_benchmarking_via_build_tool(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None,
        project_root: Path | None,
        min_loops: int,
        max_loops: int,
        target_duration_seconds: float,
        inner_iterations: int,
    ) -> tuple[Path, Any]:
        import re
        import time

        from codeflash.languages.java.test_runner import _find_multi_module_root, _get_combined_junit_xml

        project_root = project_root or cwd
        maven_root, test_module = _find_multi_module_root(project_root, test_paths)

        all_stdout: list[str] = []
        all_stderr: list[str] = []
        total_start_time = time.time()
        loop_count = 0
        last_result = None

        per_loop_timeout = max(timeout or 0, 120, 60 + inner_iterations)

        logger.debug("Using Maven-based benchmarking (fallback mode)")

        for loop_idx in range(1, max_loops + 1):
            run_env = os.environ.copy()
            run_env.update(test_env)
            run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
            run_env["CODEFLASH_MODE"] = "performance"
            run_env["CODEFLASH_TEST_ITERATION"] = "0"
            if "CODEFLASH_INNER_ITERATIONS" not in run_env:
                run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

            result = self.run_tests_via_build_tool(
                maven_root, test_paths, run_env, timeout=per_loop_timeout, mode="performance", test_module=test_module
            )

            last_result = result
            loop_count = loop_idx

            if result.stdout:
                all_stdout.append(result.stdout)
            if result.stderr:
                all_stderr.append(result.stderr)

            elapsed = time.time() - total_start_time
            if loop_idx >= min_loops and elapsed >= target_duration_seconds:
                logger.debug("Stopping Maven benchmark after %d loops (%.2fs elapsed)", loop_idx, elapsed)
                break

            if result.returncode != 0:
                timing_pattern = re.compile(r"!######[^:]*:[^:]*:[^:]*:[^:]*:[^:]+:[^:]+######!")
                has_timing_markers = bool(timing_pattern.search(result.stdout or ""))
                if not has_timing_markers:
                    logger.warning("Tests failed in Maven loop %d with no timing markers, stopping", loop_idx)
                    break
                logger.debug("Some tests failed in Maven loop %d but timing markers present, continuing", loop_idx)

        combined_stdout = "\n".join(all_stdout)
        combined_stderr = "\n".join(all_stderr)

        total_iterations = loop_count * inner_iterations
        logger.debug(
            "Maven fallback: %d loops x %d iterations = %d total in %.2fs",
            loop_count,
            inner_iterations,
            total_iterations,
            time.time() - total_start_time,
        )

        combined_result = subprocess.CompletedProcess(
            args=last_result.args if last_result else ["mvn", "test"],
            returncode=last_result.returncode if last_result else -1,
            stdout=combined_stdout,
            stderr=combined_stderr,
        )

        reports_dir = self.get_reports_dir(maven_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, -1)

        return result_xml_path, combined_result

    def run_tests_with_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess, Path, Path | None]:
        from codeflash.languages.java.test_runner import _get_combined_junit_xml

        coverage_xml_path = self.setup_coverage(build_root, test_module, build_root)

        result = self.run_tests_via_build_tool(
            build_root,
            test_paths,
            run_env,
            timeout=timeout,
            mode="behavior",
            enable_coverage=True,
            test_module=test_module,
        )

        reports_dir = self.get_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)

        return result, result_xml_path, coverage_xml_path

    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        if test_module:
            test_module_pom = build_root / test_module / "pom.xml"
            if test_module_pom.exists():
                if not is_jacoco_configured(test_module_pom):
                    logger.info("Adding JaCoCo plugin to test module pom.xml: %s", test_module_pom)
                    add_jacoco_plugin_to_pom(test_module_pom)
                return get_jacoco_xml_path(build_root / test_module)
        else:
            pom_path = project_root / "pom.xml"
            if pom_path.exists():
                if not is_jacoco_configured(pom_path):
                    logger.info("Adding JaCoCo plugin to pom.xml for coverage collection")
                    add_jacoco_plugin_to_pom(pom_path)
                return get_jacoco_xml_path(project_root)
        return None
