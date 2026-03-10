"""Gradle build tool strategy for Java projects.

Implements BuildToolStrategy for Gradle-based projects, handling compilation,
classpath extraction, test execution, and JaCoCo coverage.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from codeflash.languages.java.build_tool_strategy import BuildToolStrategy

_BUILD = "build"

logger = logging.getLogger(__name__)

# Cache for classpath strings — keyed on (gradle_root, test_module).
_classpath_cache: dict[tuple[Path, str | None], str] = {}

# Cache for multi-module dependency installs — keyed on (gradle_root, test_module).
_multimodule_deps_installed: set[tuple[Path, str]] = set()

# Gradle init script that prints the test runtime classpath.
# Uses projectsEvaluated to avoid triggering configuration of unrelated subprojects.
_CLASSPATH_INIT_SCRIPT = """\
gradle.projectsEvaluated {
    allprojects {
        tasks.register("codeflashPrintClasspath") {
            doLast {
                def cp = configurations.findByName('testRuntimeClasspath')
                if (cp != null && cp.isCanBeResolved()) {
                    println "CODEFLASH_CP_START"
                    println cp.asPath
                    println "CODEFLASH_CP_END"
                }
            }
        }
    }
}
"""

# Gradle init script that applies JaCoCo plugin for coverage collection.
# Uses projectsEvaluated to avoid triggering configuration of unrelated subprojects.
_JACOCO_INIT_SCRIPT = """\
gradle.projectsEvaluated {
    allprojects {
        apply plugin: 'jacoco'
        jacocoTestReport {
            reports {
                xml.required = true
                html.required = false
            }
        }
    }
}
"""


def find_gradle_build_file(project_root: Path) -> Path | None:
    kts = project_root / "build.gradle.kts"
    if kts.exists():
        return kts
    groovy = project_root / "build.gradle"
    if groovy.exists():
        return groovy
    return None


def add_codeflash_dependency(build_file: Path, runtime_jar_path: Path) -> bool:
    if not build_file.exists():
        return False

    try:
        content = build_file.read_text(encoding="utf-8")

        if "codeflash-runtime" in content:
            logger.info("codeflash-runtime dependency already present in %s", build_file.name)
            return True

        is_kts = build_file.name.endswith(".kts")
        jar_str = str(runtime_jar_path).replace("\\", "/")

        if is_kts:
            dep_line = f'    testImplementation(files("{jar_str}"))  // codeflash-runtime\n'
        else:
            dep_line = f"    testImplementation files('{jar_str}')  // codeflash-runtime\n"

        # Try to insert inside existing dependencies block
        if "dependencies {" in content or "dependencies{" in content:
            last_deps_idx = content.rfind("dependencies")
            if last_deps_idx != -1:
                brace_depth = 0
                insert_pos = -1
                for i in range(last_deps_idx, len(content)):
                    if content[i] == "{":
                        brace_depth += 1
                    elif content[i] == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            insert_pos = i
                            break
                if insert_pos != -1:
                    content = content[:insert_pos] + dep_line + content[insert_pos:]
                    build_file.write_text(content, encoding="utf-8")
                    logger.info("Added codeflash-runtime dependency to %s", build_file.name)
                    return True

        # No existing dependencies block — append one
        if is_kts:
            content += f'\ndependencies {{\n    testImplementation(files("{jar_str}"))  // codeflash-runtime\n}}\n'
        else:
            content += f"\ndependencies {{\n    testImplementation files('{jar_str}')  // codeflash-runtime\n}}\n"
        build_file.write_text(content, encoding="utf-8")
        logger.info("Added codeflash-runtime dependency to %s", build_file.name)
        return True

    except Exception as e:
        logger.exception("Failed to add dependency to %s: %s", build_file.name, e)
        return False


class GradleStrategy(BuildToolStrategy):
    """Gradle-specific build tool operations."""

    @property
    def name(self) -> str:
        return "Gradle"

    def find_executable(self, build_root: Path) -> str | None:
        # Walk up from build_root to find gradlew — for multi-module projects
        # the wrapper lives at the repo root, which may be a parent of build_root.
        current = build_root.resolve()
        while True:
            gradlew_path = current / "gradlew"
            if gradlew_path.exists():
                return str(gradlew_path)
            gradlew_bat_path = current / "gradlew.bat"
            if gradlew_bat_path.exists():
                return str(gradlew_bat_path)
            parent = current.parent
            if parent == current:
                break
            current = parent
        # Fall back to system Gradle
        return shutil.which("gradle")

    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        runtime_jar = self.find_runtime_jar()
        if runtime_jar is None:
            logger.error("codeflash-runtime JAR not found. Generated tests will fail to compile.")
            return False

        if test_module:
            module_root = build_root / test_module
        else:
            module_root = build_root

        libs_dir = module_root / "libs"
        libs_dir.mkdir(parents=True, exist_ok=True)
        dest_jar = libs_dir / "codeflash-runtime-1.0.0.jar"

        if not dest_jar.exists():
            logger.info("Copying codeflash-runtime JAR to %s", dest_jar)
            shutil.copy2(runtime_jar, dest_jar)

        build_file = find_gradle_build_file(module_root)
        if build_file is None:
            logger.warning("No build.gradle(.kts) found at %s, cannot add codeflash-runtime dependency", module_root)
            return False

        if not add_codeflash_dependency(build_file, dest_jar):
            logger.error("Failed to add codeflash-runtime dependency to %s", build_file)
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

        gradle = self.find_executable(build_root)
        if not gradle:
            logger.error("Gradle not found — cannot pre-install multi-module dependencies")
            return False

        cmd = [gradle, f":{test_module}:classes", "-x", "test", "--build-cache", "--no-daemon"]

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
    ) -> subprocess.CompletedProcess[str]:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        gradle = self.find_executable(build_root)
        if not gradle:
            logger.error("Gradle not found")
            return subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found")

        if test_module:
            cmd = [gradle, f":{test_module}:testClasses", "--no-daemon"]
        else:
            cmd = [gradle, "testClasses", "--no-daemon"]

        logger.debug("Compiling tests: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Gradle compilation failed: %s", e)
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

        gradle = self.find_executable(build_root)
        if not gradle:
            return None

        # Write init script to a temp file
        init_script_fd, init_script_path = tempfile.mkstemp(suffix=".gradle", prefix="codeflash_cp_")
        try:
            with os.fdopen(init_script_fd, "w", encoding="utf-8") as f:
                f.write(_CLASSPATH_INIT_SCRIPT)

            if test_module:
                task = f":{test_module}:codeflashPrintClasspath"
            else:
                task = "codeflashPrintClasspath"

            cmd = [gradle, "--init-script", init_script_path, task, "-q", "--no-daemon"]

            logger.debug("Getting classpath: %s", " ".join(cmd))

            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
                logger.error("Failed to get classpath: %s", result.stderr)
                return None

            classpath = self._parse_classpath_output(result.stdout)
            if not classpath:
                logger.error("Classpath not found in Gradle output")
                return None

            if test_module:
                module_path = build_root / test_module
            else:
                module_path = build_root

            test_classes = module_path / "build" / "classes" / "java" / "test"
            main_classes = module_path / "build" / "classes" / "java" / "main"

            cp_parts = [classpath]
            if test_classes.exists():
                cp_parts.append(str(test_classes))
            if main_classes.exists():
                cp_parts.append(str(main_classes))

            if test_module:
                for module_dir in build_root.iterdir():
                    if module_dir.is_dir() and module_dir.name != test_module:
                        module_classes = module_dir / "build" / "classes" / "java" / "main"
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
            Path(init_script_path).unlink(missing_ok=True)

    @staticmethod
    def _parse_classpath_output(stdout: str) -> str | None:
        in_cp = False
        for line in stdout.splitlines():
            if line.strip() == "CODEFLASH_CP_START":
                in_cp = True
                continue
            if line.strip() == "CODEFLASH_CP_END":
                break
            if in_cp and line.strip():
                return line.strip()
        return None

    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        build_dir = self.get_build_output_dir(build_root, test_module)
        return build_dir / "test-results" / "test"

    def get_build_output_dir(self, build_root: Path, test_module: str | None) -> Path:
        if test_module:
            return build_root.joinpath(test_module, _BUILD)
        return build_root.joinpath(_BUILD)

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
    ) -> subprocess.CompletedProcess[str]:
        from codeflash.languages.java.test_runner import _build_test_filter, _run_cmd_kill_pg_on_timeout

        gradle = self.find_executable(build_root)
        if not gradle:
            logger.error("Gradle not found")
            return subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found")

        test_filter = _build_test_filter(test_paths, mode=mode)
        logger.debug("Built test filter for mode=%s: '%s' (empty=%s)", mode, test_filter, not test_filter)

        if test_module:
            task = f":{test_module}:test"
        else:
            task = "test"

        # Write an init script that configures JVM args for the test task.
        # -Dorg.gradle.jvmargs only affects the Gradle daemon, NOT the forked test JVM.
        add_opens = [
            "--add-opens",
            "java.base/java.util=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.lang=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.io=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.math=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.net=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.util.zip=ALL-UNNAMED",
        ]
        all_jvm_args = list(add_opens)
        if javaagent_arg:
            all_jvm_args.insert(0, javaagent_arg)

        per_test_timeout = max(timeout // 3, 10)
        quoted_args = ", ".join(f'"{a}"' for a in all_jvm_args)
        init_script_content = (
            f"gradle.projectsEvaluated {{\n"
            f"    allprojects {{\n"
            f"        tasks.withType(Test) {{\n"
            f"            jvmArgs({quoted_args})\n"
            f'            systemProperty "junit.jupiter.execution.timeout.default", "{per_test_timeout}s"\n'
            f"        }}\n"
            f"    }}\n"
            f"}}\n"
        )

        init_fd, init_path = tempfile.mkstemp(suffix=".gradle", prefix="codeflash_jvmargs_")
        with os.fdopen(init_fd, "w", encoding="utf-8") as f:
            f.write(init_script_content)

        cmd = [gradle, task, "--no-daemon", "--rerun", "--init-script", init_path]

        if test_filter:
            for class_filter in test_filter.split(","):
                class_filter = class_filter.strip()
                if class_filter:
                    cmd.extend(["--tests", class_filter])
            logger.debug("Added --tests filters to Gradle command")
        else:
            error_msg = (
                f"Test filter is EMPTY for mode={mode}! "
                f"Gradle will run ALL tests instead of the specified tests. "
                f"This indicates a problem with test file instrumentation or path resolution."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Append jacocoTestReport AFTER --tests so Gradle doesn't try to apply --tests to it
        if enable_coverage:
            cmd.append("jacocoTestReport")

        logger.debug("Running Gradle command: %s in %s", " ".join(cmd), build_root)

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
                compilation_error_indicators = [
                    "Compilation failed",
                    "COMPILATION ERROR",
                    "cannot find symbol",
                    "error: package",
                ]
                combined_output = (result.stdout or "") + (result.stderr or "")
                has_compilation_error = any(
                    indicator.lower() in combined_output.lower() for indicator in compilation_error_indicators
                )
                if has_compilation_error:
                    logger.error(
                        "Gradle compilation failed for %s tests. "
                        "Check that generated test code is syntactically valid Java. "
                        "Return code: %s",
                        mode,
                        result.returncode,
                    )
                    output_lines = combined_output.split("\n")
                    error_context = "\n".join(output_lines[:50]) if len(output_lines) > 50 else combined_output
                    logger.error("Gradle compilation error output:\n%s", error_context)

            return result

        except Exception as e:
            logger.exception("Gradle test execution failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))
        finally:
            Path(init_path).unlink(missing_ok=True)

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
        import time

        from codeflash.languages.java.test_runner import _find_multi_module_root, _get_combined_junit_xml

        project_root = project_root or cwd
        gradle_root, test_module = _find_multi_module_root(project_root, test_paths)

        all_stdout: list[str] = []
        all_stderr: list[str] = []
        total_start_time = time.time()
        loop_count = 0
        last_result = None

        per_loop_timeout = max(timeout or 0, 120, 60 + inner_iterations)

        logger.debug("Using Gradle-based benchmarking (fallback mode)")

        for loop_idx in range(1, max_loops + 1):
            run_env = os.environ.copy()
            run_env.update(test_env)
            run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
            run_env["CODEFLASH_MODE"] = "performance"
            run_env["CODEFLASH_TEST_ITERATION"] = "0"
            if "CODEFLASH_INNER_ITERATIONS" not in run_env:
                run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

            result = self.run_tests_via_build_tool(
                gradle_root, test_paths, run_env, timeout=per_loop_timeout, mode="performance", test_module=test_module
            )

            last_result = result
            loop_count = loop_idx

            if result.stdout:
                all_stdout.append(result.stdout)
            if result.stderr:
                all_stderr.append(result.stderr)

            elapsed = time.time() - total_start_time
            if loop_idx >= min_loops and elapsed >= target_duration_seconds:
                logger.debug("Stopping Gradle benchmark after %d loops (%.2fs elapsed)", loop_idx, elapsed)
                break

            if result.returncode != 0:
                timing_pattern = re.compile(r"!######[^:]*:[^:]*:[^:]*:[^:]*:[^:]+:[^:]+######!")
                has_timing_markers = bool(timing_pattern.search(result.stdout or ""))
                if not has_timing_markers:
                    logger.warning("Tests failed in Gradle loop %d with no timing markers, stopping", loop_idx)
                    break
                logger.debug("Some tests failed in Gradle loop %d but timing markers present, continuing", loop_idx)

        combined_stdout = "\n".join(all_stdout)
        combined_stderr = "\n".join(all_stderr)

        total_iterations = loop_count * inner_iterations
        logger.debug(
            "Gradle fallback: %d loops x %d iterations = %d total in %.2fs",
            loop_count,
            inner_iterations,
            total_iterations,
            time.time() - total_start_time,
        )

        combined_result = subprocess.CompletedProcess(
            args=last_result.args if last_result else ["gradle", "test"],
            returncode=last_result.returncode if last_result else -1,
            stdout=combined_stdout,
            stderr=combined_stderr,
        )

        reports_dir = self.get_reports_dir(gradle_root, test_module)
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
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path | None]:
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
            module_root = build_root / test_module
        else:
            module_root = project_root

        build_file = find_gradle_build_file(module_root)
        if build_file is None:
            logger.warning("No build.gradle(.kts) found at %s, cannot setup JaCoCo", module_root)
            return None

        content = build_file.read_text(encoding="utf-8")
        if "jacoco" not in content.lower():
            logger.info("Adding JaCoCo plugin to %s for coverage collection", build_file.name)
            is_kts = build_file.name.endswith(".kts")
            if is_kts:
                plugin_line = "plugins {\n    jacoco\n}\n"
            else:
                plugin_line = "apply plugin: 'jacoco'\n"

            if "plugins {" in content or "plugins{" in content:
                # Insert jacoco inside existing plugins block
                plugins_idx = content.find("plugins")
                brace_depth = 0
                for i in range(plugins_idx, len(content)):
                    if content[i] == "{":
                        brace_depth += 1
                    elif content[i] == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            insert = "    jacoco\n" if is_kts else "    id 'jacoco'\n"
                            content = content[:i] + insert + content[i:]
                            break
            else:
                content = plugin_line + content

            build_file.write_text(content, encoding="utf-8")

        return module_root / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"

    def get_test_run_command(self, project_root: Path, test_classes: list[str] | None = None) -> list[str]:
        from codeflash.languages.java.test_runner import _validate_java_class_name

        if test_classes:
            for test_class in test_classes:
                if not _validate_java_class_name(test_class):
                    msg = f"Invalid test class name: '{test_class}'. Test names must follow Java identifier rules."
                    raise ValueError(msg)

        gradle = self.find_executable(project_root) or "gradle"
        cmd = [gradle, "test", "--no-daemon"]
        if test_classes:
            for cls in test_classes:
                cmd.extend(["--tests", cls])
        return cmd
