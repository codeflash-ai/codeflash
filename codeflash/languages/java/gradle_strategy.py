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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from codeflash.languages.java.build_tool_strategy import BuildToolStrategy, module_to_dir
from codeflash.languages.java.build_tools import CODEFLASH_RUNTIME_JAR_NAME, BuildTool, JavaProjectInfo

_RE_INCLUDE = re.compile(r"""include\s*\(?([^)\n]+)\)?""")

_RE_QUOTED = re.compile(r"""['"]([^'"]+)['"]""")

_BUILD = "build"

logger = logging.getLogger(__name__)

# Groovy init script that disables validation/analysis plugins.
# Equivalent to Maven's -Dcheckstyle.skip=true, -Dspotbugs.skip=true, etc.
# Using an init script is safe even if the plugins aren't applied — unlike
# `-x taskName` which fails if the task doesn't exist.
_GRADLE_SKIP_VALIDATION_INIT_SCRIPT = """\
gradle.projectsEvaluated {
    allprojects {
        // Disable checkstyle, spotbugs, pmd by type (catches all source sets, not just Main/Test)
        try { tasks.withType(Checkstyle) { enabled = false } } catch (e) {}
        try { tasks.withType(Class.forName('com.github.spotbugs.snom.SpotBugsTask')) { enabled = false } } catch (e) {}
        try { tasks.withType(Pmd) { enabled = false } } catch (e) {}
        // Disable remaining validation tasks by name
        tasks.matching { task ->
            task.name in [
                'checkstyleMain', 'checkstyleTest',
                'spotbugsMain', 'spotbugsTest',
                'pmdMain', 'pmdTest',
                'rat', 'japicmp',
                'jarHell', 'thirdPartyAudit'
            ]
        }.configureEach {
            enabled = false
        }
        tasks.withType(JavaCompile) {
            options.compilerArgs.removeAll { it == '-Werror' }
            options.compilerArgs.removeAll { it == '-Xlint:all' }
            if (options.hasProperty('errorprone')) {
                options.errorprone {
                    enabled = false
                }
            }
        }
    }
}
"""

# Lazily-created temp file for the validation-skip init script.
_skip_validation_init_path: str | None = None


def _get_skip_validation_init_script() -> str:
    """Return the path to a persistent temp init script that disables validation tasks."""
    global _skip_validation_init_path
    if _skip_validation_init_path is None or not Path(_skip_validation_init_path).exists():
        fd, path = tempfile.mkstemp(suffix=".gradle", prefix="codeflash_skip_validation_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_GRADLE_SKIP_VALIDATION_INIT_SCRIPT)
        _skip_validation_init_path = path
    return _skip_validation_init_path


# Lazily-created temp file for the JaCoCo init script.
_jacoco_init_path: str | None = None


def _get_jacoco_init_script() -> str:
    global _jacoco_init_path
    if _jacoco_init_path is None or not Path(_jacoco_init_path).exists():
        fd, path = tempfile.mkstemp(suffix=".gradle", prefix="codeflash_jacoco_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_JACOCO_INIT_SCRIPT)
        _jacoco_init_path = path
    return _jacoco_init_path


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
# Uses projectsEvaluated + JavaPlugin guard so it only applies to subprojects
# that actually compile Java (skips container/root projects without java plugin).
_JACOCO_INIT_SCRIPT = """\
gradle.projectsEvaluated {
    allprojects { project ->
        project.plugins.withType(JavaPlugin) {
            project.apply plugin: 'jacoco'
            project.jacocoTestReport {
                reports {
                    xml.required = true
                    html.required = false
                }
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


def _find_top_level_dependencies_block(build_file: Path, content: str) -> int | None:
    """Find the insert position (before closing }) of the top-level dependencies block using tree-sitter.

    Returns the byte offset of the closing brace, or None if no top-level dependencies block exists.
    Only matches `dependencies { }` at the root level — ignores blocks nested inside
    `buildscript`, `subprojects`, `allprojects`, etc.
    """
    import tree_sitter as ts

    is_kts = build_file.name.endswith(".kts")
    source_bytes = content.encode("utf-8")

    if is_kts:
        import tree_sitter_kotlin as tsk

        parser = ts.Parser(ts.Language(tsk.language()))
    else:
        import tree_sitter_groovy as tsg

        parser = ts.Parser(ts.Language(tsg.language()))

    tree = parser.parse(source_bytes)

    # Walk only direct children of root to find top-level `dependencies { }`
    for child in tree.root_node.children:
        # Groovy: expression_statement > method_invocation(identifier="dependencies", closure)
        # Kotlin: call_expression(identifier="dependencies", annotated_lambda)
        node = child
        if node.type == "expression_statement" and node.child_count > 0:
            node = node.children[0]

        if node.type not in ("method_invocation", "call_expression"):
            continue

        name_node = None
        body_node = None
        for c in node.children:
            if c.type == "identifier":
                name_node = c
            elif c.type in ("closure", "annotated_lambda", "lambda_literal"):
                body_node = c

        if name_node is None or body_node is None:
            continue

        name = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
        if name != "dependencies":
            continue

        # Find the closing brace of this block
        closing_brace = body_node.children[-1] if body_node.children else None
        # For Kotlin, annotated_lambda wraps lambda_literal
        if closing_brace is not None and closing_brace.type == "lambda_literal":
            closing_brace = closing_brace.children[-1] if closing_brace.children else None

        if closing_brace is not None and closing_brace.type == "}":
            return closing_brace.start_byte

    return None


def _is_multimodule_project(build_root: Path) -> bool:
    """Check if this is a multi-module Gradle project by looking for include directives in settings files."""
    for settings_name in ("settings.gradle", "settings.gradle.kts"):
        settings_file = build_root / settings_name
        if settings_file.exists():
            try:
                content = settings_file.read_text(encoding="utf-8")
                if re.search(r'include\s*[\(\'"]', content):
                    return True
            except Exception:
                pass
    return False


def add_codeflash_dependency_multimodule(build_file: Path, runtime_jar_path: Path) -> bool:
    """Add codeflash-runtime dependency wrapped in a subprojects block for multi-module projects.

    This avoids adding testImplementation to the root build file directly, which would fail
    if the root project doesn't apply the java plugin.
    """
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
            block = (
                f"\nsubprojects {{\n"
                f'    plugins.withId("java") {{\n'
                f"        dependencies {{\n"
                f'            testImplementation(files("{jar_str}"))  // codeflash-runtime\n'
                f"        }}\n"
                f"    }}\n"
                f"}}\n"
            )
        else:
            block = (
                f"\nsubprojects {{\n"
                f"    plugins.withId('java') {{\n"
                f"        dependencies {{\n"
                f"            testImplementation files('{jar_str}')  // codeflash-runtime\n"
                f"        }}\n"
                f"    }}\n"
                f"}}\n"
            )

        content += block
        build_file.write_text(content, encoding="utf-8")
        logger.info("Added codeflash-runtime dependency to %s (subprojects block)", build_file.name)
        return True

    except Exception as e:
        logger.exception("Failed to add dependency to %s: %s", build_file.name, e)
        return False


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

        # Use tree-sitter to find the top-level dependencies block
        insert_pos = _find_top_level_dependencies_block(build_file, content)
        if insert_pos is not None:
            content = content[:insert_pos] + dep_line + content[insert_pos:]
            build_file.write_text(content, encoding="utf-8")
            logger.info("Added codeflash-runtime dependency to %s (tree-sitter)", build_file.name)
            return True

        # No existing dependencies block — append one
        if is_kts:
            content += f'\ndependencies {{\n    testImplementation(files("{jar_str}"))  // codeflash-runtime\n}}\n'
        else:
            content += f"\ndependencies {{\n    testImplementation files('{jar_str}')  // codeflash-runtime\n}}\n"
        build_file.write_text(content, encoding="utf-8")
        logger.info("Added codeflash-runtime dependency to %s (new block)", build_file.name)
        return True

    except Exception as e:
        logger.exception("Failed to add dependency to %s: %s", build_file.name, e)
        return False


def _normalize_gradle_xml_reports(reports_dir: Path) -> None:
    """Normalize Gradle JUnit XML reports to match Maven Surefire format.

    Gradle's JUnit Platform XML differs from Maven Surefire in ways that
    can crash the downstream parser:
    1. <failure>/<error> elements may omit the ``message`` attribute —
       Maven always sets it.
    2. Timeout information may only appear in the element body text,
       not in the ``message`` attribute.

    This function rewrites the XML files in-place so they conform to the
    Maven Surefire contract the parser expects.
    """
    if not reports_dir.exists():
        return
    for xml_file in reports_dir.glob("TEST-*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            modified = False
            for tag in ("failure", "error"):
                for elem in root.iter(tag):
                    if elem.get("message") is None:
                        body = (elem.text or "").strip()
                        first_line = body.split("\n", 1)[0] if body else ""
                        elem.set("message", first_line)
                        modified = True
            if modified:
                tree.write(xml_file, encoding="unicode", xml_declaration=True)
        except ET.ParseError:
            logger.debug("Failed to normalize Gradle XML report %s", xml_file)


def _extract_gradle_include_modules(content: str) -> list[str]:
    """Extract module names from include() directives in settings.gradle."""
    modules: list[str] = []
    for match in _RE_INCLUDE.finditer(content):
        args = match.group(1)
        for quoted in _RE_QUOTED.findall(args):
            module = quoted.lstrip(":")
            if module:
                modules.append(module)
    return modules


def _parse_gradle_settings_modules(project_root: Path) -> list[str]:
    """Parse settings.gradle(.kts) to find included modules."""
    for settings_name in ["settings.gradle", "settings.gradle.kts"]:
        settings_path = project_root / settings_name
        if settings_path.exists():
            try:
                content = settings_path.read_text(encoding="utf-8")
                return _extract_gradle_include_modules(content)
            except Exception:
                continue
    return []


def _discover_gradle_submodule_roots(project_root: Path) -> tuple[list[Path], list[Path]]:
    """Discover source and test roots from Gradle submodules."""
    source_roots: list[Path] = []
    test_roots: list[Path] = []

    modules = _parse_gradle_settings_modules(project_root)
    for module_name in modules:
        module_path = module_name.replace(":", "/")
        module_dir = project_root / module_path
        if not module_dir.is_dir():
            continue

        std_src = module_dir / "src" / "main" / "java"
        if std_src.exists():
            source_roots.append(std_src)

        std_test = module_dir / "src" / "test" / "java"
        if std_test.exists():
            test_roots.append(std_test)

    return source_roots, test_roots


class GradleStrategy(BuildToolStrategy):
    """Gradle-specific build tool operations."""

    @property
    def name(self) -> str:
        return "Gradle"

    def get_project_info(self, project_root: Path) -> JavaProjectInfo | None:
        source_roots: list[Path] = []
        test_roots: list[Path] = []

        main_src = project_root / "src" / "main" / "java"
        if main_src.exists():
            source_roots.append(main_src)

        test_src = project_root / "src" / "test" / "java"
        if test_src.exists():
            test_roots.append(test_src)

        sub_sources, sub_tests = _discover_gradle_submodule_roots(project_root)
        for root_path in sub_sources:
            if root_path not in source_roots:
                source_roots.append(root_path)
        for root_path in sub_tests:
            if root_path not in test_roots:
                test_roots.append(root_path)

        return JavaProjectInfo(
            project_root=project_root,
            build_tool=BuildTool.GRADLE,
            source_roots=source_roots,
            test_roots=test_roots,
            target_dir=project_root / "build",
            group_id=None,
            artifact_id=None,
            version=None,
            java_version=None,
        )

    def find_executable(self, build_root: Path) -> str | None:
        return self.find_wrapper_executable(build_root, ("gradlew", "gradlew.bat"), "gradle")

    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        runtime_jar = self.find_runtime_jar()
        if runtime_jar is None:
            from codeflash.languages.java.maven_strategy import download_from_maven_central_http

            runtime_jar = download_from_maven_central_http()
        if runtime_jar is None:
            logger.error("codeflash-runtime JAR not found. Generated tests will fail to compile.")
            return False

        if test_module:
            module_root = build_root / module_to_dir(test_module)
        else:
            module_root = build_root

        libs_dir = module_root / "libs"
        libs_dir.mkdir(parents=True, exist_ok=True)
        dest_jar = libs_dir / CODEFLASH_RUNTIME_JAR_NAME

        if not dest_jar.exists():
            logger.info("Copying codeflash-runtime JAR to %s", dest_jar)
            shutil.copy2(runtime_jar, dest_jar)

        build_file = find_gradle_build_file(module_root)
        if build_file is None:
            logger.warning("No build.gradle(.kts) found at %s, cannot add codeflash-runtime dependency", module_root)
            return False

        if not test_module and _is_multimodule_project(build_root):
            if not add_codeflash_dependency_multimodule(build_file, dest_jar):
                logger.error("Failed to add codeflash-runtime dependency to %s", build_file)
                return False
        elif not add_codeflash_dependency(build_file, dest_jar):
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
        cmd.extend(["--init-script", _get_skip_validation_init_script()])

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
        cmd.extend(["--init-script", _get_skip_validation_init_script()])

        logger.debug("Compiling tests: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Gradle compilation failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def compile_source_only(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        gradle = self.find_executable(build_root)
        if not gradle:
            logger.error("Gradle not found")
            return subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found")

        if test_module:
            cmd = [gradle, f":{test_module}:classes", "--no-daemon"]
        else:
            cmd = [gradle, "classes", "--no-daemon"]
        cmd.extend(["--init-script", _get_skip_validation_init_script()])

        logger.debug("Compiling source only: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Gradle source compilation failed: %s", e)
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
                module_path = build_root / module_to_dir(test_module)
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
                module_dir_name = module_to_dir(test_module)
                for module_dir in build_root.iterdir():
                    if module_dir.is_dir() and module_dir.name != module_dir_name:
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
            return build_root.joinpath(module_to_dir(test_module), _BUILD)
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
            f"            reports.junitXml.outputPerTestCase = true\n"
            f"            filter.failOnNoMatchingTests = false\n"
            f"        }}\n"
            f"    }}\n"
            f"}}\n"
        )

        if not test_filter:
            error_msg = (
                f"Test filter is EMPTY for mode={mode}! "
                f"Gradle will run ALL tests instead of the specified tests. "
                f"This indicates a problem with test file instrumentation or path resolution."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        init_fd, init_path = tempfile.mkstemp(suffix=".gradle", prefix="codeflash_jvmargs_")
        try:
            with os.fdopen(init_fd, "w", encoding="utf-8") as f:
                f.write(init_script_content)

            cmd = [gradle, task, "--no-daemon", "--rerun", "--init-script", init_path]
            cmd.extend(["--init-script", _get_skip_validation_init_script()])

            if enable_coverage:
                jacoco_init = _get_jacoco_init_script()
                cmd.extend(["--init-script", jacoco_init])
                # --continue ensures Gradle keeps going even if some tests fail,
                # so jacocoTestReport runs even after test failures
                # (matches Maven's -Dmaven.test.failure.ignore=true).
                cmd.append("--continue")

            # Note: multi-module --tests filtering is handled by
            #   filter.failOnNoMatchingTests = false in the init script above
            #   (matches Maven's -DfailIfNoTests=false).
            for class_filter in test_filter.split(","):
                class_filter = class_filter.strip()
                if class_filter:
                    cmd.extend(["--tests", class_filter])
            logger.debug("Added --tests filters to Gradle command")

            # Append jacocoTestReport AFTER --tests so Gradle doesn't try to apply --tests to it.
            # Must be module-qualified for multi-module projects — running it at root level fails
            # if the root project doesn't have the java plugin (e.g., eureka).
            if enable_coverage:
                jacoco_task = f":{test_module}:jacocoTestReport" if test_module else "jacocoTestReport"
                cmd.append(jacoco_task)

            logger.debug("Running Gradle command: %s in %s", " ".join(cmd), build_root)

            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            # Normalize XML reports so <failure>/<error> always have a message
            # attribute — Maven Surefire always sets it, Gradle may omit it.
            reports_dir = self.get_reports_dir(build_root, test_module)
            _normalize_gradle_xml_reports(reports_dir)

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
        # JaCoCo plugin is applied via init script (_JACOCO_INIT_SCRIPT) at test execution time,
        # so no build file modification is needed here. Just return the expected report path.
        if test_module:
            module_root = build_root / module_to_dir(test_module)
        else:
            module_root = project_root
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
