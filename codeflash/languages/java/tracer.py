from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from codeflash.languages.java.build_tools import find_maven_executable, find_source_root
from codeflash.languages.java.test_runner import _find_runtime_jar

logger = logging.getLogger(__name__)


class JavaTracer:
    """Traces Java test execution using a Java agent to collect profiling data.

    The agent instruments user code with ASM bytecode manipulation to record:
    - Per-method call counts and timing (total/cumulative)
    - Caller-callee call graphs
    - Serialized method arguments (via Kryo) for replay test generation

    Output is written to a SQLite file with the same schema as the Python tracer,
    so ProfileStats and FunctionRanker work unchanged.
    """

    def __init__(
        self,
        project_root: Path,
        module_root: Path,
        trace_output_path: Path | None = None,
        max_function_count: int = 256,
        timeout: int = 300,
    ) -> None:
        self.project_root = project_root
        self.module_root = module_root
        self.max_function_count = max_function_count
        self.timeout = timeout

        if trace_output_path is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".sqlite", prefix="codeflash_trace_")
            os.close(fd)
            self.trace_output_path = Path(tmp_path)
        else:
            self.trace_output_path = trace_output_path

    def run(self, test_command: list[str] | None = None) -> Path | None:
        """Run Maven tests with the Java agent attached for profiling.

        Returns the path to the SQLite trace file, or None on failure.
        """
        agent_jar = _find_runtime_jar()
        if agent_jar is None:
            logger.error("codeflash-runtime JAR not found. Cannot run Java tracer.")
            return None

        mvn = find_maven_executable(self.project_root)
        if not mvn:
            logger.error("Maven not found. Cannot run Java tracer.")
            return None

        # Determine package prefixes from module_root
        package_prefixes = self._detect_package_prefixes()
        if not package_prefixes:
            logger.error("Could not detect package prefixes for instrumentation.")
            return None

        # Determine source root for file path resolution
        source_root = find_source_root(self.project_root)
        source_root_str = str(source_root) if source_root else str(self.project_root / "src" / "main" / "java")

        # Build agent arguments
        agent_args = (
            f"packages={','.join(package_prefixes)}"
            f";output={self.trace_output_path}"
            f";sourceRoot={source_root_str}"
            f";maxFunctionCount={self.max_function_count}"
        )

        # Build the -javaagent argLine including module opens
        agent_arg_line = (
            f"-javaagent:{agent_jar}={agent_args}"
            " --add-opens java.base/java.util=ALL-UNNAMED"
            " --add-opens java.base/java.lang=ALL-UNNAMED"
            " --add-opens java.base/java.lang.reflect=ALL-UNNAMED"
            " --add-opens java.base/java.io=ALL-UNNAMED"
            " --add-opens java.base/java.math=ALL-UNNAMED"
            " --add-opens java.base/java.net=ALL-UNNAMED"
            " --add-opens java.base/java.util.zip=ALL-UNNAMED"
        )

        # Build Maven command
        cmd = [mvn, "test", "-B", "-fae", f"-DargLine={agent_arg_line}"]

        logger.info("Running Java tracer with agent: %s", agent_jar)
        logger.debug("Agent args: %s", agent_args)
        logger.debug("Maven command: %s", " ".join(cmd))

        env = os.environ.copy()

        try:
            result = subprocess.run(
                cmd, check=False, cwd=self.project_root, env=env, capture_output=True, text=True, timeout=self.timeout
            )

            if result.returncode != 0:
                logger.warning(
                    "Maven tests exited with code %d during tracing. "
                    "Some tests may have failed, but trace data may still be usable.",
                    result.returncode,
                )
                logger.debug("Maven stdout: %s", result.stdout[:2000] if result.stdout else "")
                logger.debug("Maven stderr: %s", result.stderr[:2000] if result.stderr else "")

            if self.trace_output_path.exists() and self.trace_output_path.stat().st_size > 0:
                logger.info("Java trace data written to: %s", self.trace_output_path)
                return self.trace_output_path
            logger.error("Java tracer did not produce output at %s", self.trace_output_path)
            return None

        except subprocess.TimeoutExpired:
            logger.exception("Java tracer timed out after %d seconds", self.timeout)
            return None
        except Exception:
            logger.exception("Java tracer failed")
            return None

    def _detect_package_prefixes(self) -> list[str]:
        """Detect Java package prefixes from the project's source root.

        Scans the source directory structure to find top-level packages.
        For example, if source root has com/example/myapp/, returns ["com.example.myapp"].
        """
        source_root = find_source_root(self.project_root)
        if source_root is None:
            source_root = self.project_root / "src" / "main" / "java"

        if not source_root.exists():
            logger.warning("Source root does not exist: %s", source_root)
            return []

        # Walk the source tree to find package directories containing .java files
        packages = set()
        for java_file in source_root.rglob("*.java"):
            try:
                rel_path = java_file.relative_to(source_root)
                # Get the package from the directory structure
                parts = list(rel_path.parent.parts)
                if parts:
                    # Use the top-level package (e.g., com.example)
                    # Include enough levels to be specific but not overly broad
                    depth = min(len(parts), 3)
                    package = ".".join(parts[:depth])
                    packages.add(package)
            except ValueError:
                continue

        if not packages:
            # Fallback: try reading package declarations from Java files
            for java_file in source_root.rglob("*.java"):
                try:
                    content = java_file.read_text(encoding="utf-8")
                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("package "):
                            pkg = line[8:].rstrip(";").strip()
                            # Use top 2-3 levels
                            parts = pkg.split(".")
                            depth = min(len(parts), 3)
                            packages.add(".".join(parts[:depth]))
                            break
                        if (
                            line
                            and not line.startswith("//")
                            and not line.startswith("/*")
                            and not line.startswith("*")
                        ):
                            break
                except Exception:
                    continue

        return sorted(packages)
