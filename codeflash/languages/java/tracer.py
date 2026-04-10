from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import TYPE_CHECKING

from codeflash.code_utils.env_utils import is_ci
from codeflash.languages.java.line_profiler import find_agent_jar
from codeflash.languages.java.replay_test import generate_replay_tests

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

GRACEFUL_SHUTDOWN_WAIT = 5  # seconds to wait after SIGTERM before SIGKILL


def _run_java_with_graceful_timeout(
    java_command: list[str], env: dict[str, str], timeout: int, stage_name: str
) -> None:
    """Run a Java command with graceful timeout handling.

    Sends SIGTERM first (allowing JFR dump and shutdown hooks to run),
    then SIGKILL if the process doesn't exit within GRACEFUL_SHUTDOWN_WAIT seconds.
    """
    if not timeout:
        subprocess.run(java_command, env=env, check=False)
        return

    import signal

    proc = subprocess.Popen(java_command, env=env)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s stage timed out after %d seconds, sending SIGTERM for graceful shutdown...", stage_name, timeout
        )
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=GRACEFUL_SHUTDOWN_WAIT)
        except subprocess.TimeoutExpired:
            logger.warning("%s stage did not exit after SIGTERM, sending SIGKILL", stage_name)
            proc.kill()
            proc.wait()


# --add-opens flags needed for Kryo serialization on Java 16+
ADD_OPENS_FLAGS = (
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.math=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.time=ALL-UNNAMED"
)


class JavaTracer:
    """Orchestrates Java tracing: combined JFR profiling + argument capture in a single JVM invocation."""

    def trace(
        self,
        java_command: list[str],
        trace_db_path: Path,
        packages: list[str],
        project_root: Path | None = None,
        max_function_count: int = 256,
        timeout: int = 0,
    ) -> tuple[Path, Path]:
        """Run the Java program once with both JFR profiling and argument capture.

        Returns (trace_db_path, jfr_file_path).
        """
        jfr_file = trace_db_path.with_suffix(".jfr")
        trace_db_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = self.create_tracer_config(
            trace_db_path, packages, project_root=project_root, max_function_count=max_function_count, timeout=timeout
        )
        combined_env = self.build_combined_env(jfr_file, config_path)

        logger.info("Running combined JFR profiling + argument capture...")
        _run_java_with_graceful_timeout(java_command, combined_env, timeout, "Combined tracing")

        if not jfr_file.exists():
            logger.warning("JFR file was not created at %s", jfr_file)
        if not trace_db_path.exists():
            logger.error("Trace database was not created at %s", trace_db_path)

        return trace_db_path, jfr_file

    def create_tracer_config(
        self,
        trace_db_path: Path,
        packages: list[str],
        project_root: Path | None = None,
        max_function_count: int = 256,
        timeout: int = 0,
    ) -> Path:
        config = {
            "dbPath": str(trace_db_path.resolve()),
            "packages": packages,
            "excludePackages": [],
            "maxFunctionCount": max_function_count,
            "timeout": timeout,
            "projectRoot": str(project_root.resolve()) if project_root else "",
            "inMemoryDb": is_ci(),
        }

        config_path = trace_db_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def build_jfr_env(self, jfr_file: Path) -> dict[str, str]:
        env = os.environ.copy()
        jfr_opts = f"-XX:StartFlightRecording=filename={jfr_file.resolve()},settings=profile,dumponexit=true"
        existing = env.get("JAVA_TOOL_OPTIONS", "")
        env["JAVA_TOOL_OPTIONS"] = f"{existing} {jfr_opts}".strip()
        return env

    def build_agent_env(self, config_path: Path, classpath: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        agent_jar = find_agent_jar(classpath=classpath)
        if agent_jar is None:
            msg = "codeflash-runtime JAR not found, cannot run tracing agent"
            raise FileNotFoundError(msg)

        agent_opts = f"{ADD_OPENS_FLAGS} -javaagent:{agent_jar}=trace={config_path.resolve()}"
        existing = env.get("JAVA_TOOL_OPTIONS", "")
        env["JAVA_TOOL_OPTIONS"] = f"{existing} {agent_opts}".strip()
        return env

    def build_combined_env(self, jfr_file: Path, config_path: Path, classpath: str | None = None) -> dict[str, str]:
        """Build env with both JFR recording and tracing agent in a single JAVA_TOOL_OPTIONS."""
        env = os.environ.copy()
        jfr_opts = (
            f"-XX:StartFlightRecording=filename={jfr_file.resolve()},settings=profile,dumponexit=true"
            ",jdk.ExecutionSample#period=1ms"
        )
        agent_jar = find_agent_jar(classpath=classpath)
        if agent_jar is None:
            msg = "codeflash-runtime JAR not found, cannot run tracing agent"
            raise FileNotFoundError(msg)
        agent_opts = f"{ADD_OPENS_FLAGS} -javaagent:{agent_jar}=trace={config_path.resolve()}"
        existing = env.get("JAVA_TOOL_OPTIONS", "")
        env["JAVA_TOOL_OPTIONS"] = f"{existing} {jfr_opts} {agent_opts}".strip()
        return env

    @staticmethod
    def detect_packages_from_source(module_root: Path) -> list[str]:
        """Scan Java files for package declarations and return unique package prefixes."""
        packages: set[str] = set()
        for java_file in module_root.rglob("*.java"):
            try:
                in_block_comment = False
                with java_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if in_block_comment:
                            if "*/" in stripped:
                                in_block_comment = False
                            continue
                        if stripped.startswith("/*"):
                            if "*/" not in stripped:
                                in_block_comment = True
                            continue
                        if stripped.startswith("package "):
                            pkg = stripped[8:].rstrip(";").strip()
                            parts = pkg.split(".")
                            prefix = ".".join(parts[: min(3, len(parts))])
                            packages.add(prefix)
                            break
                        if stripped and not stripped.startswith("//"):
                            break
            except (OSError, UnicodeDecodeError):
                continue

        return sorted(packages)


def run_java_tracer(
    java_command: list[str],
    trace_db_path: Path,
    packages: list[str],
    project_root: Path,
    output_dir: Path,
    max_function_count: int = 256,
    timeout: int = 0,
    max_run_count: int = 256,
    test_framework: str = "junit5",
) -> tuple[Path, Path, int]:
    """High-level entry point: trace a Java command and generate replay tests.

    Returns (trace_db_path, jfr_file, test_count).
    """
    tracer = JavaTracer()
    trace_db, jfr_file = tracer.trace(
        java_command=java_command,
        trace_db_path=trace_db_path,
        packages=packages,
        project_root=project_root,
        max_function_count=max_function_count,
        timeout=timeout,
    )

    test_count = generate_replay_tests(
        trace_db_path=trace_db,
        output_dir=output_dir,
        project_root=project_root,
        max_run_count=max_run_count,
        test_framework=test_framework,
    )

    return trace_db, jfr_file, test_count
