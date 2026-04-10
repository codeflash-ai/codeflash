from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from codeflash.languages.java.line_profiler import find_agent_jar
from codeflash.languages.java.replay_test import generate_replay_tests, parse_replay_test_metadata
from codeflash.languages.java.tracer import ADD_OPENS_FLAGS, JavaTracer

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "java_tracer_e2e"
WORKLOAD_SOURCE = FIXTURE_DIR / "src" / "main" / "java" / "com" / "example" / "Workload.java"
WORKLOAD_CLASS = "com.example.Workload"
WORKLOAD_PACKAGE = "com.example"


@pytest.fixture(scope="module")
def compiled_workload() -> Path:
    """Compile the Java workload fixture (once per module)."""
    classes_dir = FIXTURE_DIR / "target" / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["javac", "--release", "11", "-d", str(classes_dir), str(WORKLOAD_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"javac failed: {result.stderr}"
    return classes_dir


@pytest.fixture
def trace_db(tmp_path: Path) -> Path:
    return tmp_path / "trace.db"


class TestTracingAgent:
    def test_agent_jar_found(self) -> None:
        jar = find_agent_jar()
        assert jar is not None, "codeflash-runtime JAR not found"
        assert jar.exists()

    def test_agent_captures_invocations(self, compiled_workload: Path, trace_db: Path) -> None:
        """Test that the tracing agent captures method invocations into SQLite."""
        agent_jar = find_agent_jar()
        assert agent_jar is not None

        import json

        config = {
            "dbPath": str(trace_db),
            "packages": [WORKLOAD_PACKAGE],
            "excludePackages": [],
            "maxFunctionCount": 256,
            "timeout": 0,
            "projectRoot": str(FIXTURE_DIR),
        }
        config_path = trace_db.with_suffix(".config.json")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        result = subprocess.run(
            [
                "java",
                *ADD_OPENS_FLAGS.split(),
                f"-javaagent:{agent_jar}=trace={config_path}",
                "-cp",
                str(compiled_workload),
                WORKLOAD_CLASS,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert "Workload complete." in result.stdout, f"Workload failed to run: {result.stderr}"
        assert trace_db.exists(), "Trace DB not created"

        # Verify database contents
        conn = sqlite3.connect(str(trace_db))
        try:
            rows = conn.execute("SELECT function, classname, descriptor, length(args) FROM function_calls").fetchall()
            assert len(rows) >= 3, f"Expected at least 3 captured invocations, got {len(rows)}"

            # Check that specific methods were captured
            functions = {row[0] for row in rows}
            assert "computeSum" in functions
            assert "repeatString" in functions

            # Verify all rows have non-empty args blobs
            for row in rows:
                assert row[3] > 0, f"Empty args blob for {row[0]}"

            # Verify metadata
            metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
            assert "totalCaptures" in metadata
            assert int(metadata["totalCaptures"]) >= 3
        finally:
            conn.close()

    def test_max_function_count_limit(self, compiled_workload: Path, trace_db: Path) -> None:
        """Test that maxFunctionCount limits captures per method."""
        agent_jar = find_agent_jar()
        assert agent_jar is not None

        import json

        config = {
            "dbPath": str(trace_db),
            "packages": [WORKLOAD_PACKAGE],
            "excludePackages": [],
            "maxFunctionCount": 2,
            "timeout": 0,
            "projectRoot": str(FIXTURE_DIR),
        }
        config_path = trace_db.with_suffix(".config.json")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        subprocess.run(
            [
                "java",
                *ADD_OPENS_FLAGS.split(),
                f"-javaagent:{agent_jar}=trace={config_path}",
                "-cp",
                str(compiled_workload),
                WORKLOAD_CLASS,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        conn = sqlite3.connect(str(trace_db))
        try:
            # computeSum is called 2 times (direct calls in main)
            compute_count = conn.execute(
                "SELECT COUNT(*) FROM function_calls WHERE function = 'computeSum'"
            ).fetchone()[0]
            assert compute_count <= 2, f"Expected at most 2 computeSum captures, got {compute_count}"
        finally:
            conn.close()


class TestReplayTestGeneration:
    def test_generates_test_files(self, compiled_workload: Path, trace_db: Path, tmp_path: Path) -> None:
        """Test that replay test files are generated from trace DB."""
        # First, create a trace
        agent_jar = find_agent_jar()
        assert agent_jar is not None

        import json

        config = {
            "dbPath": str(trace_db),
            "packages": [WORKLOAD_PACKAGE],
            "excludePackages": [],
            "maxFunctionCount": 256,
            "timeout": 0,
            "projectRoot": str(FIXTURE_DIR),
        }
        config_path = trace_db.with_suffix(".config.json")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        subprocess.run(
            [
                "java",
                *ADD_OPENS_FLAGS.split(),
                f"-javaagent:{agent_jar}=trace={config_path}",
                "-cp",
                str(compiled_workload),
                WORKLOAD_CLASS,
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )

        # Generate replay tests
        output_dir = tmp_path / "replay_tests"
        count = generate_replay_tests(
            trace_db_path=trace_db,
            output_dir=output_dir,
            project_root=FIXTURE_DIR,
        )

        assert count >= 1, f"Expected at least 1 test file, got {count}"
        test_files = list(output_dir.glob("*.java"))
        assert len(test_files) >= 1

        # Find the main workload test file
        workload_files = [f for f in test_files if "Workload" in f.name and "ConstructorAccess" not in f.name]
        assert len(workload_files) == 1
        content = workload_files[0].read_text(encoding="utf-8")
        assert "package codeflash.replay;" in content
        assert "import org.junit.jupiter.api.Test;" in content
        assert "ReplayHelper" in content
        assert "replay_computeSum_0" in content
        assert "replay_repeatString_0" in content

    def test_metadata_parsing(self, compiled_workload: Path, trace_db: Path, tmp_path: Path) -> None:
        """Test that metadata comments are correctly parsed from generated tests."""
        agent_jar = find_agent_jar()
        assert agent_jar is not None

        import json

        config = {
            "dbPath": str(trace_db),
            "packages": [WORKLOAD_PACKAGE],
            "excludePackages": [],
            "maxFunctionCount": 256,
            "timeout": 0,
            "projectRoot": str(FIXTURE_DIR),
        }
        config_path = trace_db.with_suffix(".config.json")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        subprocess.run(
            [
                "java",
                *ADD_OPENS_FLAGS.split(),
                f"-javaagent:{agent_jar}=trace={config_path}",
                "-cp",
                str(compiled_workload),
                WORKLOAD_CLASS,
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )

        output_dir = tmp_path / "replay_tests"
        generate_replay_tests(trace_db_path=trace_db, output_dir=output_dir, project_root=FIXTURE_DIR)

        test_files = [f for f in output_dir.glob("*.java") if "ConstructorAccess" not in f.name]
        test_file = test_files[0]
        metadata = parse_replay_test_metadata(test_file)

        assert "functions" in metadata
        assert "trace_file" in metadata
        assert "classname" in metadata
        assert "computeSum" in metadata["functions"]
        assert metadata["classname"] == "com.example.Workload"
        assert metadata["trace_file"] == trace_db.as_posix()


class TestJavaTracerOrchestration:
    def test_two_stage_trace(self, compiled_workload: Path, tmp_path: Path) -> None:
        """Test the full two-stage JavaTracer flow (JFR + agent)."""
        trace_db_path = tmp_path / "trace.db"
        tracer = JavaTracer()

        trace_db, _jfr_file = tracer.trace(
            java_command=["java", "-cp", str(compiled_workload), WORKLOAD_CLASS],
            trace_db_path=trace_db_path,
            packages=[WORKLOAD_PACKAGE],
            project_root=FIXTURE_DIR,
        )

        assert trace_db.exists(), "Trace DB not created by JavaTracer"

        # Verify trace DB has captures
        conn = sqlite3.connect(str(trace_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM function_calls").fetchone()[0]
            assert count >= 5, f"Expected at least 5 captured invocations, got {count}"
        finally:
            conn.close()

    def test_full_trace_and_replay_generation(self, compiled_workload: Path, tmp_path: Path) -> None:
        """Test the full flow: trace → generate replay tests."""
        from codeflash.languages.java.tracer import run_java_tracer

        trace_db_path = tmp_path / "trace.db"
        output_dir = tmp_path / "replay_tests"

        trace_db, _jfr_file, test_count = run_java_tracer(
            java_command=["java", "-cp", str(compiled_workload), WORKLOAD_CLASS],
            trace_db_path=trace_db_path,
            packages=[WORKLOAD_PACKAGE],
            project_root=FIXTURE_DIR,
            output_dir=output_dir,
        )

        assert trace_db.exists()
        assert test_count >= 1

        # Verify the generated test files
        test_files = list(output_dir.glob("*.java"))
        assert len(test_files) >= 1
        workload_files = [f for f in test_files if "Workload" in f.name and "ConstructorAccess" not in f.name]
        assert len(workload_files) == 1
        content = workload_files[0].read_text(encoding="utf-8")
        assert "replay_computeSum" in content
        assert "replay_repeatString" in content

    def test_package_detection(self) -> None:
        """Test that package detection finds Java packages from source files."""
        packages = JavaTracer.detect_packages_from_source(FIXTURE_DIR)
        assert "com.example" in packages
