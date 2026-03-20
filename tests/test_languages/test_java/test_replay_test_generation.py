"""Tests for Java replay test generation — JUnit 4/5 support, overload handling, instrumentation skip."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codeflash.languages.java.replay_test import generate_replay_tests, parse_replay_test_metadata


@pytest.fixture
def trace_db(tmp_path: Path) -> Path:
    """Create a trace database with sample function calls."""
    db_path = tmp_path / "trace.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE function_calls("
        "type TEXT, function TEXT, classname TEXT, filename TEXT, "
        "line_number INTEGER, descriptor TEXT, time_ns INTEGER, args BLOB)"
    )
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("call", "add", "com.example.Calculator", "Calculator.java", 10, "(II)I", 1000, b"\x00"),
    )
    conn.execute(
        "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("call", "add", "com.example.Calculator", "Calculator.java", 10, "(II)I", 2000, b"\x00"),
    )
    conn.execute(
        "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("call", "multiply", "com.example.Calculator", "Calculator.java", 20, "(II)I", 3000, b"\x00"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def trace_db_overloaded(tmp_path: Path) -> Path:
    """Create a trace database with overloaded methods (same name, different descriptors)."""
    db_path = tmp_path / "trace_overloaded.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE function_calls("
        "type TEXT, function TEXT, classname TEXT, filename TEXT, "
        "line_number INTEGER, descriptor TEXT, time_ns INTEGER, args BLOB)"
    )
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    # Two overloads of estimateKeySize with different descriptors
    for i in range(3):
        conn.execute(
            "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("call", "estimateKeySize", "com.example.Command", "Command.java", 10, "(I)I", i * 1000, b"\x00"),
        )
    for i in range(2):
        conn.execute(
            "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "call",
                "estimateKeySize",
                "com.example.Command",
                "Command.java",
                15,
                "(Ljava/lang/String;)I",
                (i + 10) * 1000,
                b"\x00",
            ),
        )
    conn.commit()
    conn.close()
    return db_path


class TestGenerateReplayTestsJunit5:
    def test_generates_junit5_by_default(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        count = generate_replay_tests(trace_db, output_dir, tmp_path)
        assert count == 1

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "import org.junit.jupiter.api.Test;" in content
        assert "import org.junit.jupiter.api.AfterAll;" in content
        assert "@Test void replay_add_0()" in content

    def test_junit5_class_is_package_private(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path)

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "class ReplayTest_" in content
        assert "public class ReplayTest_" not in content


class TestGenerateReplayTestsJunit4:
    def test_generates_junit4_imports(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        count = generate_replay_tests(trace_db, output_dir, tmp_path, test_framework="junit4")
        assert count == 1

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "import org.junit.Test;" in content
        assert "import org.junit.AfterClass;" in content
        assert "org.junit.jupiter" not in content

    def test_junit4_methods_are_public(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path, test_framework="junit4")

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "@Test public void replay_add_0()" in content

    def test_junit4_class_is_public(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path, test_framework="junit4")

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "public class ReplayTest_" in content

    def test_junit4_cleanup_uses_afterclass(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path, test_framework="junit4")

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")
        assert "@AfterClass" in content
        assert "@AfterAll" not in content


class TestOverloadedMethods:
    def test_no_duplicate_method_names(self, trace_db_overloaded: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        count = generate_replay_tests(trace_db_overloaded, output_dir, tmp_path)
        assert count == 1

        test_file = list(output_dir.glob("*.java"))[0]
        content = test_file.read_text(encoding="utf-8")

        # Should have 5 unique methods (3 from first overload + 2 from second)
        assert "replay_estimateKeySize_0" in content
        assert "replay_estimateKeySize_1" in content
        assert "replay_estimateKeySize_2" in content
        assert "replay_estimateKeySize_3" in content
        assert "replay_estimateKeySize_4" in content

        # Verify no duplicates by counting occurrences
        lines = content.splitlines()
        method_lines = [l for l in lines if "void replay_estimateKeySize_" in l]
        method_names = [l.split("void ")[1].split("(")[0] for l in method_lines]
        assert len(method_names) == len(set(method_names)), f"Duplicate methods: {method_names}"


class TestReplayTestInstrumentationSkip:
    def test_skip_instrumentation_for_replay_tests(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path)

        test_file = list(output_dir.glob("*.java"))[0]

        from codeflash.languages.java.support import JavaSupport

        support = JavaSupport()

        # Instrument in behavior mode
        success, instrumented = support.instrument_existing_test(
            test_path=test_file,
            call_positions=[],
            function_to_optimize=None,
            tests_project_root=tmp_path,
            mode="behavior",
        )
        assert success
        assert instrumented is not None

        # Should just rename the class, no behavior setup code
        assert "__perfinstrumented" in instrumented
        assert "CODEFLASH_LOOP_INDEX" not in instrumented
        assert "// Codeflash behavior instrumentation" not in instrumented

    def test_skip_instrumentation_for_perf_mode(self, trace_db: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        generate_replay_tests(trace_db, output_dir, tmp_path)

        test_file = list(output_dir.glob("*.java"))[0]

        from codeflash.languages.java.support import JavaSupport

        support = JavaSupport()

        success, instrumented = support.instrument_existing_test(
            test_path=test_file,
            call_positions=[],
            function_to_optimize=None,
            tests_project_root=tmp_path,
            mode="performance",
        )
        assert success
        assert "__perfonlyinstrumented" in instrumented

    def test_regular_tests_still_get_instrumented(self, tmp_path: Path) -> None:
        """Non-replay test files should still be instrumented normally."""
        from codeflash.languages.java.discovery import discover_functions_from_source

        src = """
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
"""
        funcs = discover_functions_from_source(src, tmp_path / "Calculator.java")
        target = funcs[0]

        test_file = tmp_path / "CalculatorTest.java"
        test_file.write_text(
            """
import org.junit.jupiter.api.Test;
public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        calc.add(1, 2);
    }
}
""",
            encoding="utf-8",
        )

        from codeflash.languages.java.support import JavaSupport

        support = JavaSupport()
        success, instrumented = support.instrument_existing_test(
            test_path=test_file,
            call_positions=[],
            function_to_optimize=target,
            tests_project_root=tmp_path,
            mode="behavior",
        )
        assert success
        # Regular tests should have behavior instrumentation
        assert "CODEFLASH_LOOP_INDEX" in instrumented
