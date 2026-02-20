"""End-to-end Java run-and-parse integration tests.

Analogous to tests/test_languages/test_javascript_run_and_parse.py and
tests/test_instrument_tests.py::test_perfinjector_bubble_sort_results for Python.

Tests the full pipeline: instrument → run → parse → assert precise field values.
"""

import os
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.languages.current import set_current_language
from codeflash.languages.java.instrumentation import instrument_existing_test
from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
from codeflash.optimization.optimizer import Optimizer

os.environ.setdefault("CODEFLASH_API_KEY", "cf-test-key")

# Kryo ZigZag-encoded integers: pattern is bytes([0x02, 2*N]) for int N.
KRYO_INT_5 = bytes([0x02, 0x0A])
KRYO_INT_6 = bytes([0x02, 0x0C])

POM_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>codeflash-test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.junit.platform</groupId>
            <artifactId>junit-platform-console-standalone</artifactId>
            <version>1.9.3</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.xerial</groupId>
            <artifactId>sqlite-jdbc</artifactId>
            <version>3.44.1.0</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>com.google.code.gson</groupId>
            <artifactId>gson</artifactId>
            <version>2.10.1</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>com.codeflash</groupId>
            <artifactId>codeflash-runtime</artifactId>
            <version>1.0.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
                <configuration>
                    <redirectTestOutputToFile>false</redirectTestOutputToFile>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
"""


def skip_if_maven_not_available():
    from codeflash.languages.java.build_tools import find_maven_executable

    if not find_maven_executable():
        pytest.skip("Maven not available")


@pytest.fixture
def java_project(tmp_path: Path):
    """Create a temporary Maven project and set up Java language context."""
    import codeflash.languages.current as current_module

    current_module._current_language = None
    set_current_language(Language.JAVA)

    src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
    test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (tmp_path / "pom.xml").write_text(POM_CONTENT, encoding="utf-8")

    yield tmp_path, src_dir, test_dir

    current_module._current_language = None
    set_current_language(Language.PYTHON)


def _make_optimizer(project_root: Path, test_dir: Path, function_name: str, src_file: Path) -> tuple:
    """Create an Optimizer and FunctionOptimizer for the given function."""
    fto = FunctionToOptimize(
        function_name=function_name,
        file_path=src_file,
        parents=[],
        language="java",
    )
    opt = Optimizer(
        Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        )
    )
    func_optimizer = opt.create_function_optimizer(fto)
    assert func_optimizer is not None
    return fto, func_optimizer


def _create_test_results_db(path: Path, results: list[dict]) -> None:
    """Create a SQLite database with test_results table matching instrumentation schema."""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE test_results (
            test_module_path TEXT,
            test_class_name TEXT,
            test_function_name TEXT,
            function_getting_tested TEXT,
            loop_index INTEGER,
            iteration_id TEXT,
            runtime INTEGER,
            return_value BLOB,
            verification_type TEXT
        )
        """
    )
    for row in results:
        cursor.execute(
            """
            INSERT INTO test_results
            (test_module_path, test_class_name, test_function_name,
             function_getting_tested, loop_index, iteration_id,
             runtime, return_value, verification_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("test_module_path", "AdderTest"),
                row.get("test_class_name", "AdderTest"),
                row.get("test_function_name", "testAdd"),
                row.get("function_getting_tested", "add"),
                row.get("loop_index", 1),
                row.get("iteration_id", "1_0"),
                row.get("runtime", 1000000),
                row.get("return_value"),
                row.get("verification_type", "FUNCTION_CALL"),
            ),
        )
    conn.commit()
    conn.close()


ADDER_JAVA = """package com.example;
public class Adder {
    public int add(int a, int b) {
        return a + b;
    }
}
"""

ADDER_TEST_JAVA = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AdderTest {
    @Test
    public void testAdd() {
        Adder adder = new Adder();
        assertEquals(5, adder.add(2, 3));
    }
}
"""

PRECISE_WAITER_JAVA = """package com.example;
public class PreciseWaiter {
    // Volatile field to prevent compiler optimization of busy loop
    private volatile long busyWork = 0;

    /**
     * Precise busy-wait using System.nanoTime() (monotonic clock).
     * Performs continuous CPU work to prevent CPU sleep/yield.
     * Achieves <1% variance by never yielding the CPU to the scheduler.
     */
    public long waitNanos(long targetNanos) {
        long startTime = System.nanoTime();
        long endTime = startTime + targetNanos;

        while (System.nanoTime() < endTime) {
            // Busy work to keep CPU occupied and prevent optimizations
            busyWork++;
        }

        // Return actual elapsed time for verification
        return System.nanoTime() - startTime;
    }
}
"""


class TestJavaRunAndParseBehavior:
    def test_behavior_single_test_method(self, java_project):
        """Full pipeline: instrument → run → parse with precise field assertions."""
        skip_if_maven_not_available()
        project_root, src_dir, test_dir = java_project

        (src_dir / "Adder.java").write_text(ADDER_JAVA, encoding="utf-8")
        test_file = test_dir / "AdderTest.java"
        test_file.write_text(ADDER_TEST_JAVA, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "Adder.java",
            starting_line=3,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )
        success, instrumented = instrument_existing_test(
            test_string=ADDER_TEST_JAVA, function_to_optimize=func_info, mode="behavior", test_path=test_file
        )
        assert success

        instrumented_file = test_dir / "AdderTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        _, func_optimizer = _make_optimizer(project_root, test_dir, "add", src_dir / "Adder.java")
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=instrumented_file,
                    test_type=TestType.EXISTING_UNIT_TEST,
                    original_file_path=test_file,
                    benchmarking_file_path=instrumented_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=1,
            max_outer_loops=2,
            testing_time=0.1,
        )

        assert len(test_results.test_results) >= 1
        result = test_results.test_results[0]
        assert result.did_pass is True
        assert result.runtime is not None
        assert result.runtime > 0
        assert result.id.test_function_name == "testAdd"
        assert result.id.test_class_name == "AdderTest"
        assert result.id.function_getting_tested == "add"

    def test_behavior_multiple_test_methods(self, java_project):
        """Two @Test methods — both should appear in parsed results."""
        skip_if_maven_not_available()
        project_root, src_dir, test_dir = java_project

        (src_dir / "Adder.java").write_text(ADDER_JAVA, encoding="utf-8")

        multi_test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AdderMultiTest {
    @Test
    public void testAddPositive() {
        Adder adder = new Adder();
        assertEquals(5, adder.add(2, 3));
    }

    @Test
    public void testAddZero() {
        Adder adder = new Adder();
        assertEquals(0, adder.add(0, 0));
    }
}
"""
        test_file = test_dir / "AdderMultiTest.java"
        test_file.write_text(multi_test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "Adder.java",
            starting_line=3,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )
        success, instrumented = instrument_existing_test(
            test_string=multi_test_source, function_to_optimize=func_info, mode="behavior", test_path=test_file
        )
        assert success

        instrumented_file = test_dir / "AdderMultiTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        _, func_optimizer = _make_optimizer(project_root, test_dir, "add", src_dir / "Adder.java")
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=instrumented_file,
                    test_type=TestType.EXISTING_UNIT_TEST,
                    original_file_path=test_file,
                    benchmarking_file_path=instrumented_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=1,
            max_outer_loops=2,
            testing_time=0.1,
        )

        assert len(test_results.test_results) >= 2
        for result in test_results.test_results:
            assert result.did_pass is True
            assert result.runtime is not None
            assert result.runtime > 0

        test_names = {r.id.test_function_name for r in test_results.test_results}
        assert "testAddPositive" in test_names
        assert "testAddZero" in test_names

    def test_behavior_return_value_correctness(self, tmp_path):
        """Verify the Comparator JAR correctly identifies equivalent vs. differing results.

        Uses manually-constructed SQLite databases with known Kryo-encoded values
        to exercise the full comparator pipeline without requiring Maven.
        """
        from codeflash.languages.java.comparator import compare_test_results

        row = {
            "test_module_path": "AdderTest",
            "test_class_name": "AdderTest",
            "test_function_name": "testAdd",
            "function_getting_tested": "add",
            "loop_index": 1,
            "iteration_id": "1_0",
            "runtime": 1000000,
            "return_value": KRYO_INT_5,  # Kryo ZigZag encoding of int 5
            "verification_type": "FUNCTION_CALL",
        }

        original_db = tmp_path / "original.sqlite"
        candidate_db = tmp_path / "candidate.sqlite"
        wrong_db = tmp_path / "wrong.sqlite"

        _create_test_results_db(original_db, [row])
        _create_test_results_db(candidate_db, [row])  # identical → equivalent
        _create_test_results_db(wrong_db, [{**row, "return_value": KRYO_INT_6}])  # int 6 ≠ 5

        equivalent, diffs = compare_test_results(original_db, candidate_db)
        assert equivalent is True
        assert len(diffs) == 0

        equivalent, diffs = compare_test_results(original_db, wrong_db)
        assert equivalent is False


class TestJavaRunAndParsePerformance:
    """Tests that the performance instrumentation produces correct timing data.

    Uses precise busy-wait with System.nanoTime() (monotonic clock) to achieve
    <5% timing variance, accounting for JIT warmup effects where first iterations
    are cold and subsequent iterations benefit from JIT optimization.
    """

    PRECISE_WAITER_TEST = """package com.example;

import org.junit.jupiter.api.Test;

public class PreciseWaiterTest {
    @Test
    public void testWaitNanos() {
        // Wait exactly 10 milliseconds (10,000,000 nanoseconds)
        new PreciseWaiter().waitNanos(10_000_000L);
    }
}
"""

    def _setup_precise_waiter_project(self, java_project):
        """Write PreciseWaiter.java to the project and return (project_root, src_dir, test_dir)."""
        project_root, src_dir, test_dir = java_project
        (src_dir / "PreciseWaiter.java").write_text(PRECISE_WAITER_JAVA, encoding="utf-8")
        return project_root, src_dir, test_dir

    def _instrument_and_run(self, project_root, src_dir, test_dir, test_source, test_filename, inner_iterations=2):
        """Instrument a performance test and run it, returning test_results."""
        test_file = test_dir / test_filename
        test_file.write_text(test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="waitNanos",
            file_path=src_dir / "PreciseWaiter.java",
            starting_line=11,
            ending_line=22,
            parents=[],
            is_method=True,
            language="java",
        )
        success, instrumented = instrument_existing_test(
            test_string=test_source, function_to_optimize=func_info, mode="performance", test_path=test_file
        )
        assert success

        stem = test_filename.replace(".java", "")
        instrumented_filename = f"{stem}__perfonlyinstrumented.java"
        instrumented_file = test_dir / instrumented_filename
        instrumented_file.write_text(instrumented, encoding="utf-8")

        _, func_optimizer = _make_optimizer(project_root, test_dir, "waitNanos", src_dir / "PreciseWaiter.java")
        func_optimizer.test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.EXISTING_UNIT_TEST,
                    original_file_path=test_file,
                    benchmarking_file_path=instrumented_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            min_outer_loops=2,
            max_outer_loops=2,
            inner_iterations=inner_iterations,
            testing_time=0.0,
        )
        return test_results

    def test_performance_inner_loop_count_and_timing(self, java_project):
        """2 outer × 2 inner = 4 results with <5% variance and accurate 10ms timing."""
        skip_if_maven_not_available()
        project_root, src_dir, test_dir = self._setup_precise_waiter_project(java_project)

        test_results = self._instrument_and_run(
            project_root,
            src_dir,
            test_dir,
            self.PRECISE_WAITER_TEST,
            "PreciseWaiterTest.java",
            inner_iterations=2,
        )

        # 2 outer loops × 2 inner iterations = 4 total results
        assert len(test_results.test_results) == 4, (
            f"Expected 4 results (2 outer loops × 2 inner iterations), got {len(test_results.test_results)}"
        )

        # Verify all tests passed and collect runtimes
        runtimes = []
        for result in test_results.test_results:
            assert result.did_pass is True
            assert result.runtime is not None
            assert result.runtime > 0
            runtimes.append(result.runtime)

        # Verify timing consistency using coefficient of variation (stddev/mean)
        import statistics

        mean_runtime = statistics.mean(runtimes)
        stddev_runtime = statistics.stdev(runtimes)
        coefficient_of_variation = stddev_runtime / mean_runtime

        # Target: 10ms (10,000,000 ns), allow <5% coefficient of variation
        # (accounts for JIT warmup - first iteration is cold, subsequent are optimized)
        expected_ns = 10_000_000
        runtimes_ms = [r / 1_000_000 for r in runtimes]

        assert coefficient_of_variation < 0.05, (
            f"Timing variance too high: CV={coefficient_of_variation:.2%} (should be <5%). "
            f"Runtimes: {runtimes_ms} ms (mean={mean_runtime / 1_000_000:.3f}ms)"
        )

        # Verify measured time is close to expected 10ms (allow ±5% for JIT warmup)
        assert expected_ns * 0.95 <= mean_runtime <= expected_ns * 1.05, (
            f"Mean runtime {mean_runtime / 1_000_000:.3f}ms not close to expected 10.0ms"
        )

        # Verify total_passed_runtime sums minimum runtime per test case
        # InvocationId includes iteration_id, so each inner iteration is a separate "test case"
        # With 2 inner iterations: 2 test cases (iteration_id=0 and iteration_id=1)
        # total = min(outer loop runtimes for iter 0) + min(outer loop runtimes for iter 1) ≈ 20ms
        total_runtime = test_results.total_passed_runtime()
        runtime_by_test = test_results.usable_runtime_data_by_test_case()

        # Should have 2 test cases (one per inner iteration)
        assert len(runtime_by_test) == 2, (
            f"Expected 2 test cases (iteration_id=0 and 1), got {len(runtime_by_test)}"
        )

        # Each test case should have 2 runtimes (2 outer loops)
        for test_id, test_runtimes in runtime_by_test.items():
            assert len(test_runtimes) == 2, (
                f"Expected 2 runtimes (2 outer loops) for {test_id.iteration_id}, got {len(test_runtimes)}"
            )

        # Total should be sum of 2 minimums (one per inner iteration) ≈ 20ms
        # Minimums filter out JIT warmup, so use tighter ±2% tolerance
        expected_total_ns = 2 * expected_ns
        assert expected_total_ns * 0.98 <= total_runtime <= expected_total_ns * 1.02, (
            f"total_passed_runtime {total_runtime / 1_000_000:.3f}ms not close to expected "
            f"{expected_total_ns / 1_000_000:.1f}ms (2 inner iterations × 10ms each, ±2%)"
        )

    def test_performance_multiple_test_methods_inner_loop(self, java_project):
        """Two @Test methods: 2 outer × 2 inner = 8 results with <5% variance."""
        skip_if_maven_not_available()
        project_root, src_dir, test_dir = self._setup_precise_waiter_project(java_project)

        multi_test_source = """package com.example;

import org.junit.jupiter.api.Test;

public class PreciseWaiterMultiTest {
    @Test
    public void testWaitNanos1() {
        // Wait exactly 10 milliseconds
        new PreciseWaiter().waitNanos(10_000_000L);
    }

    @Test
    public void testWaitNanos2() {
        // Wait exactly 10 milliseconds
        new PreciseWaiter().waitNanos(10_000_000L);
    }
}
"""
        test_results = self._instrument_and_run(
            project_root,
            src_dir,
            test_dir,
            multi_test_source,
            "PreciseWaiterMultiTest.java",
            inner_iterations=2,
        )

        # 2 test methods × 2 outer loops × 2 inner iterations = 8 total results
        assert len(test_results.test_results) == 8, (
            f"Expected 8 results (2 methods × 2 outer loops × 2 inner iterations), got {len(test_results.test_results)}"
        )

        # Verify all tests passed and collect runtimes
        runtimes = []
        for result in test_results.test_results:
            assert result.did_pass is True
            assert result.runtime is not None
            assert result.runtime > 0
            runtimes.append(result.runtime)

        # Verify timing consistency using coefficient of variation (stddev/mean)
        import statistics

        mean_runtime = statistics.mean(runtimes)
        stddev_runtime = statistics.stdev(runtimes)
        coefficient_of_variation = stddev_runtime / mean_runtime

        # Target: 10ms (10,000,000 ns), allow <5% coefficient of variation
        # (accounts for JIT warmup - first iteration is cold, subsequent are optimized)
        expected_ns = 10_000_000
        runtimes_ms = [r / 1_000_000 for r in runtimes]

        assert coefficient_of_variation < 0.05, (
            f"Timing variance too high: CV={coefficient_of_variation:.2%} (should be <5%). "
            f"Runtimes: {runtimes_ms} ms (mean={mean_runtime / 1_000_000:.3f}ms)"
        )

        # Verify measured time is close to expected 10ms (allow ±5% for JIT warmup)
        assert expected_ns * 0.95 <= mean_runtime <= expected_ns * 1.05, (
            f"Mean runtime {mean_runtime / 1_000_000:.3f}ms not close to expected 10.0ms"
        )

        # Verify total_passed_runtime sums minimum runtime per test case
        # InvocationId includes iteration_id, so: 2 test methods × 2 inner iterations = 4 "test cases"
        # total = sum of 4 minimums (each test method × inner iteration gets min of 2 outer loops) ≈ 40ms
        total_runtime = test_results.total_passed_runtime()
        runtime_by_test = test_results.usable_runtime_data_by_test_case()

        # Should have 4 test cases (2 test methods × 2 inner iterations)
        assert len(runtime_by_test) == 4, (
            f"Expected 4 test cases (2 methods × 2 iterations), got {len(runtime_by_test)}"
        )

        # Each test case should have 2 runtimes (2 outer loops)
        for test_id, test_runtimes in runtime_by_test.items():
            assert len(test_runtimes) == 2, (
                f"Expected 2 runtimes (2 outer loops) for {test_id.test_function_name}:{test_id.iteration_id}, "
                f"got {len(test_runtimes)}"
            )

        # Total should be sum of 4 minimums ≈ 40ms
        # Minimums filter out JIT warmup, so use tighter ±2% tolerance
        expected_total_ns = 4 * expected_ns  # 4 test cases × 10ms each
        assert expected_total_ns * 0.98 <= total_runtime <= expected_total_ns * 1.02, (
            f"total_passed_runtime {total_runtime / 1_000_000:.3f}ms not close to expected "
            f"{expected_total_ns / 1_000_000:.1f}ms (2 methods × 2 inner iterations × 10ms, ±2%)"
        )
