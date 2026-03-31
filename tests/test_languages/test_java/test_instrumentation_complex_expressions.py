"""Tests for Java instrumentation of calls inside complex expressions.

Verifies that target function calls inside cast, ternary, and binary expressions
are correctly instrumented in both performance and behavior modes, rather than
being silently skipped.
"""

import os
from pathlib import Path

os.environ["CODEFLASH_API_KEY"] = "cf-test-key"

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.java.instrumentation import instrument_existing_test


class TestComplexExpressionPerformanceMode:
    """Tests for calls inside complex expressions (cast, ternary, binary) in performance mode."""

    def test_cast_expression_performance_mode(self, tmp_path: Path):
        """A target call inside a cast expression must get timing instrumentation."""
        test_file = (tmp_path / "CastTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class CastTest {
    @Test
    public void testAddWithCast() {
        Calculator calc = new Calculator();
        long result = (long) calc.add(2, 2);
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="performance",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class CastTest__perfonlyinstrumented {\n'
            '    @Test\n'
            '    public void testAddWithCast() {\n'
            '        // Codeflash timing instrumentation with inner loop for JIT warmup\n'
            '        int _cf_outerLoop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_maxInnerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        String _cf_mod1 = "CastTest__perfonlyinstrumented";\n'
            '        String _cf_cls1 = "CastTest__perfonlyinstrumented";\n'
            '        String _cf_test1 = "testAddWithCast";\n'
            '        String _cf_fn1 = "add";\n'
            '\n'
            '        long result = 0L;        \n'
            '        Calculator calc = new Calculator();\n'
            '        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {\n'
            '            int _cf_loopId1 = _cf_outerLoop1 * _cf_maxInnerIterations1 + _cf_i1;\n'
            '            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + "######$!");\n'
            '            long _cf_end1 = -1;\n'
            '            long _cf_start1 = 0;\n'
            '            try {\n'
            '                _cf_start1 = System.nanoTime();\n'
            '                result = (long) calc.add(2, 2);\n'
            '                _cf_end1 = System.nanoTime();\n'
            '            } finally {\n'
            '                long _cf_end1_finally = System.nanoTime();\n'
            '                long _cf_dur1 = (_cf_end1 != -1 ? _cf_end1 : _cf_end1_finally) - _cf_start1;\n'
            '                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + ":" + _cf_dur1 + "######!");\n'
            '            }\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected

    def test_ternary_expression_performance_mode(self, tmp_path: Path):
        """A target call inside a ternary expression must get timing instrumentation."""
        test_file = (tmp_path / "TernaryTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class TernaryTest {
    @Test
    public void testAddWithTernary() {
        Calculator calc = new Calculator();
        boolean condition = true;
        int x = condition ? calc.add(1, 2) : 0;
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="performance",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class TernaryTest__perfonlyinstrumented {\n'
            '    @Test\n'
            '    public void testAddWithTernary() {\n'
            '        // Codeflash timing instrumentation with inner loop for JIT warmup\n'
            '        int _cf_outerLoop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_maxInnerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        String _cf_mod1 = "TernaryTest__perfonlyinstrumented";\n'
            '        String _cf_cls1 = "TernaryTest__perfonlyinstrumented";\n'
            '        String _cf_test1 = "testAddWithTernary";\n'
            '        String _cf_fn1 = "add";\n'
            '\n'
            '        int x = 0;        \n'
            '        Calculator calc = new Calculator();\n'
            '        boolean condition = true;\n'
            '        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {\n'
            '            int _cf_loopId1 = _cf_outerLoop1 * _cf_maxInnerIterations1 + _cf_i1;\n'
            '            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L9_1" + "######$!");\n'
            '            long _cf_end1 = -1;\n'
            '            long _cf_start1 = 0;\n'
            '            try {\n'
            '                _cf_start1 = System.nanoTime();\n'
            '                x = condition ? calc.add(1, 2) : 0;\n'
            '                _cf_end1 = System.nanoTime();\n'
            '            } finally {\n'
            '                long _cf_end1_finally = System.nanoTime();\n'
            '                long _cf_dur1 = (_cf_end1 != -1 ? _cf_end1 : _cf_end1_finally) - _cf_start1;\n'
            '                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L9_1" + ":" + _cf_dur1 + "######!");\n'
            '            }\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected

    def test_binary_expression_performance_mode(self, tmp_path: Path):
        """Target calls inside a binary expression must get timing instrumentation."""
        test_file = (tmp_path / "BinaryTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class BinaryTest {
    @Test
    public void testAddWithBinary() {
        Calculator calc = new Calculator();
        int x = calc.add(1, 2) + calc.add(3, 4);
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="performance",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class BinaryTest__perfonlyinstrumented {\n'
            '    @Test\n'
            '    public void testAddWithBinary() {\n'
            '        // Codeflash timing instrumentation with inner loop for JIT warmup\n'
            '        int _cf_outerLoop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_maxInnerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        String _cf_mod1 = "BinaryTest__perfonlyinstrumented";\n'
            '        String _cf_cls1 = "BinaryTest__perfonlyinstrumented";\n'
            '        String _cf_test1 = "testAddWithBinary";\n'
            '        String _cf_fn1 = "add";\n'
            '\n'
            '        int x = 0;        \n'
            '        Calculator calc = new Calculator();\n'
            '        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {\n'
            '            int _cf_loopId1 = _cf_outerLoop1 * _cf_maxInnerIterations1 + _cf_i1;\n'
            '            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + "######$!");\n'
            '            long _cf_end1 = -1;\n'
            '            long _cf_start1 = 0;\n'
            '            try {\n'
            '                _cf_start1 = System.nanoTime();\n'
            '                x = calc.add(1, 2) + calc.add(3, 4);\n'
            '                _cf_end1 = System.nanoTime();\n'
            '            } finally {\n'
            '                long _cf_end1_finally = System.nanoTime();\n'
            '                long _cf_dur1 = (_cf_end1 != -1 ? _cf_end1 : _cf_end1_finally) - _cf_start1;\n'
            '                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + ":" + _cf_dur1 + "######!");\n'
            '            }\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected

    def test_simple_call_for_reference(self, tmp_path: Path):
        """Reference: a simple direct call is instrumented correctly."""
        test_file = (tmp_path / "SimpleTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class SimpleTest {
    @Test
    public void testSimple() {
        Calculator calc = new Calculator();
        calc.add(2, 2);
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="performance",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class SimpleTest__perfonlyinstrumented {\n'
            '    @Test\n'
            '    public void testSimple() {\n'
            '        // Codeflash timing instrumentation with inner loop for JIT warmup\n'
            '        int _cf_outerLoop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_maxInnerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));\n'
            '        String _cf_mod1 = "SimpleTest__perfonlyinstrumented";\n'
            '        String _cf_cls1 = "SimpleTest__perfonlyinstrumented";\n'
            '        String _cf_test1 = "testSimple";\n'
            '        String _cf_fn1 = "add";\n'
            '        \n'
            '        Calculator calc = new Calculator();\n'
            '        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {\n'
            '            int _cf_loopId1 = _cf_outerLoop1 * _cf_maxInnerIterations1 + _cf_i1;\n'
            '            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + "######$!");\n'
            '            long _cf_end1 = -1;\n'
            '            long _cf_start1 = 0;\n'
            '            try {\n'
            '                _cf_start1 = System.nanoTime();\n'
            '                calc.add(2, 2);\n'
            '                _cf_end1 = System.nanoTime();\n'
            '            } finally {\n'
            '                long _cf_end1_finally = System.nanoTime();\n'
            '                long _cf_dur1 = (_cf_end1 != -1 ? _cf_end1 : _cf_end1_finally) - _cf_start1;\n'
            '                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loopId1 + ":" + "L8_1" + ":" + _cf_dur1 + "######!");\n'
            '            }\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected


class TestComplexExpressionBehaviorMode:
    """Tests for calls inside complex expressions (cast, ternary) in behavior mode."""

    def test_cast_expression_behavior_mode(self, tmp_path: Path):
        """A target call inside a cast expression must get behavior instrumentation."""
        test_file = (tmp_path / "CastTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class CastTest {
    @Test
    public void testAddWithCast() {
        Calculator calc = new Calculator();
        long result = (long) calc.add(2, 2);
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="behavior",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            'import java.sql.Connection;\n'
            'import java.sql.DriverManager;\n'
            'import java.sql.PreparedStatement;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class CastTest__perfinstrumented {\n'
            '    @Test\n'
            '    public void testAddWithCast() {\n'
            '        // Codeflash behavior instrumentation\n'
            '        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_iter1 = 1;\n'
            '        String _cf_mod1 = "CastTest__perfinstrumented";\n'
            '        String _cf_cls1 = "CastTest__perfinstrumented";\n'
            '        String _cf_fn1 = "add";\n'
            '        String _cf_outputFile1 = System.getenv("CODEFLASH_OUTPUT_FILE");\n'
            '        String _cf_testIteration1 = System.getenv("CODEFLASH_TEST_ITERATION");\n'
            '        if (_cf_testIteration1 == null) _cf_testIteration1 = "0";\n'
            '        String _cf_test1 = "testAddWithCast";\n'
            '        Calculator calc = new Calculator();\n'
            '        Object _cf_result1_1 = null;\n'
            '        long _cf_end1_1 = -1;\n'
            '        long _cf_start1_1 = 0;\n'
            '        byte[] _cf_serializedResult1_1 = null;\n'
            '        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":L11_1" + "######$!");\n'
            '        try {\n'
            '            _cf_start1_1 = System.nanoTime();\n'
            '            _cf_result1_1 = calc.add(2, 2);\n'
            '            _cf_end1_1 = System.nanoTime();\n'
            '            _cf_serializedResult1_1 = com.codeflash.Serializer.serialize((Object) _cf_result1_1);\n'
            '        } finally {\n'
            '            long _cf_end1_1_finally = System.nanoTime();\n'
            '            long _cf_dur1_1 = (_cf_end1_1 != -1 ? _cf_end1_1 : _cf_end1_1_finally) - _cf_start1_1;\n'
            '            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + "L11_1" + "######!");\n'
            '            // Write to SQLite if output file is set\n'
            '            if (_cf_outputFile1 != null && !_cf_outputFile1.isEmpty()) {\n'
            '                try {\n'
            '                    Class.forName("org.sqlite.JDBC");\n'
            '                    try (Connection _cf_conn1_1 = DriverManager.getConnection("jdbc:sqlite:" + _cf_outputFile1)) {\n'
            '                        try (java.sql.Statement _cf_stmt1_1 = _cf_conn1_1.createStatement()) {\n'
            '                            _cf_stmt1_1.execute("CREATE TABLE IF NOT EXISTS test_results (" +\n'
            '                                "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +\n'
            '                                "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +\n'
            '                                "runtime INTEGER, return_value BLOB, verification_type TEXT)");\n'
            '                        }\n'
            '                        String _cf_sql1_1 = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";\n'
            '                        try (PreparedStatement _cf_pstmt1_1 = _cf_conn1_1.prepareStatement(_cf_sql1_1)) {\n'
            '                            _cf_pstmt1_1.setString(1, _cf_mod1);\n'
            '                            _cf_pstmt1_1.setString(2, _cf_cls1);\n'
            '                            _cf_pstmt1_1.setString(3, _cf_test1);\n'
            '                            _cf_pstmt1_1.setString(4, _cf_fn1);\n'
            '                            _cf_pstmt1_1.setInt(5, _cf_loop1);\n'
            '                            _cf_pstmt1_1.setString(6, "L11_1");\n'
            '                            _cf_pstmt1_1.setLong(7, _cf_dur1_1);\n'
            '                            _cf_pstmt1_1.setBytes(8, _cf_serializedResult1_1);\n'
            '                            _cf_pstmt1_1.setString(9, "function_call");\n'
            '                            _cf_pstmt1_1.executeUpdate();\n'
            '                        }\n'
            '                    }\n'
            '                } catch (Exception _cf_e1_1) {\n'
            '                    System.err.println("CodeflashHelper: SQLite error: " + _cf_e1_1.getMessage());\n'
            '                }\n'
            '            }\n'
            '        }\n'
            '        long result = (long) (long)_cf_result1_1;\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected

    def test_ternary_expression_behavior_mode(self, tmp_path: Path):
        """A target call inside a ternary expression must get behavior instrumentation."""
        test_file = (tmp_path / "TernaryTest.java").resolve()
        source = """\
import org.junit.jupiter.api.Test;

public class TernaryTest {
    @Test
    public void testAddWithTernary() {
        Calculator calc = new Calculator();
        boolean condition = true;
        int x = condition ? calc.add(1, 2) : 0;
    }
}
"""
        test_file.write_text(source, encoding="utf-8")

        func = FunctionToOptimize(
            function_name="add",
            file_path=(tmp_path / "Calculator.java").resolve(),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_string=source,
            function_to_optimize=func,
            mode="behavior",
            test_path=test_file,
        )

        expected = (
            'import org.junit.jupiter.api.Test;\n'
            'import java.sql.Connection;\n'
            'import java.sql.DriverManager;\n'
            'import java.sql.PreparedStatement;\n'
            '\n'
            '@SuppressWarnings("CheckReturnValue")\n'
            'public class TernaryTest__perfinstrumented {\n'
            '    @Test\n'
            '    public void testAddWithTernary() {\n'
            '        // Codeflash behavior instrumentation\n'
            '        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));\n'
            '        int _cf_iter1 = 1;\n'
            '        String _cf_mod1 = "TernaryTest__perfinstrumented";\n'
            '        String _cf_cls1 = "TernaryTest__perfinstrumented";\n'
            '        String _cf_fn1 = "add";\n'
            '        String _cf_outputFile1 = System.getenv("CODEFLASH_OUTPUT_FILE");\n'
            '        String _cf_testIteration1 = System.getenv("CODEFLASH_TEST_ITERATION");\n'
            '        if (_cf_testIteration1 == null) _cf_testIteration1 = "0";\n'
            '        String _cf_test1 = "testAddWithTernary";\n'
            '        Calculator calc = new Calculator();\n'
            '        boolean condition = true;\n'
            '        Object _cf_result1_1 = null;\n'
            '        long _cf_end1_1 = -1;\n'
            '        long _cf_start1_1 = 0;\n'
            '        byte[] _cf_serializedResult1_1 = null;\n'
            '        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":L12_1" + "######$!");\n'
            '        try {\n'
            '            _cf_start1_1 = System.nanoTime();\n'
            '            _cf_result1_1 = calc.add(1, 2);\n'
            '            _cf_end1_1 = System.nanoTime();\n'
            '            _cf_serializedResult1_1 = com.codeflash.Serializer.serialize((Object) _cf_result1_1);\n'
            '        } finally {\n'
            '            long _cf_end1_1_finally = System.nanoTime();\n'
            '            long _cf_dur1_1 = (_cf_end1_1 != -1 ? _cf_end1_1 : _cf_end1_1_finally) - _cf_start1_1;\n'
            '            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + "." + _cf_test1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + "L12_1" + "######!");\n'
            '            // Write to SQLite if output file is set\n'
            '            if (_cf_outputFile1 != null && !_cf_outputFile1.isEmpty()) {\n'
            '                try {\n'
            '                    Class.forName("org.sqlite.JDBC");\n'
            '                    try (Connection _cf_conn1_1 = DriverManager.getConnection("jdbc:sqlite:" + _cf_outputFile1)) {\n'
            '                        try (java.sql.Statement _cf_stmt1_1 = _cf_conn1_1.createStatement()) {\n'
            '                            _cf_stmt1_1.execute("CREATE TABLE IF NOT EXISTS test_results (" +\n'
            '                                "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +\n'
            '                                "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +\n'
            '                                "runtime INTEGER, return_value BLOB, verification_type TEXT)");\n'
            '                        }\n'
            '                        String _cf_sql1_1 = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";\n'
            '                        try (PreparedStatement _cf_pstmt1_1 = _cf_conn1_1.prepareStatement(_cf_sql1_1)) {\n'
            '                            _cf_pstmt1_1.setString(1, _cf_mod1);\n'
            '                            _cf_pstmt1_1.setString(2, _cf_cls1);\n'
            '                            _cf_pstmt1_1.setString(3, _cf_test1);\n'
            '                            _cf_pstmt1_1.setString(4, _cf_fn1);\n'
            '                            _cf_pstmt1_1.setInt(5, _cf_loop1);\n'
            '                            _cf_pstmt1_1.setString(6, "L12_1");\n'
            '                            _cf_pstmt1_1.setLong(7, _cf_dur1_1);\n'
            '                            _cf_pstmt1_1.setBytes(8, _cf_serializedResult1_1);\n'
            '                            _cf_pstmt1_1.setString(9, "function_call");\n'
            '                            _cf_pstmt1_1.executeUpdate();\n'
            '                        }\n'
            '                    }\n'
            '                } catch (Exception _cf_e1_1) {\n'
            '                    System.err.println("CodeflashHelper: SQLite error: " + _cf_e1_1.getMessage());\n'
            '                }\n'
            '            }\n'
            '        }\n'
            '        int x = condition ? (int)_cf_result1_1 : 0;\n'
            '    }\n'
            '}\n'
        )
        assert success is True
        assert result == expected
