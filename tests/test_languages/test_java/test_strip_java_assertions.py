"""Tests for strip mode assertion removal in JavaAssertTransformer.

strip_java_assertions() produces clean output for PR display:
- Assertions with target function calls → bare `call;` statements (no capture variables)
- Assertions without target function calls → removed entirely
- Exception assertions → simple try/catch without numbered variables
"""

from codeflash.languages.java.remove_asserts import strip_java_assertions


class TestStripJUnit4Assertions:
    """Strip mode with JUnit 4 style assertions."""

    def test_assertequals_static_call_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacci() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacci() {
        Fibonacci.fibonacci(10);
    }
}
"""
        assert strip_java_assertions(source, "fibonacci") == expected

    def test_assertequals_instance_call_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        calc.add(2, 2);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_asserttrue_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet() {
        assertTrue(bs.get(67));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet() {
        bs.get(67);
    }
}
"""
        assert strip_java_assertions(source, "get") == expected

    def test_assertfalse_with_message_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet_IndexZero_ReturnsFalse() {
        assertFalse("New BitSet should have bit 0 unset", instance.get(0));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet_IndexZero_ReturnsFalse() {
        instance.get(0);
    }
}
"""
        assert strip_java_assertions(source, "get") == expected

    def test_assertnull_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class ParserTest {
    @Test
    public void testParseNull() {
        assertNull(parser.parse(null));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class ParserTest {
    @Test
    public void testParseNull() {
        parser.parse(null);
    }
}
"""
        assert strip_java_assertions(source, "parse") == expected

    def test_assertnotnull_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacciSequence() {
        assertNotNull(Fibonacci.fibonacciSequence(5));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacciSequence() {
        Fibonacci.fibonacciSequence(5);
    }
}
"""
        assert strip_java_assertions(source, "fibonacciSequence") == expected

    def test_qualified_assert_becomes_bare_call(self):
        source = """\
import org.junit.Test;
import org.junit.Assert;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Assert.assertEquals(4, calc.add(2, 2));
    }
}
"""
        expected = """\
import org.junit.Test;
import org.junit.Assert;

public class CalculatorTest {
    @Test
    public void testAdd() {
        calc.add(2, 2);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_assertion_without_target_call_removed(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FooTest {
    @Test
    public void testSomething() {
        int x = compute();
        assertEquals(42, x);
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FooTest {
    @Test
    public void testSomething() {
        int x = compute();
    }
}
"""
        assert strip_java_assertions(source, "compute") == expected

    def test_multiple_assertions_mixed_presence_of_target_calls(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testMultiple() {
        assertEquals(4, calc.add(2, 2));
        assertEquals(0, calc.add(0, 0));
        assertEquals(42, someOtherValue);
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testMultiple() {
        calc.add(2, 2);
        calc.add(0, 0);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected


class TestStripJUnit5Assertions:
    """Strip mode with JUnit 5 style assertions."""

    def test_junit5_assertequals_becomes_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        assertEquals(55L, fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        fibonacci(10);
    }
}
"""
        assert strip_java_assertions(source, "fibonacci") == expected

    def test_junit5_qualified_assertions_becomes_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class CalculatorTest {
    @Test
    void testAdd() {
        Assertions.assertEquals(10, calc.add(4, 6));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class CalculatorTest {
    @Test
    void testAdd() {
        calc.add(4, 6);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_junit5_assertarrayequals_becomes_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SorterTest {
    @Test
    void testSort() {
        assertArrayEquals(new int[]{1, 2, 3}, sorter.sort(new int[]{3, 1, 2}));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SorterTest {
    @Test
    void testSort() {
        sorter.sort(new int[]{3, 1, 2});
    }
}
"""
        assert strip_java_assertions(source, "sort") == expected


class TestStripAssertJAssertions:
    """Strip mode with AssertJ fluent assertions."""

    def test_assertj_isequalto_becomes_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testAdd() {
        assertThat(calc.add(2, 3)).isEqualTo(5);
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testAdd() {
        calc.add(2, 3);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_assertj_isnull_becomes_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class ParserTest {
    @Test
    void testParseNull() {
        assertThat(parser.parse(null)).isNull();
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class ParserTest {
    @Test
    void testParseNull() {
        parser.parse(null);
    }
}
"""
        assert strip_java_assertions(source, "parse") == expected

    def test_assertj_chained_assertions_become_bare_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class StringUtilsTest {
    @Test
    void testProcess() {
        assertThat(utils.process("hello")).isNotNull().isNotEmpty().isEqualTo("HELLO");
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class StringUtilsTest {
    @Test
    void testProcess() {
        utils.process("hello");
    }
}
"""
        assert strip_java_assertions(source, "process") == expected

    def test_assertj_without_target_call_removed(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class FooTest {
    @Test
    void testSomething() {
        int result = compute();
        assertThat(result).isEqualTo(42);
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class FooTest {
    @Test
    void testSomething() {
        int result = compute();
    }
}
"""
        assert strip_java_assertions(source, "compute") == expected


class TestStripExceptionAssertions:
    """Strip mode for assertThrows / assertDoesNotThrow."""

    def test_assertthrows_becomes_simple_try_catch(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testDivideByZero() {
        assertThrows(ArithmeticException.class, () -> calculator.divide(1, 0));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testDivideByZero() {
        try { calculator.divide(1, 0); } catch (ArithmeticException ignored) {}
    }
}
"""
        assert strip_java_assertions(source, "divide") == expected

    def test_assertthrows_no_numbered_variables(self):
        """Strip mode must not emit _cf_ignored1, _cf_caught1, etc."""
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
    @Test
    void testThrows() {
        assertThrows(IllegalArgumentException.class, () -> foo.bar(-1));
        assertThrows(NullPointerException.class, () -> foo.bar(null));
    }
}
"""
        result = strip_java_assertions(source, "bar")
        assert "_cf_ignored" not in result
        assert "_cf_caught" not in result

    def test_assertthrows_with_assigned_variable_becomes_simple_try_catch(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ValidatorTest {
    @Test
    void testException() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> validator.validate(-1));
    }
}
"""
        result = strip_java_assertions(source, "validate")
        assert "_cf_caught" not in result
        assert "_cf_ignored" not in result
        assert "validator.validate(-1)" in result

    def test_multiple_assertthrows_no_numbered_variables(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
    @Test
    void testMultiple() {
        assertThrows(ArithmeticException.class, () -> calc.divide(1, 0));
        assertThrows(ArithmeticException.class, () -> calc.divide(2, 0));
        assertThrows(ArithmeticException.class, () -> calc.divide(3, 0));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
    @Test
    void testMultiple() {
        try { calc.divide(1, 0); } catch (ArithmeticException ignored) {}
        try { calc.divide(2, 0); } catch (ArithmeticException ignored) {}
        try { calc.divide(3, 0); } catch (ArithmeticException ignored) {}
    }
}
"""
        assert strip_java_assertions(source, "divide") == expected


class TestStripNoCaptureVariables:
    """Verify no _cf_result capture variables appear anywhere in strip mode output."""

    def test_no_cf_result_variables(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void testFib() {
        assertEquals(1, fib(1));
        assertEquals(1, fib(2));
        assertEquals(2, fib(3));
        assertEquals(3, fib(4));
        assertEquals(5, fib(5));
    }
}
"""
        result = strip_java_assertions(source, "fib")
        assert "_cf_result" not in result

    def test_no_cf_result_with_assertj(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class UtilsTest {
    @Test
    void testProcess() {
        assertThat(utils.process("a")).isEqualTo("A");
        assertThat(utils.process("b")).isEqualTo("B");
    }
}
"""
        result = strip_java_assertions(source, "process")
        assert "_cf_result" not in result

    def test_no_instrumentation_artifacts_at_all(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        assertEquals(3, calc.add(1, 2));
    }
    @Test
    void testThrows() {
        assertThrows(Exception.class, () -> calc.add(-1, -1));
    }
}
"""
        result = strip_java_assertions(source, "add")
        assert "_cf_result" not in result
        assert "_cf_ignored" not in result
        assert "_cf_caught" not in result
        assert "__perfinstrumented" not in result


class TestStripPreservesNonAssertionCode:
    """Verify non-assertion code is untouched in strip mode."""

    def test_setup_code_preserved(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        int expected = 4;
        assertEquals(expected, calc.add(2, 2));
        System.out.println("done");
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        int expected = 4;
        calc.add(2, 2);
        System.out.println("done");
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_package_and_imports_preserved(self):
        source = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

public class SorterTest {
    @Test
    void testSort() {
        assertEquals(List.of(1, 2, 3), sorter.sort(List.of(3, 1, 2)));
    }
}
"""
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

public class SorterTest {
    @Test
    void testSort() {
        sorter.sort(List.of(3, 1, 2));
    }
}
"""
        assert strip_java_assertions(source, "sort") == expected

    def test_no_assertions_unchanged(self):
        source = """\
import org.junit.jupiter.api.Test;

public class CalcTest {
    @Test
    void testAdd() {
        int result = calc.add(1, 2);
    }
}
"""
        assert strip_java_assertions(source, "add") == source

    def test_empty_source_unchanged(self):
        assert strip_java_assertions("", "add") == ""

    def test_whitespace_only_unchanged(self):
        assert strip_java_assertions("   \n  ", "add") == "   \n  "


class TestStripVsCaptureMode:
    """Verify strip mode output differs from capture mode in the expected ways."""

    def test_strip_has_no_type_annotation(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibTest {
    @Test
    public void testFib() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        strip_result = strip_java_assertions(source, "fibonacci")
        expected_strip = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibTest {
    @Test
    public void testFib() {
        Fibonacci.fibonacci(10);
    }
}
"""
        assert strip_result == expected_strip
        # Capture mode would have: int _cf_result1 = Fibonacci.fibonacci(10);
        assert "int" not in strip_result
        assert "_cf_result" not in strip_result

    def test_strip_multiple_calls_no_counters(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalcTest {
    @Test
    public void testOps() {
        assertEquals(3, calc.add(1, 2));
        assertEquals(5, calc.add(3, 2));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalcTest {
    @Test
    public void testOps() {
        calc.add(1, 2);
        calc.add(3, 2);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_strip_exception_uses_fixed_name_not_counter(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DivTest {
    @Test
    void testDivide() {
        assertThrows(ArithmeticException.class, () -> calc.divide(5, 0));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DivTest {
    @Test
    void testDivide() {
        try { calc.divide(5, 0); } catch (ArithmeticException ignored) {}
    }
}
"""
        assert strip_java_assertions(source, "divide") == expected


class TestStripMultipleTestMethods:
    """Strip mode across multiple test methods in one class."""

    def test_multiple_test_methods_each_stripped_independently(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        assertEquals(3, calc.add(1, 2));
    }

    @Test
    void testAddNegative() {
        assertEquals(-1, calc.add(-3, 2));
    }

    @Test
    void testAddZero() {
        assertEquals(0, calc.add(0, 0));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        calc.add(1, 2);
    }

    @Test
    void testAddNegative() {
        calc.add(-3, 2);
    }

    @Test
    void testAddZero() {
        calc.add(0, 0);
    }
}
"""
        assert strip_java_assertions(source, "add") == expected

    def test_mixed_target_and_nontarget_calls_across_methods(self):
        # When the target call is nested inside another function call (e.g. isPositive(calc.add(1,2))),
        # the transformer preserves the entire top-level argument expression, not just the inner call.
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        assertEquals(3, calc.add(1, 2));
        assertTrue(isPositive(calc.add(1, 2)));
    }

    @Test
    void testUnrelated() {
        assertEquals("hello", someString());
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void testAdd() {
        calc.add(1, 2);
        isPositive(calc.add(1, 2));
    }

    @Test
    void testUnrelated() {
    }
}
"""
        assert strip_java_assertions(source, "add") == expected
