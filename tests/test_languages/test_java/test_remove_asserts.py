"""Tests for Java assertion removal transformer.

Tests the transform_java_assertions function with exact string equality assertions
to ensure assertions are correctly removed while preserving target function calls.

Covers:
- JUnit 4 assertions (org.junit.Assert.*)
- JUnit 5 assertions (org.junit.jupiter.api.Assertions.*)
- AssertJ fluent assertions (assertThat(...).isEqualTo(...))
- Hamcrest assertions (assertThat(actual, is(expected)))
- assertThrows / assertDoesNotThrow with lambdas
- Variable assignments from assertThrows
- Multiple target calls in a single assertion
- Assertions without target calls (should be removed)
- Nested assertions (assertAll)
- Edge cases: static calls, qualified calls, method chaining
"""

from codeflash.languages.java.remove_asserts import (
    JavaAssertTransformer,
    transform_java_assertions,
)


class TestJUnit4Assertions:
    """Tests for JUnit 4 style assertions (org.junit.Assert.*)."""

    def test_assertfalse_with_message(self):
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
        Object _cf_result1 = instance.get(0);
    }
}
"""
        result = transform_java_assertions(source, "get")
        assert result == expected

    def test_asserttrue_with_message(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet_SetBit_DetectedTrue() {
        assertTrue("Bit at index 67 should be detected as set", bs.get(67));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test
    public void testGet_SetBit_DetectedTrue() {
        Object _cf_result1 = bs.get(67);
    }
}
"""
        result = transform_java_assertions(source, "get")
        assert result == expected

    def test_assertequals_with_static_call(self):
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
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertequals_with_instance_call(self):
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
        Object _cf_result1 = calc.add(2, 2);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected

    def test_assertnull(self):
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
        Object _cf_result1 = parser.parse(null);
    }
}
"""
        result = transform_java_assertions(source, "parse")
        assert result == expected

    def test_assertnotnull(self):
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
        Object _cf_result1 = Fibonacci.fibonacciSequence(5);
    }
}
"""
        result = transform_java_assertions(source, "fibonacciSequence")
        assert result == expected

    def test_assertnotequals(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testSubtract() {
        assertNotEquals(0, calc.subtract(5, 3));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {
    @Test
    public void testSubtract() {
        Object _cf_result1 = calc.subtract(5, 3);
    }
}
"""
        result = transform_java_assertions(source, "subtract")
        assert result == expected

    def test_assertarrayequals(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacciSequence() {
        assertArrayEquals(new long[]{0, 1, 1, 2, 3}, Fibonacci.fibonacciSequence(5));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class FibonacciTest {
    @Test
    public void testFibonacciSequence() {
        Object _cf_result1 = Fibonacci.fibonacciSequence(5);
    }
}
"""
        result = transform_java_assertions(source, "fibonacciSequence")
        assert result == expected

    def test_qualified_assert_call(self):
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
        Object _cf_result1 = calc.add(2, 2);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected

    def test_expected_exception_annotation(self):
        source = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class BitSetTest {
    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testGet_NegativeIndex_Throws() {
        instance.get(-1);
    }
}
"""
        result = transform_java_assertions(source, "get")
        assert result == source


class TestJUnit5Assertions:
    """Tests for JUnit 5 style assertions (org.junit.jupiter.api.Assertions.*)."""

    def test_assertequals_static_import(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertEquals(1, Fibonacci.fibonacci(1));
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
        Object _cf_result3 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertequals_qualified(self):
        source = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Assertions.assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthrows_expression_lambda(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        try { Fibonacci.fibonacci(-1); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthrows_block_lambda(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            Fibonacci.fibonacci(-1);
        });
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        try { Fibonacci.fibonacci(-1); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthrows_assigned_to_variable(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        IllegalArgumentException ex = null;
        try { Fibonacci.fibonacci(-1); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertdoesnotthrow(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testDoesNotThrow() {
        assertDoesNotThrow(() -> Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testDoesNotThrow() {
        try { Fibonacci.fibonacci(10); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertsame(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CacheTest {
    @Test
    void testCacheSameInstance() {
        assertSame(expected, cache.get("key"));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CacheTest {
    @Test
    void testCacheSameInstance() {
        Object _cf_result1 = cache.get("key");
    }
}
"""
        result = transform_java_assertions(source, "get")
        assert result == expected

    def test_asserttrue_boolean_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIsFibonacci() {
        assertTrue(Fibonacci.isFibonacci(5));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIsFibonacci() {
        Object _cf_result1 = Fibonacci.isFibonacci(5);
    }
}
"""
        result = transform_java_assertions(source, "isFibonacci")
        assert result == expected

    def test_assertfalse_boolean_call(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIsNotFibonacci() {
        assertFalse(Fibonacci.isFibonacci(4));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIsNotFibonacci() {
        Object _cf_result1 = Fibonacci.isFibonacci(4);
    }
}
"""
        result = transform_java_assertions(source, "isFibonacci")
        assert result == expected


class TestAssertJFluent:
    """Tests for AssertJ fluent style assertions."""

    def test_assertthat_isequalto(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        assertThat(Fibonacci.fibonacci(10)).isEqualTo(55);
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthat_chained(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class ListTest {
    @Test
    void testGetItems() {
        assertThat(store.getItems()).isNotNull().hasSize(3).contains("apple");
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class ListTest {
    @Test
    void testGetItems() {
        Object _cf_result1 = store.getItems();
    }
}
"""
        result = transform_java_assertions(source, "getItems")
        assert result == expected

    def test_assertthat_isnull(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class ParserTest {
    @Test
    void testParseReturnsNull() {
        assertThat(parser.parse("invalid")).isNull();
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class ParserTest {
    @Test
    void testParseReturnsNull() {
        Object _cf_result1 = parser.parse("invalid");
    }
}
"""
        result = transform_java_assertions(source, "parse")
        assert result == expected

    def test_assertthat_qualified(self):
        source = """\
import org.junit.jupiter.api.Test;
import org.assertj.core.api.Assertions;

public class CalcTest {
    @Test
    void testAdd() {
        Assertions.assertThat(calc.add(1, 2)).isEqualTo(3);
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import org.assertj.core.api.Assertions;

public class CalcTest {
    @Test
    void testAdd() {
        Object _cf_result1 = calc.add(1, 2);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected


class TestHamcrestAssertions:
    """Tests for Hamcrest style assertions."""

    def test_hamcrest_assertthat_is(self):
        source = """\
import org.junit.Test;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assertThat(calc.add(2, 3), is(5));
    }
}
"""
        expected = """\
import org.junit.Test;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Object _cf_result1 = calc.add(2, 3);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected

    def test_hamcrest_qualified_assertthat(self):
        source = """\
import org.junit.Test;
import org.hamcrest.MatcherAssert;
import static org.hamcrest.Matchers.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        MatcherAssert.assertThat(calc.add(2, 3), equalTo(5));
    }
}
"""
        expected = """\
import org.junit.Test;
import org.hamcrest.MatcherAssert;
import static org.hamcrest.Matchers.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Object _cf_result1 = calc.add(2, 3);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected


class TestMultipleTargetCalls:
    """Tests for assertions containing multiple target function calls."""

    def test_multiple_calls_in_one_assertion(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testConsecutive() {
        assertTrue(Fibonacci.areConsecutiveFibonacci(Fibonacci.fibonacci(5), Fibonacci.fibonacci(6)));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testConsecutive() {
        Object _cf_result1 = Fibonacci.areConsecutiveFibonacci(Fibonacci.fibonacci(5), Fibonacci.fibonacci(6));
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_multiple_assertions_in_one_method(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testMultiple() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertEquals(1, Fibonacci.fibonacci(1));
        assertEquals(1, Fibonacci.fibonacci(2));
        assertEquals(2, Fibonacci.fibonacci(3));
        assertEquals(5, Fibonacci.fibonacci(5));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testMultiple() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
        Object _cf_result3 = Fibonacci.fibonacci(2);
        Object _cf_result4 = Fibonacci.fibonacci(3);
        Object _cf_result5 = Fibonacci.fibonacci(5);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestNoTargetCalls:
    """Tests for assertions that do NOT contain calls to the target function."""

    def test_assertion_without_target_removed(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SetupTest {
    @Test
    void testSetup() {
        assertNotNull(config);
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SetupTest {
    @Test
    void testSetup() {
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_no_assertions_at_all(self):
        source = """\
import org.junit.jupiter.api.Test;

public class FibonacciTest {
    @Test
    void testPrint() {
        System.out.println(Fibonacci.fibonacci(10));
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == source


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_source(self):
        result = transform_java_assertions("", "fibonacci")
        assert result == ""

    def test_whitespace_only_source(self):
        result = transform_java_assertions("   \n\n  ", "fibonacci")
        assert result == "   \n\n  "

    def test_multiline_assertion(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        assertEquals(
            55,
            Fibonacci.fibonacci(10)
        );
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertion_with_string_containing_parens(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ParserTest {
    @Test
    void testParse() {
        assertEquals("result(1)", parser.parse("input(1)"));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ParserTest {
    @Test
    void testParse() {
        Object _cf_result1 = parser.parse("input(1)");
    }
}
"""
        result = transform_java_assertions(source, "parse")
        assert result == expected

    def test_preserves_non_test_code(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testSequence() {
        int n = 10;
        long[] expected = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
        assertArrayEquals(expected, Fibonacci.fibonacciSequence(n));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testSequence() {
        int n = 10;
        long[] expected = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
        Object _cf_result1 = Fibonacci.fibonacciSequence(n);
    }
}
"""
        result = transform_java_assertions(source, "fibonacciSequence")
        assert result == expected

    def test_nested_method_calls(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIndex() {
        assertEquals(10, Fibonacci.fibonacciIndex(Fibonacci.fibonacci(10)));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testIndex() {
        Object _cf_result1 = Fibonacci.fibonacciIndex(Fibonacci.fibonacci(10));
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_chained_method_on_result(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testUpTo() {
        assertEquals(7, Fibonacci.fibonacciUpTo(20).size());
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testUpTo() {
        Object _cf_result1 = Fibonacci.fibonacciUpTo(20);
    }
}
"""
        result = transform_java_assertions(source, "fibonacciUpTo")
        assert result == expected


class TestBitSetLikeQuestDB:
    """Tests modeled after the QuestDB BitSetTest pattern shown by the user.

    This covers the real-world scenario of JUnit 4 tests with message strings,
    reflection-based setup, expected exceptions, and multiple assertion types.
    """

    BITSET_TEST_SOURCE = """\
package io.questdb.std;

import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Field;

import static org.junit.Assert.*;

public class BitSetTest {
    private BitSet instance;

    @Before
    public void setUp() {
        instance = new BitSet();
    }

    @Test
    public void testGet_IndexZero_ReturnsFalse() {
        assertFalse("New BitSet should have bit 0 unset", instance.get(0));
    }

    @Test
    public void testGet_SpecificIndexWithinRange_ReturnsFalse() {
        assertFalse("New BitSet should have bit 100 unset", instance.get(100));
    }

    @Test
    public void testGet_LastIndexOfInitialRange_ReturnsFalse() {
        int lastIndex = 16 * BitSet.BITS_PER_WORD - 1;
        assertFalse("Last index of initial range should be unset", instance.get(lastIndex));
    }

    @Test
    public void testGet_IndexBeyondAllocated_ReturnsFalse() {
        int beyond = 16 * BitSet.BITS_PER_WORD;
        assertFalse("Index beyond allocated range should return false", instance.get(beyond));
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testGet_NegativeIndex_ThrowsArrayIndexOutOfBoundsException() {
        instance.get(-1);
    }

    @Test
    public void testGet_SetWordUsingReflection_DetectedTrue() throws Exception {
        BitSet bs = new BitSet(128);
        Field wordsField = BitSet.class.getDeclaredField("words");
        wordsField.setAccessible(true);
        long[] words = new long[2];
        words[1] = 1L << 3;
        wordsField.set(bs, words);
        assertTrue("Bit at index 67 should be detected as set", bs.get(64 + 3));
    }

    @Test
    public void testGet_LargeIndexDoesNotThrow_ReturnsFalse() {
        assertFalse("Very large index should return false without throwing", instance.get(Integer.MAX_VALUE));
    }

    @Test
    public void testGet_BitBoundaryWordEdge63_ReturnsFalse() {
        assertFalse("Bit index 63 (end of first word) should be unset by default", instance.get(63));
    }

    @Test
    public void testGet_BitBoundaryWordEdge64_ReturnsFalse() {
        assertFalse("Bit index 64 (start of second word) should be unset by default", instance.get(64));
    }

    @Test
    public void testGet_LargeBitSetLastIndex_ReturnsFalse() {
        int nBits = 1_000_000;
        BitSet big = new BitSet(nBits);
        int last = nBits - 1;
        assertFalse("Last bit of a large BitSet should be unset by default", big.get(last));
    }
}
"""

    EXPECTED = """\
package io.questdb.std;

import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Field;

import static org.junit.Assert.*;

public class BitSetTest {
    private BitSet instance;

    @Before
    public void setUp() {
        instance = new BitSet();
    }

    @Test
    public void testGet_IndexZero_ReturnsFalse() {
        Object _cf_result1 = instance.get(0);
    }

    @Test
    public void testGet_SpecificIndexWithinRange_ReturnsFalse() {
        Object _cf_result2 = instance.get(100);
    }

    @Test
    public void testGet_LastIndexOfInitialRange_ReturnsFalse() {
        int lastIndex = 16 * BitSet.BITS_PER_WORD - 1;
        Object _cf_result3 = instance.get(lastIndex);
    }

    @Test
    public void testGet_IndexBeyondAllocated_ReturnsFalse() {
        int beyond = 16 * BitSet.BITS_PER_WORD;
        Object _cf_result4 = instance.get(beyond);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testGet_NegativeIndex_ThrowsArrayIndexOutOfBoundsException() {
        instance.get(-1);
    }

    @Test
    public void testGet_SetWordUsingReflection_DetectedTrue() throws Exception {
        BitSet bs = new BitSet(128);
        Field wordsField = BitSet.class.getDeclaredField("words");
        wordsField.setAccessible(true);
        long[] words = new long[2];
        words[1] = 1L << 3;
        wordsField.set(bs, words);
        Object _cf_result5 = bs.get(64 + 3);
    }

    @Test
    public void testGet_LargeIndexDoesNotThrow_ReturnsFalse() {
        Object _cf_result6 = instance.get(Integer.MAX_VALUE);
    }

    @Test
    public void testGet_BitBoundaryWordEdge63_ReturnsFalse() {
        Object _cf_result7 = instance.get(63);
    }

    @Test
    public void testGet_BitBoundaryWordEdge64_ReturnsFalse() {
        Object _cf_result8 = instance.get(64);
    }

    @Test
    public void testGet_LargeBitSetLastIndex_ReturnsFalse() {
        int nBits = 1_000_000;
        BitSet big = new BitSet(nBits);
        int last = nBits - 1;
        Object _cf_result9 = big.get(last);
    }
}
"""

    def test_all_assertfalse_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_asserttrue_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_setup_code_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_reflection_code_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_expected_exception_test_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_package_and_imports_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_class_structure_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_large_index_assertions_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED

    def test_no_assertfalse_remain(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert result == self.EXPECTED


class TestTransformMethod:
    """Tests for JavaAssertTransformer.transform() -- each branch and code path."""

    # --- Early returns ---

    def test_none_source_returns_unchanged(self):
        transformer = JavaAssertTransformer("fibonacci")
        assert transformer.transform("") == ""

    def test_whitespace_only_returns_unchanged(self):
        transformer = JavaAssertTransformer("fibonacci")
        ws = "   \n\t\n  "
        assert transformer.transform(ws) == ws

    def test_no_assertions_found_returns_unchanged(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void test1() {
        long result = Fibonacci.fibonacci(10);
        System.out.println(result);
    }
}
"""
        result = transformer.transform(source)
        assert result == source
        assert transformer.invocation_counter == 0

    def test_assertions_exist_but_no_target_calls_are_removed(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test1() {
        assertEquals(4, calculator.add(2, 2));
        assertTrue(validator.isValid("x"));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test1() {
    }
}
"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 0

    # --- Counter numbering in source order ---

    def test_counters_assigned_in_source_order(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void testA() {
        assertEquals(0, Fibonacci.fibonacci(0));
    }
    @Test
    void testB() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
    @Test
    void testC() {
        assertEquals(1, Fibonacci.fibonacci(1));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void testA() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
    }
    @Test
    void testB() {
        Object _cf_result2 = Fibonacci.fibonacci(10);
    }
    @Test
    void testC() {
        Object _cf_result3 = Fibonacci.fibonacci(1);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 3

    def test_counter_increments_across_transform_call(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertEquals(1, Fibonacci.fibonacci(1));
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        transformer.transform(source)
        assert transformer.invocation_counter == 3

    # --- Nested assertion filtering ---

    def test_nested_assertions_inside_assertall_only_outer_replaced(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertAll(
            () -> assertEquals(0, Fibonacci.fibonacci(0)),
            () -> assertEquals(1, Fibonacci.fibonacci(1))
        );
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected

    def test_non_nested_assertions_all_replaced(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertTrue(Fibonacci.isFibonacci(5));
        assertFalse(Fibonacci.isFibonacci(4));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected

    # --- Reverse replacement preserves positions ---

    def test_reverse_replacement_preserves_all_positions(self):
        transformer = JavaAssertTransformer("compute")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void test() {
        assertEquals(1, engine.compute(1));
        assertEquals(4, engine.compute(2));
        assertEquals(9, engine.compute(3));
        assertEquals(16, engine.compute(4));
        assertEquals(25, engine.compute(5));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void test() {
        Object _cf_result1 = engine.compute(1);
        Object _cf_result2 = engine.compute(2);
        Object _cf_result3 = engine.compute(3);
        Object _cf_result4 = engine.compute(4);
        Object _cf_result5 = engine.compute(5);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 5

    # --- Mixed assertions: some with target, some without ---

    def test_mixed_assertions_all_removed(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertNotNull(config);
        assertEquals(0, Fibonacci.fibonacci(0));
        assertTrue(isReady);
        assertEquals(1, Fibonacci.fibonacci(1));
        assertFalse(isDone);
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 2

    # --- Exception assertions in transform ---

    def test_exception_assertion_without_target_calls_still_replaced(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertThrows(Exception.class, () -> thrower.doSomething());
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        try { thrower.doSomething(); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transformer.transform(source)
        assert result == expected

    # --- Full output exact equality ---

    def test_single_assertion_exact_output(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        result = transformer.transform(source)
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        Object _cf_result1 = Fibonacci.fibonacci(10);
    }
}
"""
        assert result == expected

    def test_multiple_assertions_exact_output(self):
        transformer = JavaAssertTransformer("add")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void test() {
        assertEquals(3, calc.add(1, 2));
        assertEquals(7, calc.add(3, 4));
    }
}
"""
        result = transformer.transform(source)
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {
    @Test
    void test() {
        Object _cf_result1 = calc.add(1, 2);
        Object _cf_result2 = calc.add(3, 4);
    }
}
"""
        assert result == expected

    # --- Idempotency ---

    def test_transform_already_transformed_is_noop(self):
        transformer1 = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        first_pass = transformer1.transform(source)
        transformer2 = JavaAssertTransformer("fibonacci")
        second_pass = transformer2.transform(first_pass)
        assert second_pass == first_pass
        assert transformer2.invocation_counter == 0


class TestJavaAssertTransformerClass:
    """Tests for the JavaAssertTransformer class directly."""

    def test_invocation_counter_increments(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test1() {
        assertEquals(0, Fibonacci.fibonacci(0));
    }

    @Test
    void test2() {
        assertEquals(55, Fibonacci.fibonacci(10));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibTest {
    @Test
    void test1() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
    }

    @Test
    void test2() {
        Object _cf_result2 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 2

    def test_framework_detection_junit5(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.*;\n"
        framework = transformer._detect_framework(source)
        assert framework == "junit5"

    def test_framework_detection_junit4(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "import org.junit.Test;\nimport static org.junit.Assert.*;\n"
        framework = transformer._detect_framework(source)
        assert framework == "junit4"

    def test_framework_detection_assertj(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "import org.assertj.core.api.Assertions;\n"
        framework = transformer._detect_framework(source)
        assert framework == "assertj"

    def test_framework_detection_hamcrest(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "import org.hamcrest.MatcherAssert;\nimport org.hamcrest.Matchers;\n"
        framework = transformer._detect_framework(source)
        assert framework == "hamcrest"

    def test_framework_detection_testng(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "import org.testng.Assert;\n"
        framework = transformer._detect_framework(source)
        assert framework == "testng"

    def test_framework_detection_default_junit5(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = "public class Test {}"
        framework = transformer._detect_framework(source)
        assert framework == "junit5"


class TestAssertAll:
    """Tests for assertAll (JUnit 5 grouped assertions)."""

    def test_assertall_with_target_calls(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testMultipleFibonacci() {
        assertAll(
            () -> assertEquals(0, Fibonacci.fibonacci(0)),
            () -> assertEquals(1, Fibonacci.fibonacci(1)),
            () -> assertEquals(55, Fibonacci.fibonacci(10))
        );
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testMultipleFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
        Object _cf_result3 = Fibonacci.fibonacci(10);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestAssertThrowsEdgeCases:
    """Edge cases for assertThrows transformation."""

    def test_assertthrows_with_multiline_lambda(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        assertThrows(
            IllegalArgumentException.class,
            () -> Fibonacci.fibonacci(-1)
        );
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        try { Fibonacci.fibonacci(-1); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthrows_with_complex_lambda_body(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            int n = -5;
            Fibonacci.fibonacci(n);
        });
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        try { int n = -5;
            Fibonacci.fibonacci(n); } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertthrows_with_final_variable(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        final IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeThrows() {
        IllegalArgumentException ex = null;
        try { Fibonacci.fibonacci(-1); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; } catch (Exception _cf_ignored1) {}
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestAllAssertionsRemoved:
    """Tests verifying that ALL assertions are removed (the default behavior)."""

    MULTI_FUNCTION_TEST = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {

    @Test
    void testFibonacci() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertEquals(1, Fibonacci.fibonacci(1));
        assertEquals(5, Fibonacci.fibonacci(5));
    }

    @Test
    void testIsFibonacci() {
        assertTrue(Fibonacci.isFibonacci(0));
        assertTrue(Fibonacci.isFibonacci(1));
        assertFalse(Fibonacci.isFibonacci(4));
    }

    @Test
    void testIsPerfectSquare() {
        assertTrue(Fibonacci.isPerfectSquare(0));
        assertTrue(Fibonacci.isPerfectSquare(4));
        assertFalse(Fibonacci.isPerfectSquare(5));
    }

    @Test
    void testFibonacciSequence() {
        assertArrayEquals(new long[]{0, 1, 1}, Fibonacci.fibonacciSequence(3));
    }

    @Test
    void testFibonacciIndex() {
        assertEquals(0, Fibonacci.fibonacciIndex(0));
        assertEquals(5, Fibonacci.fibonacciIndex(5));
    }

    @Test
    void testSumFibonacci() {
        assertEquals(0, Fibonacci.sumFibonacci(0));
        assertEquals(4, Fibonacci.sumFibonacci(4));
    }

    @Test
    void testFibonacciNegative() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }
}
"""

    EXPECTED = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {

    @Test
    void testFibonacci() {
        Object _cf_result1 = Fibonacci.fibonacci(0);
        Object _cf_result2 = Fibonacci.fibonacci(1);
        Object _cf_result3 = Fibonacci.fibonacci(5);
    }

    @Test
    void testIsFibonacci() {
    }

    @Test
    void testIsPerfectSquare() {
    }

    @Test
    void testFibonacciSequence() {
    }

    @Test
    void testFibonacciIndex() {
    }

    @Test
    void testSumFibonacci() {
    }

    @Test
    void testFibonacciNegative() {
        try { Fibonacci.fibonacci(-1); } catch (Exception _cf_ignored4) {}
    }
}
"""

    def test_all_assertions_removed(self):
        result = transform_java_assertions(self.MULTI_FUNCTION_TEST, "fibonacci")
        assert result == self.EXPECTED

    def test_preserves_non_assertion_code(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalcTest {

    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        int result = calc.setup();
        assertEquals(5, calc.add(2, 3));
        assertTrue(calc.isReady());
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
        int result = calc.setup();
        Object _cf_result1 = calc.add(2, 3);
    }
}
"""
        result = transform_java_assertions(source, "add")
        assert result == expected

    def test_assertj_all_removed(self):
        source = """\
import org.assertj.core.api.Assertions;
import static org.assertj.core.api.Assertions.assertThat;

public class FibTest {
    @Test
    void test() {
        assertThat(Fibonacci.fibonacci(5)).isEqualTo(5);
        assertThat(Fibonacci.isFibonacci(5)).isTrue();
    }
}
"""
        expected = """\
import org.assertj.core.api.Assertions;
import static org.assertj.core.api.Assertions.assertThat;

public class FibTest {
    @Test
    void test() {
        Object _cf_result1 = Fibonacci.fibonacci(5);
    }
}
"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_mixed_frameworks_all_removed(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MixedTest {
    @Test
    void test() {
        assertEquals(5, obj.target(1));
        assertNull(obj.other());
        assertNotNull(obj.another());
        assertTrue(obj.check());
    }
}
"""
        expected = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MixedTest {
    @Test
    void test() {
        Object _cf_result1 = obj.target(1);
    }
}
"""
        result = transform_java_assertions(source, "target")
        assert result == expected
