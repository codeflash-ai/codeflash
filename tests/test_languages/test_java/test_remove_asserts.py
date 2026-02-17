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
        result = transform_java_assertions(source, "get")
        assert 'assertFalse("New BitSet should have bit 0 unset", instance.get(0));' not in result
        assert "Object _cf_result1 = instance.get(0);" in result

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
        result = transform_java_assertions(source, "get")
        assert 'assertTrue("Bit at index 67 should be detected as set", bs.get(67));' not in result
        assert "Object _cf_result1 = bs.get(67);" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertEquals(55, Fibonacci.fibonacci(10));" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(10);" in result

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
        result = transform_java_assertions(source, "add")
        assert "assertEquals(4, calc.add(2, 2));" not in result
        assert "Object _cf_result1 = calc.add(2, 2);" in result
        # Non-assertion code should be preserved
        assert "Calculator calc = new Calculator();" in result

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
        result = transform_java_assertions(source, "parse")
        assert "assertNull(parser.parse(null));" not in result
        assert "Object _cf_result1 = parser.parse(null);" in result

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
        result = transform_java_assertions(source, "fibonacciSequence")
        assert "assertNotNull(Fibonacci.fibonacciSequence(5));" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacciSequence(5);" in result

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
        result = transform_java_assertions(source, "subtract")
        assert "assertNotEquals(0, calc.subtract(5, 3));" not in result
        assert "Object _cf_result1 = calc.subtract(5, 3);" in result

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
        result = transform_java_assertions(source, "fibonacciSequence")
        assert "assertArrayEquals(new long[]{0, 1, 1, 2, 3}, Fibonacci.fibonacciSequence(5));" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacciSequence(5);" in result

    def test_qualified_assert_call(self):
        """Test Assert.assertEquals (JUnit 4 qualified)."""
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
        result = transform_java_assertions(source, "add")
        assert "Assert.assertEquals(4, calc.add(2, 2));" not in result
        assert "Object _cf_result1 = calc.add(2, 2);" in result

    def test_expected_exception_annotation(self):
        """Test that @Test(expected=...) tests with target calls are handled."""
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
        # No assertions to remove here, but the call should remain
        result = transform_java_assertions(source, "get")
        assert "instance.get(-1);" in result


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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertEquals" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(0);" in result
        assert "Object _cf_result2 = Fibonacci.fibonacci(1);" in result
        assert "Object _cf_result3 = Fibonacci.fibonacci(10);" in result

    def test_assertequals_qualified(self):
        """Test Assertions.assertEquals (JUnit 5 qualified)."""
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
        result = transform_java_assertions(source, "fibonacci")
        assert "Assertions.assertEquals(55, Fibonacci.fibonacci(10));" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(10);" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "try {" in result
        assert "Fibonacci.fibonacci(-1);" in result
        assert "catch (Exception" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "try {" in result
        assert "Fibonacci.fibonacci(-1);" in result
        assert "catch (Exception" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "IllegalArgumentException ex = null;" in result
        assert "Fibonacci.fibonacci(-1);" in result
        assert "_cf_caught" in result
        assert "ex = _cf_caught" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertDoesNotThrow" not in result
        assert "try {" in result
        assert "Fibonacci.fibonacci(10);" in result

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
        result = transform_java_assertions(source, "get")
        assert "assertSame" not in result
        assert 'Object _cf_result1 = cache.get("key");' in result

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
        result = transform_java_assertions(source, "isFibonacci")
        assert "assertTrue(Fibonacci.isFibonacci(5));" not in result
        assert "Object _cf_result1 = Fibonacci.isFibonacci(5);" in result

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
        result = transform_java_assertions(source, "isFibonacci")
        assert "assertFalse(Fibonacci.isFibonacci(4));" not in result
        assert "Object _cf_result1 = Fibonacci.isFibonacci(4);" in result


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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThat(Fibonacci.fibonacci(10)).isEqualTo(55);" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(10);" in result

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
        result = transform_java_assertions(source, "getItems")
        assert 'assertThat(store.getItems()).isNotNull().hasSize(3).contains("apple");' not in result
        assert "Object _cf_result1 = store.getItems();" in result

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
        result = transform_java_assertions(source, "parse")
        assert 'assertThat(parser.parse("invalid")).isNull();' not in result
        assert 'Object _cf_result1 = parser.parse("invalid");' in result

    def test_assertthat_qualified(self):
        """Test Assertions.assertThat (qualified call)."""
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
        result = transform_java_assertions(source, "add")
        assert "assertThat" not in result
        assert "Object _cf_result1 = calc.add(1, 2);" in result


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
        result = transform_java_assertions(source, "add")
        assert "assertThat(calc.add(2, 3), is(5));" not in result
        assert "Object _cf_result1 = calc.add(2, 3);" in result

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
        result = transform_java_assertions(source, "add")
        assert "assertThat" not in result
        assert "Object _cf_result1 = calc.add(2, 3);" in result


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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertTrue" not in result
        # Both fibonacci calls are preserved inside the containing areConsecutiveFibonacci call
        assert "Object _cf_result1 = Fibonacci.areConsecutiveFibonacci(Fibonacci.fibonacci(5), Fibonacci.fibonacci(6));" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertEquals" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(0);" in result
        assert "Object _cf_result2 = Fibonacci.fibonacci(1);" in result
        assert "Object _cf_result3 = Fibonacci.fibonacci(2);" in result
        assert "Object _cf_result4 = Fibonacci.fibonacci(3);" in result
        assert "Object _cf_result5 = Fibonacci.fibonacci(5);" in result


class TestNoTargetCalls:
    """Tests for assertions that do NOT contain calls to the target function."""

    def test_assertion_without_target_removed(self):
        """Assertions not containing the target function should be removed."""
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
        result = transform_java_assertions(source, "fibonacci")
        # The assertNotNull without target call should be removed
        assert "assertNotNull(config);" not in result
        # The assertEquals with target call should be transformed
        assert "Object _cf_result1 = Fibonacci.fibonacci(10);" in result

    def test_no_assertions_at_all(self):
        """Source with no assertions should be returned unchanged."""
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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertEquals" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacci(10);" in result

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
        result = transform_java_assertions(source, "parse")
        assert "assertEquals" not in result
        assert 'Object _cf_result1 = parser.parse("input(1)");' in result

    def test_preserves_non_test_code(self):
        """Non-assertion code like setup, variable declarations should be preserved."""
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
        result = transform_java_assertions(source, "fibonacciSequence")
        assert "int n = 10;" in result
        assert "long[] expected = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};" in result
        assert "Object _cf_result1 = Fibonacci.fibonacciSequence(n);" in result

    def test_nested_method_calls(self):
        """Target function call nested inside another method call inside assertion."""
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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertEquals" not in result
        # Should capture the full top-level expression containing the target call
        assert "Object _cf_result1 = Fibonacci.fibonacciIndex(Fibonacci.fibonacci(10));" in result

    def test_chained_method_on_result(self):
        """Target function call with chained method (e.g., result.toString())."""
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
        result = transform_java_assertions(source, "fibonacciUpTo")
        assert "assertEquals" not in result
        assert "Object _cf_result1 = Fibonacci.fibonacciUpTo(20);" in result


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

    def test_all_assertfalse_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        # All assertFalse calls with target should be transformed
        assert "Object _cf_result1 = instance.get(0);" in result
        assert "Object _cf_result2 = instance.get(100);" in result
        assert "Object _cf_result3 = instance.get(lastIndex);" in result
        assert "Object _cf_result4 = instance.get(beyond);" in result

    def test_asserttrue_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert "Object" in result
        # assertTrue should also be transformed
        assert "bs.get(64 + 3);" in result

    def test_setup_code_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert "instance = new BitSet();" in result
        assert "int lastIndex = 16 * BitSet.BITS_PER_WORD - 1;" in result
        assert "int beyond = 16 * BitSet.BITS_PER_WORD;" in result

    def test_reflection_code_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert 'Field wordsField = BitSet.class.getDeclaredField("words");' in result
        assert "wordsField.setAccessible(true);" in result
        assert "long[] words = new long[2];" in result
        assert "words[1] = 1L << 3;" in result
        assert "wordsField.set(bs, words);" in result

    def test_expected_exception_test_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        # The expected-exception test has no assertion, just the call
        assert "instance.get(-1);" in result
        assert "@Test(expected = ArrayIndexOutOfBoundsException.class)" in result

    def test_package_and_imports_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert "package io.questdb.std;" in result
        assert "import org.junit.Before;" in result
        assert "import org.junit.Test;" in result
        assert "import java.lang.reflect.Field;" in result

    def test_class_structure_preserved(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert "public class BitSetTest {" in result
        assert "private BitSet instance;" in result
        assert "@Before" in result
        assert "public void setUp() {" in result

    def test_large_index_assertions_transformed(self):
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        assert "instance.get(Integer.MAX_VALUE);" in result
        assert "instance.get(63);" in result
        assert "instance.get(64);" in result
        assert "big.get(last);" in result

    def test_no_assertfalse_remain(self):
        """After transformation, no assertFalse with 'get' calls should remain."""
        result = transform_java_assertions(self.BITSET_TEST_SOURCE, "get")
        import re

        # Find any remaining assertFalse/assertTrue that contain a .get( call
        remaining = re.findall(r"assert(?:True|False)\(.*\.get\(", result)
        assert remaining == [], f"Found untransformed assertions: {remaining}"


class TestTransformMethod:
    """Tests for JavaAssertTransformer.transform() — each branch and code path."""

    # --- Early returns ---

    def test_none_source_returns_unchanged(self):
        """transform() returns empty string unchanged."""
        transformer = JavaAssertTransformer("fibonacci")
        assert transformer.transform("") == ""

    def test_whitespace_only_returns_unchanged(self):
        """transform() returns whitespace-only source unchanged."""
        transformer = JavaAssertTransformer("fibonacci")
        ws = "   \n\t\n  "
        assert transformer.transform(ws) == ws

    def test_no_assertions_found_returns_unchanged(self):
        """Source with code but no assertions → _find_assertions returns [] → early return."""
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
        """Assertions found but none contain target function are removed (empty replacement)."""
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
        result = transformer.transform(source)
        assert "assertEquals(4, calculator.add(2, 2));" not in result
        assert 'assertTrue(validator.isValid("x"))' not in result
        assert transformer.invocation_counter == 0

    # --- Counter numbering in source order ---

    def test_counters_assigned_in_source_order(self):
        """Counters _cf_result1, _cf_result2, etc. follow source position (top to bottom)."""
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
        result = transformer.transform(source)
        # First assertion in source gets _cf_result1, second gets _cf_result2, etc.
        pos1 = result.index("_cf_result1")
        pos2 = result.index("_cf_result2")
        pos3 = result.index("_cf_result3")
        assert pos1 < pos2 < pos3
        assert "Fibonacci.fibonacci(0)" in result.split("_cf_result1")[1].split("\n")[0]
        assert "Fibonacci.fibonacci(10)" in result.split("_cf_result2")[1].split("\n")[0]
        assert "Fibonacci.fibonacci(1)" in result.split("_cf_result3")[1].split("\n")[0]
        assert transformer.invocation_counter == 3

    def test_counter_increments_across_transform_call(self):
        """Counter keeps incrementing across a single transform() call."""
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
        """assertEquals inside assertAll is nested → only assertAll is replaced, not inner ones individually."""
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
        result = transformer.transform(source)
        # assertAll is the outer assertion and should be replaced
        assert "assertAll" not in result
        # The individual assertEquals should NOT remain as separate replacements
        # (they are nested inside assertAll, so the nesting filter removes them)
        # But the target calls should still be captured
        lines = [l.strip() for l in result.splitlines() if "_cf_result" in l]
        assert len(lines) >= 1  # At least the outer replacement should produce captures

    def test_non_nested_assertions_all_replaced(self):
        """Multiple top-level assertions (not nested) are all removed."""
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
        result = transformer.transform(source)
        assert "assertEquals" not in result
        # assertEquals with Fibonacci.fibonacci(0) has target call, gets captured
        assert "Object _cf_result1 = Fibonacci.fibonacci(0);" in result
        # assertTrue/assertFalse don't contain "fibonacci" calls, so they are removed (empty)
        assert "assertTrue(Fibonacci.isFibonacci(5));" not in result
        assert "assertFalse(Fibonacci.isFibonacci(4));" not in result

    # --- Reverse replacement preserves positions ---

    def test_reverse_replacement_preserves_all_positions(self):
        """Replacing in reverse order ensures positions stay correct for multi-replacement."""
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
        result = transformer.transform(source)
        assert "assertEquals" not in result
        assert "Object _cf_result1 = engine.compute(1);" in result
        assert "Object _cf_result2 = engine.compute(2);" in result
        assert "Object _cf_result3 = engine.compute(3);" in result
        assert "Object _cf_result4 = engine.compute(4);" in result
        assert "Object _cf_result5 = engine.compute(5);" in result
        assert transformer.invocation_counter == 5

    # --- Mixed assertions: some with target, some without ---

    def test_mixed_assertions_all_removed(self):
        """All assertions are removed; targeted ones get capture statements."""
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
        result = transformer.transform(source)
        # Non-targeted assertions are removed
        assert "assertNotNull(config);" not in result
        assert "assertTrue(isReady);" not in result
        assert "assertFalse(isDone);" not in result
        # Targeted assertions are replaced with capture statements
        assert "Object _cf_result1 = Fibonacci.fibonacci(0);" in result
        assert "Object _cf_result2 = Fibonacci.fibonacci(1);" in result
        assert transformer.invocation_counter == 2

    # --- Exception assertions in transform ---

    def test_exception_assertion_without_target_calls_still_replaced(self):
        """assertThrows is replaced even if lambda doesn't contain the target function,
        because is_exception_assertion=True passes the filter."""
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
        result = transformer.transform(source)
        # assertThrows is an exception assertion so it passes the filter
        assert "assertThrows" not in result
        assert "try {" in result

    # --- Full output exact equality ---

    def test_single_assertion_exact_output(self):
        """Verify exact output for the simplest single-assertion case."""
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
        """Verify exact output when multiple assertions are replaced."""
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
        """Running transform on already-transformed code (no assertions) returns it unchanged."""
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
        # Second pass with a new transformer should be a no-op (no assertions left)
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
        result = transformer.transform(source)
        assert "_cf_result1" in result
        assert "_cf_result2" in result
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
        result = transform_java_assertions(source, "fibonacci")
        # assertAll should be transformed (it contains target calls)
        assert "assertAll" not in result


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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "try {" in result
        assert "Fibonacci.fibonacci(-1);" in result

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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "try {" in result

    def test_assertthrows_with_final_variable(self):
        """Test assertThrows assigned to a final variable."""
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
        result = transform_java_assertions(source, "fibonacci")
        assert "assertThrows" not in result
        assert "Fibonacci.fibonacci(-1);" in result


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

    def test_all_assertions_removed(self):
        result = transform_java_assertions(self.MULTI_FUNCTION_TEST, "fibonacci")
        # ALL assertions should be removed
        assert "assertEquals(0, Fibonacci.fibonacci(0))" not in result
        assert "assertEquals(1, Fibonacci.fibonacci(1))" not in result
        assert "assertTrue(Fibonacci.isFibonacci(0))" not in result
        assert "assertTrue(Fibonacci.isPerfectSquare(0))" not in result
        assert "assertArrayEquals" not in result
        assert "assertEquals(0, Fibonacci.fibonacciIndex(0))" not in result
        assert "assertEquals(0, Fibonacci.sumFibonacci(0))" not in result
        assert "assertFalse" not in result
        # Target function calls should be captured
        assert "Object _cf_result" in result
        assert "Fibonacci.fibonacci(0)" in result
        # Exception assertion should be converted to try/catch
        assert "assertThrows" not in result
        assert "Fibonacci.fibonacci(-1);" in result

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
        result = transform_java_assertions(source, "add")
        # Non-assertion code should be preserved
        assert "Calculator calc = new Calculator();" in result
        assert "int result = calc.setup();" in result
        # All assertions should be removed
        assert "assertEquals(5, calc.add(2, 3))" not in result
        assert "assertTrue(calc.isReady())" not in result
        # Target function call should be captured
        assert "Object _cf_result" in result
        assert "calc.add(2, 3)" in result

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
        result = transform_java_assertions(source, "fibonacci")
        # assertThat calls should be removed (only import references remain)
        assert "assertThat(Fibonacci.fibonacci(5))" not in result
        assert "assertThat(Fibonacci.isFibonacci(5))" not in result
        assert "Fibonacci.fibonacci(5)" in result
        assert "isTrue" not in result
        assert "isEqualTo" not in result

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
        result = transform_java_assertions(source, "target")
        assert "assertEquals" not in result
        assert "assertNull" not in result
        assert "assertNotNull" not in result
        assert "assertTrue" not in result
        assert "Object _cf_result" in result
        assert "obj.target(1)" in result
