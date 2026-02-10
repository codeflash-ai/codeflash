"""Tests for Java assertion removal transformer.

This test suite covers the transformation of Java test assertions into
regression test code that captures function return values.

All tests assert for full string equality, no substring matching.
"""

from codeflash.languages.java.remove_asserts import JavaAssertTransformer, transform_java_assertions


class TestBasicJUnit5Assertions:
    """Tests for basic JUnit 5 assertion transformations."""

    def test_assert_equals_basic(self):
        source = """\
@Test
void testFibonacci() {
    assertEquals(55, calculator.fibonacci(10));
}"""
        expected = """\
@Test
void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assert_equals_with_message(self):
        source = """\
@Test
void testFibonacci() {
    assertEquals(55, calculator.fibonacci(10), "Fibonacci of 10 should be 55");
}"""
        expected = """\
@Test
void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assert_true(self):
        source = """\
@Test
void testIsValid() {
    assertTrue(validator.isValid("test"));
}"""
        expected = """\
@Test
void testIsValid() {
    Object _cf_result1 = validator.isValid("test");
}"""
        result = transform_java_assertions(source, "isValid")
        assert result == expected

    def test_assert_false(self):
        source = """\
@Test
void testIsInvalid() {
    assertFalse(validator.isValid(""));
}"""
        expected = """\
@Test
void testIsInvalid() {
    Object _cf_result1 = validator.isValid("");
}"""
        result = transform_java_assertions(source, "isValid")
        assert result == expected

    def test_assert_null(self):
        source = """\
@Test
void testGetNull() {
    assertNull(processor.getValue(null));
}"""
        expected = """\
@Test
void testGetNull() {
    Object _cf_result1 = processor.getValue(null);
}"""
        result = transform_java_assertions(source, "getValue")
        assert result == expected

    def test_assert_not_null(self):
        source = """\
@Test
void testGetValue() {
    assertNotNull(processor.getValue("key"));
}"""
        expected = """\
@Test
void testGetValue() {
    Object _cf_result1 = processor.getValue("key");
}"""
        result = transform_java_assertions(source, "getValue")
        assert result == expected

    def test_assert_not_equals(self):
        source = """\
@Test
void testDifferent() {
    assertNotEquals(0, calculator.add(1, 2));
}"""
        expected = """\
@Test
void testDifferent() {
    Object _cf_result1 = calculator.add(1, 2);
}"""
        result = transform_java_assertions(source, "add")
        assert result == expected

    def test_assert_same(self):
        source = """\
@Test
void testSame() {
    assertSame(expected, factory.getInstance());
}"""
        expected = """\
@Test
void testSame() {
    Object _cf_result1 = factory.getInstance();
}"""
        result = transform_java_assertions(source, "getInstance")
        assert result == expected

    def test_assert_array_equals(self):
        source = """\
@Test
void testSort() {
    assertArrayEquals(expected, sorter.sort(input));
}"""
        expected = """\
@Test
void testSort() {
    Object _cf_result1 = sorter.sort(input);
}"""
        result = transform_java_assertions(source, "sort")
        assert result == expected


class TestJUnit5PrefixedAssertions:
    """Tests for JUnit 5 assertions with Assertions. prefix."""

    def test_assertions_prefix(self):
        source = """\
@Test
void testFibonacci() {
    Assertions.assertEquals(55, calculator.fibonacci(10));
}"""
        expected = """\
@Test
void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assert_prefix(self):
        source = """\
@Test
void testAdd() {
    Assert.assertEquals(5, calculator.add(2, 3));
}"""
        expected = """\
@Test
void testAdd() {
    Object _cf_result1 = calculator.add(2, 3);
}"""
        result = transform_java_assertions(source, "add")
        assert result == expected


class TestJUnit5ExceptionAssertions:
    """Tests for JUnit 5 exception assertions."""

    def test_assert_throws_lambda(self):
        source = """\
@Test
void testDivideByZero() {
    assertThrows(IllegalArgumentException.class, () -> calculator.divide(1, 0));
}"""
        expected = """\
@Test
void testDivideByZero() {
    try { calculator.divide(1, 0); } catch (Exception _cf_ignored1) {}
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_throws_block_lambda(self):
        source = """\
@Test
void testDivideByZero() {
    assertThrows(ArithmeticException.class, () -> {
        calculator.divide(1, 0);
    });
}"""
        expected = """\
@Test
void testDivideByZero() {
    try { calculator.divide(1, 0); } catch (Exception _cf_ignored1) {}
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_does_not_throw(self):
        source = """\
@Test
void testValidDivision() {
    assertDoesNotThrow(() -> calculator.divide(10, 2));
}"""
        expected = """\
@Test
void testValidDivision() {
    try { calculator.divide(10, 2); } catch (Exception _cf_ignored1) {}
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected


class TestStaticMethodCalls:
    """Tests for static method call handling."""

    def test_static_method_call(self):
        source = """\
@Test
void testQuickAdd() {
    assertEquals(15.0, Calculator.quickAdd(10.0, 5.0));
}"""
        expected = """\
@Test
void testQuickAdd() {
    Object _cf_result1 = Calculator.quickAdd(10.0, 5.0);
}"""
        result = transform_java_assertions(source, "quickAdd")
        assert result == expected

    def test_static_method_fully_qualified(self):
        source = """\
@Test
void testReverse() {
    assertEquals("olleh", com.example.StringUtils.reverse("hello"));
}"""
        expected = """\
@Test
void testReverse() {
    Object _cf_result1 = com.example.StringUtils.reverse("hello");
}"""
        result = transform_java_assertions(source, "reverse")
        assert result == expected


class TestMultipleAssertions:
    """Tests for multiple assertions in a single test method."""

    def test_multiple_assertions_same_function(self):
        source = """\
@Test
void testFibonacciSequence() {
    assertEquals(0, calculator.fibonacci(0));
    assertEquals(1, calculator.fibonacci(1));
    assertEquals(55, calculator.fibonacci(10));
}"""
        expected = """\
@Test
void testFibonacciSequence() {
    Object _cf_result1 = calculator.fibonacci(0);
    Object _cf_result2 = calculator.fibonacci(1);
    Object _cf_result3 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_multiple_assertions_different_functions(self):
        source = """\
@Test
void testCalculator() {
    assertEquals(5, calculator.add(2, 3));
    assertEquals(6, calculator.multiply(2, 3));
}"""
        expected = """\
@Test
void testCalculator() {
    Object _cf_result1 = calculator.add(2, 3);
    assertEquals(6, calculator.multiply(2, 3));
}"""
        result = transform_java_assertions(source, "add")
        assert result == expected


class TestAssertJFluentAssertions:
    """Tests for AssertJ fluent assertion transformations."""

    def test_assertj_basic(self):
        source = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testFibonacci() {
    assertThat(calculator.fibonacci(10)).isEqualTo(55);
}"""
        expected = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertj_chained(self):
        source = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetList() {
    assertThat(processor.getList()).hasSize(5).contains("a", "b");
}"""
        expected = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetList() {
    Object _cf_result1 = processor.getList();
}"""
        result = transform_java_assertions(source, "getList")
        assert result == expected

    def test_assertj_is_null(self):
        source = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetNull() {
    assertThat(processor.getValue(null)).isNull();
}"""
        expected = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetNull() {
    Object _cf_result1 = processor.getValue(null);
}"""
        result = transform_java_assertions(source, "getValue")
        assert result == expected

    def test_assertj_is_not_empty(self):
        source = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetList() {
    assertThat(processor.getList()).isNotEmpty();
}"""
        expected = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testGetList() {
    Object _cf_result1 = processor.getList();
}"""
        result = transform_java_assertions(source, "getList")
        assert result == expected


class TestNestedMethodCalls:
    """Tests for nested method calls in assertions."""

    def test_nested_call_in_expected(self):
        source = """\
@Test
void testCompare() {
    assertEquals(helper.getExpected(), calculator.compute(5));
}"""
        expected = """\
@Test
void testCompare() {
    Object _cf_result1 = calculator.compute(5);
}"""
        result = transform_java_assertions(source, "compute")
        assert result == expected

    def test_nested_call_as_argument(self):
        source = """\
@Test
void testProcess() {
    assertEquals(expected, processor.process(helper.getData()));
}"""
        expected = """\
@Test
void testProcess() {
    Object _cf_result1 = processor.process(helper.getData());
}"""
        result = transform_java_assertions(source, "process")
        assert result == expected

    def test_deeply_nested(self):
        source = """\
@Test
void testDeep() {
    assertEquals(expected, outer.process(inner.compute(calculator.fibonacci(5))));
}"""
        expected = """\
@Test
void testDeep() {
    Object _cf_result1 = calculator.fibonacci(5);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestWhitespacePreservation:
    """Tests for whitespace and indentation preservation."""

    def test_preserves_indentation(self):
        source = """\
    @Test
    void testFibonacci() {
        assertEquals(55, calculator.fibonacci(10));
    }"""
        expected = """\
    @Test
    void testFibonacci() {
        Object _cf_result1 = calculator.fibonacci(10);
    }"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_multiline_assertion(self):
        source = """\
@Test
void testLongAssertion() {
    assertEquals(
        expectedValue,
        calculator.computeComplexResult(
            arg1,
            arg2,
            arg3
        )
    );
}"""
        expected = """\
@Test
void testLongAssertion() {
    Object _cf_result1 = calculator.computeComplexResult(
            arg1,
            arg2,
            arg3
        );
}"""
        result = transform_java_assertions(source, "computeComplexResult")
        assert result == expected


class TestStringsWithSpecialCharacters:
    """Tests for strings containing special characters."""

    def test_string_with_parentheses(self):
        source = """\
@Test
void testFormat() {
    assertEquals("hello (world)", formatter.format("hello", "world"));
}"""
        expected = """\
@Test
void testFormat() {
    Object _cf_result1 = formatter.format("hello", "world");
}"""
        result = transform_java_assertions(source, "format")
        assert result == expected

    def test_string_with_quotes(self):
        source = """\
@Test
void testEscape() {
    assertEquals("hello \\"world\\"", formatter.escape("hello \\"world\\""));
}"""
        expected = """\
@Test
void testEscape() {
    Object _cf_result1 = formatter.escape("hello \\"world\\"");
}"""
        result = transform_java_assertions(source, "escape")
        assert result == expected

    def test_string_with_newlines(self):
        source = """\
@Test
void testMultiline() {
    assertEquals("line1\\nline2", processor.join("line1", "line2"));
}"""
        expected = """\
@Test
void testMultiline() {
    Object _cf_result1 = processor.join("line1", "line2");
}"""
        result = transform_java_assertions(source, "join")
        assert result == expected


class TestNonAssertionCodePreservation:
    """Tests that non-assertion code is preserved unchanged."""

    def test_setup_code_preserved(self):
        source = """\
@Test
void testWithSetup() {
    Calculator calc = new Calculator(2);
    int input = 10;
    assertEquals(55, calc.fibonacci(input));
}"""
        expected = """\
@Test
void testWithSetup() {
    Calculator calc = new Calculator(2);
    int input = 10;
    Object _cf_result1 = calc.fibonacci(input);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_other_method_calls_preserved(self):
        source = """\
@Test
void testWithHelper() {
    helper.setup();
    assertEquals(55, calculator.fibonacci(10));
    helper.cleanup();
}"""
        expected = """\
@Test
void testWithHelper() {
    helper.setup();
    Object _cf_result1 = calculator.fibonacci(10);
    helper.cleanup();
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_variable_declarations_preserved(self):
        source = """\
@Test
void testWithVariables() {
    int expected = 55;
    int actual = calculator.fibonacci(10);
    assertEquals(expected, actual);
}"""
        # fibonacci is assigned to 'actual', not in the assertion - no transformation
        expected = source
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestParameterizedTests:
    """Tests for parameterized test handling."""

    def test_parameterized_test(self):
        source = """\
@ParameterizedTest
@CsvSource({
    "0, 0",
    "1, 1",
    "10, 55"
})
void testFibonacciSequence(int n, long expected) {
    assertEquals(expected, calculator.fibonacci(n));
}"""
        expected = """\
@ParameterizedTest
@CsvSource({
    "0, 0",
    "1, 1",
    "10, 55"
})
void testFibonacciSequence(int n, long expected) {
    Object _cf_result1 = calculator.fibonacci(n);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestNestedTestClasses:
    """Tests for nested test class handling."""

    def test_nested_class(self):
        source = """\
@Nested
@DisplayName("Fibonacci Tests")
class FibonacciTests {
    @Test
    void testBasic() {
        assertEquals(55, calculator.fibonacci(10));
    }
}"""
        expected = """\
@Nested
@DisplayName("Fibonacci Tests")
class FibonacciTests {
    @Test
    void testBasic() {
        Object _cf_result1 = calculator.fibonacci(10);
    }
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestMockitoPreservation:
    """Tests that Mockito code is not modified."""

    def test_mockito_when_preserved(self):
        source = """\
@Test
void testWithMock() {
    when(mockService.getData()).thenReturn("test");
    assertEquals("test", processor.process(mockService));
}"""
        expected = """\
@Test
void testWithMock() {
    when(mockService.getData()).thenReturn("test");
    Object _cf_result1 = processor.process(mockService);
}"""
        result = transform_java_assertions(source, "process")
        assert result == expected

    def test_mockito_verify_preserved(self):
        source = """\
@Test
void testWithVerify() {
    processor.process(mockService);
    verify(mockService).getData();
}"""
        # No assertions to transform, source unchanged
        expected = source
        result = transform_java_assertions(source, "process")
        assert result == expected


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_source(self):
        result = transform_java_assertions("", "fibonacci")
        assert result == ""

    def test_whitespace_only(self):
        source = "   \n\t  "
        result = transform_java_assertions(source, "fibonacci")
        assert result == source

    def test_no_assertions(self):
        source = """\
@Test
void testNoAssertions() {
    calculator.fibonacci(10);
}"""
        expected = source
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assertion_without_target_function(self):
        source = """\
@Test
void testOther() {
    assertEquals(5, helper.compute(3));
}"""
        # No transformation since target function is not in the assertion
        expected = source
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_function_name_in_string(self):
        source = """\
@Test
void testWithStringContainingFunctionName() {
    assertEquals("fibonacci(10) = 55", formatter.format("fibonacci", 10, 55));
}"""
        expected = """\
@Test
void testWithStringContainingFunctionName() {
    Object _cf_result1 = formatter.format("fibonacci", 10, 55);
}"""
        result = transform_java_assertions(source, "format")
        assert result == expected


class TestJUnit4Compatibility:
    """Tests for JUnit 4 style assertions."""

    def test_junit4_assert_equals(self):
        source = """\
import static org.junit.Assert.*;

@Test
public void testFibonacci() {
    assertEquals(55, calculator.fibonacci(10));
}"""
        expected = """\
import static org.junit.Assert.*;

@Test
public void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_junit4_with_message_first(self):
        source = """\
@Test
public void testFibonacci() {
    assertEquals("Should be 55", 55, calculator.fibonacci(10));
}"""
        expected = """\
@Test
public void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestAssertAll:
    """Tests for assertAll grouped assertions."""

    def test_assert_all_basic(self):
        source = """\
@Test
void testMultiple() {
    assertAll(
        () -> assertEquals(0, calculator.fibonacci(0)),
        () -> assertEquals(1, calculator.fibonacci(1)),
        () -> assertEquals(55, calculator.fibonacci(10))
    );
}"""
        expected = """\
@Test
void testMultiple() {
    Object _cf_result1 = calculator.fibonacci(0);
    Object _cf_result2 = calculator.fibonacci(1);
    Object _cf_result3 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestTransformerClass:
    """Tests for the JavaAssertTransformer class directly."""

    def test_invocation_counter_increments(self):
        transformer = JavaAssertTransformer("fibonacci")
        source = """\
@Test
void test() {
    assertEquals(0, calc.fibonacci(0));
    assertEquals(1, calc.fibonacci(1));
}"""
        expected = """\
@Test
void test() {
    Object _cf_result1 = calc.fibonacci(0);
    Object _cf_result2 = calc.fibonacci(1);
}"""
        result = transformer.transform(source)
        assert result == expected
        assert transformer.invocation_counter == 2

    def test_qualified_name_support(self):
        transformer = JavaAssertTransformer(
            function_name="fibonacci",
            qualified_name="com.example.Calculator.fibonacci",
        )
        assert transformer.qualified_name == "com.example.Calculator.fibonacci"

    def test_custom_analyzer(self):
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        transformer = JavaAssertTransformer("fibonacci", analyzer=analyzer)
        assert transformer.analyzer is analyzer


class TestImportDetection:
    """Tests for framework detection from imports."""

    def test_detect_junit5(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;"""
        transformer = JavaAssertTransformer("test")
        transformer._detected_framework = transformer._detect_framework(source)
        assert transformer._detected_framework == "junit5"

    def test_detect_assertj(self):
        source = """\
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;"""
        transformer = JavaAssertTransformer("test")
        transformer._detected_framework = transformer._detect_framework(source)
        assert transformer._detected_framework == "assertj"

    def test_detect_testng(self):
        source = """\
import org.testng.Assert;
import org.testng.annotations.Test;"""
        transformer = JavaAssertTransformer("test")
        transformer._detected_framework = transformer._detect_framework(source)
        assert transformer._detected_framework == "testng"

    def test_detect_hamcrest(self):
        source = """\
import org.junit.Test;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;"""
        transformer = JavaAssertTransformer("test")
        transformer._detected_framework = transformer._detect_framework(source)
        assert transformer._detected_framework == "hamcrest"


class TestInstrumentGeneratedJavaTest:
    """Tests for the instrument_generated_java_test integration."""

    def test_behavior_mode_removes_assertions(self):
        from codeflash.languages.java.instrumentation import instrument_generated_java_test

        test_code = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Calculator calc = new Calculator();
        assertEquals(55, calc.fibonacci(10));
    }
}"""
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest__perfinstrumented {
    @Test
    void testFibonacci() {
        Calculator calc = new Calculator();
        Object _cf_result1 = calc.fibonacci(10);
    }
}"""
        result = instrument_generated_java_test(
            test_code=test_code,
            function_name="fibonacci",
            qualified_name="com.example.Calculator.fibonacci",
            mode="behavior",
        )
        assert result == expected

    def test_behavior_mode_with_assertj(self):
        from codeflash.languages.java.instrumentation import instrument_generated_java_test

        test_code = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class StringUtilsTest {
    @Test
    void testReverse() {
        assertThat(StringUtils.reverse("hello")).isEqualTo("olleh");
    }
}"""
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class StringUtilsTest__perfinstrumented {
    @Test
    void testReverse() {
        Object _cf_result1 = StringUtils.reverse("hello");
    }
}"""
        result = instrument_generated_java_test(
            test_code=test_code,
            function_name="reverse",
            qualified_name="com.example.StringUtils.reverse",
            mode="behavior",
        )
        assert result == expected


class TestComplexRealWorldExamples:
    """Tests based on real-world test patterns."""

    def test_calculator_test_pattern(self):
        source = """\
@Test
@DisplayName("should calculate compound interest for basic case")
void testBasicCompoundInterest() {
    String result = calculator.calculateCompoundInterest(1000.0, 0.05, 1, 12);
    assertNotNull(result);
    assertTrue(result.contains("."));
}"""
        # assertNotNull(result) and assertTrue(result.contains(".")) don't contain the target function
        # so they remain unchanged, and the variable assignment is also preserved
        expected = source
        result = transform_java_assertions(source, "calculateCompoundInterest")
        assert result == expected

    def test_string_utils_pattern(self):
        source = """\
@Test
@DisplayName("should reverse a simple string")
void testReverseSimple() {
    assertEquals("olleh", StringUtils.reverse("hello"));
    assertEquals("dlrow", StringUtils.reverse("world"));
}"""
        expected = """\
@Test
@DisplayName("should reverse a simple string")
void testReverseSimple() {
    Object _cf_result1 = StringUtils.reverse("hello");
    Object _cf_result2 = StringUtils.reverse("world");
}"""
        result = transform_java_assertions(source, "reverse")
        assert result == expected

    def test_with_before_each_setup(self):
        source = """\
private Calculator calculator;

@BeforeEach
void setUp() {
    calculator = new Calculator(2);
}

@Test
void testFibonacci() {
    assertEquals(55, calculator.fibonacci(10));
}"""
        expected = """\
private Calculator calculator;

@BeforeEach
void setUp() {
    calculator = new Calculator(2);
}

@Test
void testFibonacci() {
    Object _cf_result1 = calculator.fibonacci(10);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected


class TestConcurrencyPatterns:
    """Tests that assertion removal correctly handles Java concurrency constructs.

    Validates that synchronized blocks, volatile field access, atomic operations,
    concurrent collections, Thread.sleep, wait/notify, and synchronized method
    modifiers are all preserved verbatim after assertion transformation.
    """

    def test_synchronized_method_assertion_removal(self):
        """Assertion inside synchronized block is transformed; synchronized wrapper preserved."""
        source = """\
@Test
void testSynchronizedAccess() {
    synchronized (lock) {
        assertEquals(42, counter.incrementAndGet());
    }
}"""
        expected = """\
@Test
void testSynchronizedAccess() {
    synchronized (lock) {
        Object _cf_result1 = counter.incrementAndGet();
    }
}"""
        result = transform_java_assertions(source, "incrementAndGet")
        assert result == expected

    def test_volatile_field_read_preserved(self):
        """Assertion wrapping a volatile field reader is transformed; method call preserved."""
        source = """\
@Test
void testVolatileRead() {
    assertTrue(buffer.isReady());
}"""
        expected = """\
@Test
void testVolatileRead() {
    Object _cf_result1 = buffer.isReady();
}"""
        result = transform_java_assertions(source, "isReady")
        assert result == expected

    def test_synchronized_block_with_multiple_assertions(self):
        """Multiple assertions inside a synchronized block are all transformed."""
        source = """\
@Test
void testSynchronizedBlock() {
    synchronized (cache) {
        assertEquals(1, cache.size());
        assertNotNull(cache.get("key"));
        assertTrue(cache.containsKey("key"));
    }
}"""
        expected = """\
@Test
void testSynchronizedBlock() {
    synchronized (cache) {
        Object _cf_result1 = cache.size();
        assertNotNull(cache.get("key"));
        assertTrue(cache.containsKey("key"));
    }
}"""
        result = transform_java_assertions(source, "size")
        assert result == expected

    def test_synchronized_block_multiple_assertions_same_target(self):
        """Multiple assertions in synchronized block targeting the same function."""
        source = """\
@Test
void testSynchronizedBlock() {
    synchronized (cache) {
        assertNotNull(cache.get("key1"));
        assertNotNull(cache.get("key2"));
    }
}"""
        expected = """\
@Test
void testSynchronizedBlock() {
    synchronized (cache) {
        Object _cf_result1 = cache.get("key1");
        Object _cf_result2 = cache.get("key2");
    }
}"""
        result = transform_java_assertions(source, "get")
        assert result == expected

    def test_atomic_operations_preserved(self):
        """Atomic operations (incrementAndGet) are preserved as Object capture calls."""
        source = """\
@Test
void testAtomicCounter() {
    assertEquals(1, counter.incrementAndGet());
    assertEquals(2, counter.incrementAndGet());
}"""
        expected = """\
@Test
void testAtomicCounter() {
    Object _cf_result1 = counter.incrementAndGet();
    Object _cf_result2 = counter.incrementAndGet();
}"""
        result = transform_java_assertions(source, "incrementAndGet")
        assert result == expected

    def test_concurrent_collection_assertion(self):
        """ConcurrentHashMap putIfAbsent call is preserved in assertion transformation."""
        source = """\
@Test
void testConcurrentMap() {
    assertEquals("value", concurrentMap.putIfAbsent("key", "value"));
}"""
        expected = """\
@Test
void testConcurrentMap() {
    Object _cf_result1 = concurrentMap.putIfAbsent("key", "value");
}"""
        result = transform_java_assertions(source, "putIfAbsent")
        assert result == expected

    def test_thread_sleep_with_assertion(self):
        """Thread.sleep() before assertion is preserved verbatim."""
        source = """\
@Test
void testWithThreadSleep() throws InterruptedException {
    Thread.sleep(100);
    assertEquals(42, processor.getResult());
}"""
        expected = """\
@Test
void testWithThreadSleep() throws InterruptedException {
    Thread.sleep(100);
    Object _cf_result1 = processor.getResult();
}"""
        result = transform_java_assertions(source, "getResult")
        assert result == expected

    def test_synchronized_method_signature_preserved(self):
        """synchronized modifier on a test method is preserved after transformation."""
        source = """\
@Test
synchronized void testSyncMethod() {
    assertEquals(10, calculator.compute(5));
}"""
        expected = """\
@Test
synchronized void testSyncMethod() {
    Object _cf_result1 = calculator.compute(5);
}"""
        result = transform_java_assertions(source, "compute")
        assert result == expected

    def test_wait_notify_pattern_preserved(self):
        """wait/notify pattern around an assertion is preserved."""
        source = """\
@Test
void testWaitNotify() {
    synchronized (monitor) {
        monitor.notify();
    }
    assertTrue(listener.wasNotified());
}"""
        expected = """\
@Test
void testWaitNotify() {
    synchronized (monitor) {
        monitor.notify();
    }
    Object _cf_result1 = listener.wasNotified();
}"""
        result = transform_java_assertions(source, "wasNotified")
        assert result == expected

    def test_reentrant_lock_pattern_preserved(self):
        """ReentrantLock acquire/release around assertion is preserved."""
        source = """\
@Test
void testReentrantLock() {
    lock.lock();
    try {
        assertEquals(99, sharedResource.getValue());
    } finally {
        lock.unlock();
    }
}"""
        expected = """\
@Test
void testReentrantLock() {
    lock.lock();
    try {
        Object _cf_result1 = sharedResource.getValue();
    } finally {
        lock.unlock();
    }
}"""
        result = transform_java_assertions(source, "getValue")
        assert result == expected

    def test_count_down_latch_pattern_preserved(self):
        """CountDownLatch await/countDown around assertion is preserved."""
        source = """\
@Test
void testCountDownLatch() throws InterruptedException {
    latch.countDown();
    latch.await();
    assertEquals(42, collector.getTotal());
}"""
        expected = """\
@Test
void testCountDownLatch() throws InterruptedException {
    latch.countDown();
    latch.await();
    Object _cf_result1 = collector.getTotal();
}"""
        result = transform_java_assertions(source, "getTotal")
        assert result == expected

    def test_token_bucket_synchronized_method(self):
        """Real pattern: synchronized method call (like TokenBucket.allowRequest) inside assertion."""
        source = """\
@Test
void testTokenBucketAllowRequest() {
    TokenBucket bucket = new TokenBucket(10, 1);
    assertTrue(bucket.allowRequest());
    assertTrue(bucket.allowRequest());
}"""
        expected = """\
@Test
void testTokenBucketAllowRequest() {
    TokenBucket bucket = new TokenBucket(10, 1);
    Object _cf_result1 = bucket.allowRequest();
    Object _cf_result2 = bucket.allowRequest();
}"""
        result = transform_java_assertions(source, "allowRequest")
        assert result == expected

    def test_circular_buffer_atomic_integer_pattern(self):
        """Real pattern: CircularBuffer with AtomicInteger-backed isEmpty/isFull assertions."""
        source = """\
@Test
void testCircularBufferOperations() {
    CircularBuffer<Integer> buffer = new CircularBuffer<>(3);
    assertTrue(buffer.isEmpty());
    buffer.put(1);
    assertFalse(buffer.isEmpty());
    assertTrue(buffer.put(2));
}"""
        expected = """\
@Test
void testCircularBufferOperations() {
    CircularBuffer<Integer> buffer = new CircularBuffer<>(3);
    Object _cf_result1 = buffer.isEmpty();
    buffer.put(1);
    Object _cf_result2 = buffer.isEmpty();
    Object _cf_result3 = buffer.put(2);
}"""
        result = transform_java_assertions(source, "isEmpty")
        # isEmpty is target for assertTrue/assertFalse; but put is NOT the target
        # so only isEmpty calls inside assertions are transformed
        # Actually: assertTrue(buffer.put(2)) also contains a non-target call
        # Let's verify what actually happens
        # put is not "isEmpty", so assertTrue(buffer.put(2)) has no target call -> untouched
        expected_corrected = """\
@Test
void testCircularBufferOperations() {
    CircularBuffer<Integer> buffer = new CircularBuffer<>(3);
    Object _cf_result1 = buffer.isEmpty();
    buffer.put(1);
    Object _cf_result2 = buffer.isEmpty();
    assertTrue(buffer.put(2));
}"""
        result = transform_java_assertions(source, "isEmpty")
        assert result == expected_corrected

    def test_concurrent_assertion_with_assertj(self):
        """AssertJ assertion on a synchronized method call is correctly transformed."""
        source = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testSynchronizedMethodWithAssertJ() {
    synchronized (lock) {
        assertThat(counter.incrementAndGet()).isEqualTo(1);
    }
}"""
        expected = """\
import static org.assertj.core.api.Assertions.assertThat;

@Test
void testSynchronizedMethodWithAssertJ() {
    synchronized (lock) {
        Object _cf_result1 = counter.incrementAndGet();
    }
}"""
        result = transform_java_assertions(source, "incrementAndGet")
        assert result == expected


class TestFullyQualifiedAssertions:
    """Tests for fully qualified assertion calls like org.junit.jupiter.api.Assertions.assertXxx."""

    def test_assert_timeout_fully_qualified_with_variable_assignment(self):
        source = """\
@Test
void testLargeInput() {
    Long result = org.junit.jupiter.api.Assertions.assertTimeout(
            Duration.ofSeconds(1),
            () -> Fibonacci.fibonacci(100_000)
    );
}"""
        expected = """\
@Test
void testLargeInput() {
    Object _cf_result1 = Fibonacci.fibonacci(100_000);
}"""
        result = transform_java_assertions(source, "fibonacci")
        assert result == expected

    def test_assert_equals_fully_qualified(self):
        source = """\
@Test
void testAdd() {
    org.junit.jupiter.api.Assertions.assertEquals(5, calc.add(2, 3));
}"""
        expected = """\
@Test
void testAdd() {
    Object _cf_result1 = calc.add(2, 3);
}"""
        result = transform_java_assertions(source, "add")
        assert result == expected


class TestAssertThrowsVariableAssignment:
    """Tests for assertThrows assigned to a variable: Type var = assertThrows(...)."""

    def test_assert_throws_assigned_to_variable(self):
        source = """\
@Test
void testDivideByZero() {
    Calculator calc = new Calculator();
    IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> calc.divide(1, 0));
    assertEquals("Cannot divide by zero", ex.getMessage());
}"""
        expected = """\
@Test
void testDivideByZero() {
    Calculator calc = new Calculator();
    IllegalArgumentException ex = null;
    try { calc.divide(1, 0); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; }
    assertEquals("Cannot divide by zero", ex.getMessage());
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_throws_assigned_to_variable_block_lambda(self):
        source = """\
@Test
void testDivideByZero() {
    ArithmeticException ex = assertThrows(ArithmeticException.class, () -> {
        calculator.divide(1, 0);
    });
}"""
        expected = """\
@Test
void testDivideByZero() {
    ArithmeticException ex = null;
    try { calculator.divide(1, 0); } catch (ArithmeticException _cf_caught1) { ex = _cf_caught1; }
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_throws_assigned_with_final_modifier(self):
        source = """\
@Test
void testDivideByZero() {
    final IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> calc.divide(1, 0));
}"""
        expected = """\
@Test
void testDivideByZero() {
    IllegalArgumentException ex = null;
    try { calc.divide(1, 0); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; }
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_throws_not_assigned_unchanged(self):
        source = """\
@Test
void testDivideByZero() {
    assertThrows(IllegalArgumentException.class, () -> calculator.divide(1, 0));
}"""
        expected = """\
@Test
void testDivideByZero() {
    try { calculator.divide(1, 0); } catch (Exception _cf_ignored1) {}
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected

    def test_assert_throws_assigned_with_qualified_assertions(self):
        source = """\
@Test
void testDivideByZero() {
    IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> calc.divide(1, 0));
}"""
        expected = """\
@Test
void testDivideByZero() {
    IllegalArgumentException ex = null;
    try { calc.divide(1, 0); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; }
}"""
        result = transform_java_assertions(source, "divide")
        assert result == expected
