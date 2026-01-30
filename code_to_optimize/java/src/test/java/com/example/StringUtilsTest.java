package com.example;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for StringUtils utility class.
 */
class StringUtilsTest {

    @Test
    void testReverseString() {
        assertEquals("olleh", StringUtils.reverseString("hello"));
        assertEquals("a", StringUtils.reverseString("a"));
        assertEquals("", StringUtils.reverseString(""));
        assertNull(StringUtils.reverseString(null));
        assertEquals("dcba", StringUtils.reverseString("abcd"));
    }

    @Test
    void testIsPalindrome() {
        assertTrue(StringUtils.isPalindrome("racecar"));
        assertTrue(StringUtils.isPalindrome("madam"));
        assertTrue(StringUtils.isPalindrome("a"));
        assertTrue(StringUtils.isPalindrome(""));
        assertTrue(StringUtils.isPalindrome(null));
        assertTrue(StringUtils.isPalindrome("abba"));

        assertFalse(StringUtils.isPalindrome("hello"));
        assertFalse(StringUtils.isPalindrome("ab"));
    }

    @Test
    void testCountWords() {
        assertEquals(3, StringUtils.countWords("hello world test"));
        assertEquals(1, StringUtils.countWords("hello"));
        assertEquals(0, StringUtils.countWords(""));
        assertEquals(0, StringUtils.countWords("   "));
        assertEquals(0, StringUtils.countWords(null));
        assertEquals(4, StringUtils.countWords("  multiple   spaces   between   words  "));
    }

    @Test
    void testCapitalizeWords() {
        assertEquals("Hello World", StringUtils.capitalizeWords("hello world"));
        assertEquals("Hello", StringUtils.capitalizeWords("HELLO"));
        assertEquals("", StringUtils.capitalizeWords(""));
        assertNull(StringUtils.capitalizeWords(null));
        assertEquals("One Two Three", StringUtils.capitalizeWords("one two three"));
    }

    @Test
    void testCountOccurrences() {
        assertEquals(2, StringUtils.countOccurrences("hello hello", "hello"));
        assertEquals(3, StringUtils.countOccurrences("aaa", "a"));
        assertEquals(2, StringUtils.countOccurrences("aaa", "aa"));
        assertEquals(0, StringUtils.countOccurrences("hello", "world"));
        assertEquals(0, StringUtils.countOccurrences("hello", ""));
        assertEquals(0, StringUtils.countOccurrences(null, "test"));
    }

    @Test
    void testRemoveWhitespace() {
        assertEquals("helloworld", StringUtils.removeWhitespace("hello world"));
        assertEquals("abc", StringUtils.removeWhitespace("  a b c  "));
        assertEquals("test", StringUtils.removeWhitespace("test"));
        assertEquals("", StringUtils.removeWhitespace("   "));
        assertEquals("", StringUtils.removeWhitespace(""));
        assertNull(StringUtils.removeWhitespace(null));
    }

    @Test
    void testFindAllIndices() {
        List<Integer> indices = StringUtils.findAllIndices("hello", 'l');
        assertEquals(2, indices.size());
        assertEquals(2, indices.get(0));
        assertEquals(3, indices.get(1));

        indices = StringUtils.findAllIndices("aaa", 'a');
        assertEquals(3, indices.size());

        indices = StringUtils.findAllIndices("hello", 'z');
        assertTrue(indices.isEmpty());

        indices = StringUtils.findAllIndices("", 'a');
        assertTrue(indices.isEmpty());

        indices = StringUtils.findAllIndices(null, 'a');
        assertTrue(indices.isEmpty());
    }

    @Test
    void testIsNumeric() {
        assertTrue(StringUtils.isNumeric("12345"));
        assertTrue(StringUtils.isNumeric("0"));
        assertTrue(StringUtils.isNumeric("007"));

        assertFalse(StringUtils.isNumeric("12.34"));
        assertFalse(StringUtils.isNumeric("-123"));
        assertFalse(StringUtils.isNumeric("abc"));
        assertFalse(StringUtils.isNumeric("12a34"));
        assertFalse(StringUtils.isNumeric(""));
        assertFalse(StringUtils.isNumeric(null));
    }

    @Test
    void testRepeat() {
        assertEquals("abcabcabc", StringUtils.repeat("abc", 3));
        assertEquals("aaa", StringUtils.repeat("a", 3));
        assertEquals("", StringUtils.repeat("abc", 0));
        assertEquals("", StringUtils.repeat("abc", -1));
        assertEquals("", StringUtils.repeat(null, 3));
    }

    @Test
    void testTruncate() {
        assertEquals("hello", StringUtils.truncate("hello", 10));
        assertEquals("hel...", StringUtils.truncate("hello world", 6));
        assertEquals("hello...", StringUtils.truncate("hello world", 8));
        assertEquals("", StringUtils.truncate("hello", 0));
        assertEquals("", StringUtils.truncate(null, 10));
        assertEquals("hel", StringUtils.truncate("hello", 3));
    }

    @Test
    void testToTitleCase() {
        assertEquals("Hello", StringUtils.toTitleCase("hello"));
        assertEquals("Hello", StringUtils.toTitleCase("HELLO"));
        assertEquals("Hello", StringUtils.toTitleCase("hELLO"));
        assertEquals("A", StringUtils.toTitleCase("a"));
        assertEquals("", StringUtils.toTitleCase(""));
        assertNull(StringUtils.toTitleCase(null));
    }
}
