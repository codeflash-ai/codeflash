package com.example;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the StringUtils class.
 */
@DisplayName("StringUtils Tests")
class StringUtilsTest {

    @Nested
    @DisplayName("reverse() Tests")
    class ReverseTests {

        @Test
        @DisplayName("should reverse a simple string")
        void testReverseSimple() {
            assertEquals("olleh", StringUtils.reverse("hello"));
            assertEquals("dlrow", StringUtils.reverse("world"));
        }

        @Test
        @DisplayName("should handle single character")
        void testReverseSingleChar() {
            assertEquals("a", StringUtils.reverse("a"));
        }

        @ParameterizedTest
        @NullAndEmptySource
        @DisplayName("should handle null and empty strings")
        void testReverseNullEmpty(String input) {
            assertEquals(input, StringUtils.reverse(input));
        }

        @Test
        @DisplayName("should handle palindrome")
        void testReversePalindrome() {
            assertEquals("radar", StringUtils.reverse("radar"));
        }
    }

    @Nested
    @DisplayName("isPalindrome() Tests")
    class PalindromeTests {

        @ParameterizedTest
        @ValueSource(strings = {"radar", "level", "civic", "rotor", "kayak"})
        @DisplayName("should return true for palindromes")
        void testPalindromes(String input) {
            assertTrue(StringUtils.isPalindrome(input));
        }

        @ParameterizedTest
        @ValueSource(strings = {"hello", "world", "java", "python"})
        @DisplayName("should return false for non-palindromes")
        void testNonPalindromes(String input) {
            assertFalse(StringUtils.isPalindrome(input));
        }

        @Test
        @DisplayName("should handle case insensitivity")
        void testCaseInsensitive() {
            assertTrue(StringUtils.isPalindrome("Radar"));
            assertTrue(StringUtils.isPalindrome("LEVEL"));
        }

        @Test
        @DisplayName("should ignore spaces")
        void testIgnoreSpaces() {
            assertTrue(StringUtils.isPalindrome("race car"));
            assertTrue(StringUtils.isPalindrome("A man a plan a canal Panama"));
        }

        @Test
        @DisplayName("should return false for null")
        void testNull() {
            assertFalse(StringUtils.isPalindrome(null));
        }
    }

    @Nested
    @DisplayName("countOccurrences() Tests")
    class CountOccurrencesTests {

        @Test
        @DisplayName("should count occurrences correctly")
        void testCount() {
            assertEquals(3, StringUtils.countOccurrences("abcabc abc", "abc"));
            assertEquals(2, StringUtils.countOccurrences("hello hello", "hello"));
        }

        @Test
        @DisplayName("should return 0 for no matches")
        void testNoMatches() {
            assertEquals(0, StringUtils.countOccurrences("hello world", "xyz"));
        }

        @ParameterizedTest
        @CsvSource({
            "'aaaaaa', 'aa', 5",
            "'banana', 'ana', 2",
            "'mississippi', 'issi', 2"
        })
        @DisplayName("should handle overlapping matches")
        void testOverlapping(String str, String sub, int expected) {
            assertEquals(expected, StringUtils.countOccurrences(str, sub));
        }

        @Test
        @DisplayName("should handle null inputs")
        void testNullInputs() {
            assertEquals(0, StringUtils.countOccurrences(null, "test"));
            assertEquals(0, StringUtils.countOccurrences("test", null));
            assertEquals(0, StringUtils.countOccurrences("test", ""));
        }
    }

    @Nested
    @DisplayName("isAnagram() Tests")
    class AnagramTests {

        @Test
        @DisplayName("should detect anagrams")
        void testAnagrams() {
            assertTrue(StringUtils.isAnagram("listen", "silent"));
            assertTrue(StringUtils.isAnagram("evil", "vile"));
            assertTrue(StringUtils.isAnagram("anagram", "nagaram"));
        }

        @Test
        @DisplayName("should reject non-anagrams")
        void testNonAnagrams() {
            assertFalse(StringUtils.isAnagram("hello", "world"));
            assertFalse(StringUtils.isAnagram("abc", "abcd"));
        }

        @Test
        @DisplayName("should be case insensitive")
        void testCaseInsensitive() {
            assertTrue(StringUtils.isAnagram("Listen", "Silent"));
        }

        @Test
        @DisplayName("should handle null inputs")
        void testNullInputs() {
            assertFalse(StringUtils.isAnagram(null, "test"));
            assertFalse(StringUtils.isAnagram("test", null));
        }
    }

    @Nested
    @DisplayName("findAnagrams() Tests")
    class FindAnagramsTests {

        @Test
        @DisplayName("should find all anagram positions")
        void testFindAnagrams() {
            List<Integer> result = StringUtils.findAnagrams("cbaebabacd", "abc");
            assertEquals(2, result.size());
            assertTrue(result.contains(0));
            assertTrue(result.contains(6));
        }

        @Test
        @DisplayName("should return empty list for no matches")
        void testNoMatches() {
            List<Integer> result = StringUtils.findAnagrams("hello", "xyz");
            assertTrue(result.isEmpty());
        }

        @Test
        @DisplayName("should handle null inputs")
        void testNullInputs() {
            assertTrue(StringUtils.findAnagrams(null, "abc").isEmpty());
            assertTrue(StringUtils.findAnagrams("abc", null).isEmpty());
        }
    }

    @Nested
    @DisplayName("longestCommonPrefix() Tests")
    class LongestCommonPrefixTests {

        @Test
        @DisplayName("should find common prefix")
        void testCommonPrefix() {
            assertEquals("fl", StringUtils.longestCommonPrefix(new String[]{"flower", "flow", "flight"}));
            assertEquals("ap", StringUtils.longestCommonPrefix(new String[]{"apple", "ape", "april"}));
        }

        @Test
        @DisplayName("should return empty for no common prefix")
        void testNoCommonPrefix() {
            assertEquals("", StringUtils.longestCommonPrefix(new String[]{"dog", "car", "race"}));
        }

        @Test
        @DisplayName("should handle single string")
        void testSingleString() {
            assertEquals("hello", StringUtils.longestCommonPrefix(new String[]{"hello"}));
        }

        @Test
        @DisplayName("should handle null and empty array")
        void testNullEmpty() {
            assertEquals("", StringUtils.longestCommonPrefix(null));
            assertEquals("", StringUtils.longestCommonPrefix(new String[]{}));
        }
    }
}
