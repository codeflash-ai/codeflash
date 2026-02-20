package com.example;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the DataProcessor class.
 */
@DisplayName("DataProcessor Tests")
class DataProcessorTest {

    @Nested
    @DisplayName("findDuplicates() Tests")
    class FindDuplicatesTests {

        @Test
        @DisplayName("should find duplicates in list")
        void testFindDuplicates() {
            List<Integer> input = Arrays.asList(1, 2, 3, 2, 4, 3, 5);
            List<Integer> duplicates = DataProcessor.findDuplicates(input);

            assertEquals(2, duplicates.size());
            assertTrue(duplicates.contains(2));
            assertTrue(duplicates.contains(3));
        }

        @Test
        @DisplayName("should return empty for no duplicates")
        void testNoDuplicates() {
            List<Integer> input = Arrays.asList(1, 2, 3, 4, 5);
            List<Integer> duplicates = DataProcessor.findDuplicates(input);

            assertTrue(duplicates.isEmpty());
        }

        @Test
        @DisplayName("should handle null input")
        void testNullInput() {
            List<Integer> duplicates = DataProcessor.findDuplicates(null);
            assertTrue(duplicates.isEmpty());
        }

        @Test
        @DisplayName("should handle strings")
        void testStrings() {
            List<String> input = Arrays.asList("a", "b", "a", "c", "b", "d");
            List<String> duplicates = DataProcessor.findDuplicates(input);

            assertEquals(2, duplicates.size());
            assertTrue(duplicates.contains("a"));
            assertTrue(duplicates.contains("b"));
        }
    }

    @Nested
    @DisplayName("groupBy() Tests")
    class GroupByTests {

        @Test
        @DisplayName("should group by length")
        void testGroupByLength() {
            List<String> input = Arrays.asList("a", "bb", "ccc", "dd", "e", "fff");
            Map<Integer, List<String>> grouped = DataProcessor.groupBy(input, String::length);

            assertEquals(3, grouped.size());
            assertEquals(2, grouped.get(1).size());
            assertEquals(2, grouped.get(2).size());
            assertEquals(2, grouped.get(3).size());
        }

        @Test
        @DisplayName("should group by first character")
        void testGroupByFirstChar() {
            List<String> input = Arrays.asList("apple", "apricot", "banana", "blueberry");
            Map<Character, List<String>> grouped = DataProcessor.groupBy(input, s -> s.charAt(0));

            assertEquals(2, grouped.size());
            assertEquals(2, grouped.get('a').size());
            assertEquals(2, grouped.get('b').size());
        }

        @Test
        @DisplayName("should handle null input")
        void testNullInput() {
            Map<Integer, List<String>> grouped = DataProcessor.groupBy(null, String::length);
            assertTrue(grouped.isEmpty());
        }
    }

    @Nested
    @DisplayName("intersection() Tests")
    class IntersectionTests {

        @Test
        @DisplayName("should find intersection")
        void testIntersection() {
            List<Integer> list1 = Arrays.asList(1, 2, 3, 4, 5);
            List<Integer> list2 = Arrays.asList(4, 5, 6, 7, 8);
            List<Integer> result = DataProcessor.intersection(list1, list2);

            assertEquals(2, result.size());
            assertTrue(result.contains(4));
            assertTrue(result.contains(5));
        }

        @Test
        @DisplayName("should return empty for no intersection")
        void testNoIntersection() {
            List<Integer> list1 = Arrays.asList(1, 2, 3);
            List<Integer> list2 = Arrays.asList(4, 5, 6);
            List<Integer> result = DataProcessor.intersection(list1, list2);

            assertTrue(result.isEmpty());
        }

        @Test
        @DisplayName("should handle null inputs")
        void testNullInputs() {
            assertTrue(DataProcessor.intersection(null, Arrays.asList(1, 2, 3)).isEmpty());
            assertTrue(DataProcessor.intersection(Arrays.asList(1, 2, 3), null).isEmpty());
        }

        @Test
        @DisplayName("should not include duplicates")
        void testNoDuplicates() {
            List<Integer> list1 = Arrays.asList(1, 1, 2, 2, 3);
            List<Integer> list2 = Arrays.asList(1, 2, 2, 4);
            List<Integer> result = DataProcessor.intersection(list1, list2);

            assertEquals(2, result.size());
        }
    }

    @Nested
    @DisplayName("flatten() Tests")
    class FlattenTests {

        @Test
        @DisplayName("should flatten nested lists")
        void testFlatten() {
            List<List<Integer>> nested = Arrays.asList(
                Arrays.asList(1, 2, 3),
                Arrays.asList(4, 5),
                Arrays.asList(6, 7, 8, 9)
            );
            List<Integer> result = DataProcessor.flatten(nested);

            assertEquals(9, result.size());
            assertEquals(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9), result);
        }

        @Test
        @DisplayName("should handle empty inner lists")
        void testEmptyInnerLists() {
            List<List<Integer>> nested = Arrays.asList(
                Arrays.asList(1, 2),
                Collections.emptyList(),
                Arrays.asList(3, 4)
            );
            List<Integer> result = DataProcessor.flatten(nested);

            assertEquals(4, result.size());
        }

        @Test
        @DisplayName("should handle null")
        void testNull() {
            assertTrue(DataProcessor.flatten(null).isEmpty());
        }
    }

    @Nested
    @DisplayName("countFrequency() Tests")
    class CountFrequencyTests {

        @Test
        @DisplayName("should count frequencies correctly")
        void testCountFrequency() {
            List<String> input = Arrays.asList("a", "b", "a", "c", "a", "b");
            Map<String, Integer> freq = DataProcessor.countFrequency(input);

            assertEquals(3, freq.get("a"));
            assertEquals(2, freq.get("b"));
            assertEquals(1, freq.get("c"));
        }

        @Test
        @DisplayName("should handle null input")
        void testNullInput() {
            assertTrue(DataProcessor.countFrequency(null).isEmpty());
        }
    }

    @Nested
    @DisplayName("nthMostFrequent() Tests")
    class NthMostFrequentTests {

        @Test
        @DisplayName("should find nth most frequent")
        void testNthMostFrequent() {
            List<String> input = Arrays.asList("a", "b", "a", "c", "a", "b", "d");

            assertEquals("a", DataProcessor.nthMostFrequent(input, 1));
            assertEquals("b", DataProcessor.nthMostFrequent(input, 2));
        }

        @Test
        @DisplayName("should return null for invalid n")
        void testInvalidN() {
            List<String> input = Arrays.asList("a", "b", "c");

            assertNull(DataProcessor.nthMostFrequent(input, 0));
            assertNull(DataProcessor.nthMostFrequent(input, 10));
        }

        @Test
        @DisplayName("should handle null input")
        void testNullInput() {
            assertNull(DataProcessor.nthMostFrequent(null, 1));
        }
    }

    @Nested
    @DisplayName("partition() Tests")
    class PartitionTests {

        @Test
        @DisplayName("should partition into chunks")
        void testPartition() {
            List<Integer> input = Arrays.asList(1, 2, 3, 4, 5, 6, 7);
            List<List<Integer>> chunks = DataProcessor.partition(input, 3);

            assertEquals(3, chunks.size());
            assertEquals(Arrays.asList(1, 2, 3), chunks.get(0));
            assertEquals(Arrays.asList(4, 5, 6), chunks.get(1));
            assertEquals(Collections.singletonList(7), chunks.get(2));
        }

        @Test
        @DisplayName("should handle exact division")
        void testExactDivision() {
            List<Integer> input = Arrays.asList(1, 2, 3, 4, 5, 6);
            List<List<Integer>> chunks = DataProcessor.partition(input, 2);

            assertEquals(3, chunks.size());
            chunks.forEach(chunk -> assertEquals(2, chunk.size()));
        }

        @Test
        @DisplayName("should handle null and invalid chunk size")
        void testInvalidInputs() {
            assertTrue(DataProcessor.partition(null, 3).isEmpty());
            assertTrue(DataProcessor.partition(Arrays.asList(1, 2, 3), 0).isEmpty());
            assertTrue(DataProcessor.partition(Arrays.asList(1, 2, 3), -1).isEmpty());
        }
    }
}
