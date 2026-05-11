package com.example;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BubbleSort sorting algorithms.
 */
class BubbleSortTest {
    private static final int LARGE_SORT_SIZE = 5000;

    @Test
    void testBubbleSort() {
        assertArrayEquals(ascendingRange(LARGE_SORT_SIZE), BubbleSort.bubbleSort(descendingRange(LARGE_SORT_SIZE)));
        assertArrayEquals(new int[]{1, 2, 3}, BubbleSort.bubbleSort(new int[]{3, 2, 1}));
        assertArrayEquals(new int[]{1}, BubbleSort.bubbleSort(new int[]{1}));
        assertArrayEquals(new int[]{}, BubbleSort.bubbleSort(new int[]{}));
        assertNull(BubbleSort.bubbleSort(null));
    }

    @Test
    void testBubbleSortAlreadySorted() {
        int[] sorted = ascendingRange(LARGE_SORT_SIZE);
        assertArrayEquals(sorted, BubbleSort.bubbleSort(sorted));
    }

    @Test
    void testBubbleSortWithDuplicates() {
        int[] input = duplicateHeavyRange(LARGE_SORT_SIZE);
        assertArrayEquals(sortedCopy(input), BubbleSort.bubbleSort(input));
    }

    @Test
    void testBubbleSortWithNegatives() {
        int[] input = mixedSignedRange(LARGE_SORT_SIZE);
        assertArrayEquals(sortedCopy(input), BubbleSort.bubbleSort(input));
    }

    private static int[] ascendingRange(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = i;
        }
        return arr;
    }

    private static int[] descendingRange(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = size - i - 1;
        }
        return arr;
    }

    private static int[] duplicateHeavyRange(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (size - i - 1) % 32;
        }
        return arr;
    }

    private static int[] mixedSignedRange(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (i % 2 == 0) ? (size - i) : -(size - i);
        }
        return arr;
    }

    private static int[] sortedCopy(int[] arr) {
        int[] expected = arr.clone();
        Arrays.sort(expected);
        return expected;
    }
}
