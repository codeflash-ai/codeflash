package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BubbleSort sorting algorithms.
 */
class BubbleSortTest {

    @Test
    void testBubbleSort() {
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, BubbleSort.bubbleSort(new int[]{5, 3, 1, 4, 2}));
        assertArrayEquals(new int[]{1, 2, 3}, BubbleSort.bubbleSort(new int[]{3, 2, 1}));
        assertArrayEquals(new int[]{1}, BubbleSort.bubbleSort(new int[]{1}));
        assertArrayEquals(new int[]{}, BubbleSort.bubbleSort(new int[]{}));
        assertNull(BubbleSort.bubbleSort(null));
    }

    @Test
    void testBubbleSortAlreadySorted() {
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, BubbleSort.bubbleSort(new int[]{1, 2, 3, 4, 5}));
    }

    @Test
    void testBubbleSortWithDuplicates() {
        assertArrayEquals(new int[]{1, 2, 2, 3, 3, 4}, BubbleSort.bubbleSort(new int[]{3, 2, 4, 1, 3, 2}));
    }

    @Test
    void testBubbleSortWithNegatives() {
        assertArrayEquals(new int[]{-5, -2, 0, 3, 7}, BubbleSort.bubbleSort(new int[]{3, -2, 7, 0, -5}));
    }

    @Test
    void testBubbleSortDescending() {
        assertArrayEquals(new int[]{5, 4, 3, 2, 1}, BubbleSort.bubbleSortDescending(new int[]{1, 3, 5, 2, 4}));
        assertArrayEquals(new int[]{3, 2, 1}, BubbleSort.bubbleSortDescending(new int[]{1, 2, 3}));
        assertArrayEquals(new int[]{}, BubbleSort.bubbleSortDescending(new int[]{}));
    }

    @Test
    void testInsertionSort() {
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, BubbleSort.insertionSort(new int[]{5, 3, 1, 4, 2}));
        assertArrayEquals(new int[]{1, 2, 3}, BubbleSort.insertionSort(new int[]{3, 2, 1}));
        assertArrayEquals(new int[]{1}, BubbleSort.insertionSort(new int[]{1}));
        assertArrayEquals(new int[]{}, BubbleSort.insertionSort(new int[]{}));
    }

    @Test
    void testSelectionSort() {
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, BubbleSort.selectionSort(new int[]{5, 3, 1, 4, 2}));
        assertArrayEquals(new int[]{1, 2, 3}, BubbleSort.selectionSort(new int[]{3, 2, 1}));
        assertArrayEquals(new int[]{1}, BubbleSort.selectionSort(new int[]{1}));
    }

    @Test
    void testIsSorted() {
        assertTrue(BubbleSort.isSorted(new int[]{1, 2, 3, 4, 5}));
        assertTrue(BubbleSort.isSorted(new int[]{1}));
        assertTrue(BubbleSort.isSorted(new int[]{}));
        assertTrue(BubbleSort.isSorted(null));
        assertFalse(BubbleSort.isSorted(new int[]{5, 3, 1}));
        assertFalse(BubbleSort.isSorted(new int[]{1, 3, 2}));
    }

    @Test
    void testBubbleSortDoesNotMutateInput() {
        int[] original = {5, 3, 1, 4, 2};
        int[] copy = {5, 3, 1, 4, 2};
        BubbleSort.bubbleSort(original);
        assertArrayEquals(copy, original);
    }
}
