package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InPlaceSorterTest {

    @Test
    void testBubbleSortInPlace() {
        int[] arr = {5, 3, 1, 4, 2};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceAlreadySorted() {
        int[] arr = {1, 2, 3, 4, 5};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceReversed() {
        int[] arr = {5, 4, 3, 2, 1};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceWithDuplicates() {
        int[] arr = {3, 2, 4, 1, 3, 2};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 2, 3, 3, 4}, arr);
    }

    @Test
    void testBubbleSortInPlaceWithNegatives() {
        int[] arr = {3, -2, 7, 0, -5};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{-5, -2, 0, 3, 7}, arr);
    }

    @Test
    void testBubbleSortInPlaceSingleElement() {
        int[] arr = {42};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{42}, arr);
    }

    @Test
    void testBubbleSortInPlaceEmpty() {
        int[] arr = {};
        InPlaceSorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{}, arr);
    }

    @Test
    void testBubbleSortInPlaceNull() {
        InPlaceSorter.bubbleSortInPlace(null);
    }

}