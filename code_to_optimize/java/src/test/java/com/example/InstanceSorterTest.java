package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InstanceSorterTest {

    @Test
    void testBubbleSortInPlace() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {5, 3, 1, 4, 2};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceAlreadySorted() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {1, 2, 3, 4, 5};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceReversed() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {5, 4, 3, 2, 1};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testBubbleSortInPlaceWithDuplicates() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {3, 2, 4, 1, 3, 2};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{1, 2, 2, 3, 3, 4}, arr);
    }

    @Test
    void testBubbleSortInPlaceWithNegatives() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {3, -2, 7, 0, -5};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{-5, -2, 0, 3, 7}, arr);
    }

    @Test
    void testBubbleSortInPlaceSingleElement() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {42};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{42}, arr);
    }

    @Test
    void testBubbleSortInPlaceEmpty() {
        InstanceSorter sorter = new InstanceSorter();
        int[] arr = {};
        sorter.bubbleSortInPlace(arr);
        assertArrayEquals(new int[]{}, arr);
    }

    @Test
    void testBubbleSortInPlaceNull() {
        InstanceSorter sorter = new InstanceSorter();
        sorter.bubbleSortInPlace(null);
    }
}