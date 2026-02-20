package com.example;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class ArrayUtilsTest {

    @Test
    void testFindDuplicates() {
        List<Integer> result = ArrayUtils.findDuplicates(new int[]{1, 2, 3, 2, 4, 3, 5});
        assertEquals(2, result.size());
        assertTrue(result.contains(2));
        assertTrue(result.contains(3));
    }

    @Test
    void testFindDuplicatesNoDuplicates() {
        List<Integer> result = ArrayUtils.findDuplicates(new int[]{1, 2, 3, 4, 5});
        assertTrue(result.isEmpty());
    }

    @Test
    void testRemoveDuplicates() {
        int[] result = ArrayUtils.removeDuplicates(new int[]{1, 2, 2, 3, 3, 3, 4});
        assertArrayEquals(new int[]{1, 2, 3, 4}, result);
    }

    @Test
    void testLinearSearch() {
        assertEquals(2, ArrayUtils.linearSearch(new int[]{10, 20, 30, 40}, 30));
        assertEquals(-1, ArrayUtils.linearSearch(new int[]{10, 20, 30, 40}, 50));
        assertEquals(-1, ArrayUtils.linearSearch(null, 10));
    }

    @Test
    void testFindIntersection() {
        int[] result = ArrayUtils.findIntersection(new int[]{1, 2, 3, 4}, new int[]{3, 4, 5, 6});
        assertArrayEquals(new int[]{3, 4}, result);
    }

    @Test
    void testFindUnion() {
        int[] result = ArrayUtils.findUnion(new int[]{1, 2, 3}, new int[]{3, 4, 5});
        assertEquals(5, result.length);
    }

    @Test
    void testReverseArray() {
        assertArrayEquals(new int[]{5, 4, 3, 2, 1}, ArrayUtils.reverseArray(new int[]{1, 2, 3, 4, 5}));
        assertArrayEquals(new int[]{1}, ArrayUtils.reverseArray(new int[]{1}));
    }

    @Test
    void testRotateRight() {
        assertArrayEquals(new int[]{4, 5, 1, 2, 3}, ArrayUtils.rotateRight(new int[]{1, 2, 3, 4, 5}, 2));
        assertArrayEquals(new int[]{1, 2, 3}, ArrayUtils.rotateRight(new int[]{1, 2, 3}, 0));
    }

    @Test
    void testCountOccurrences() {
        int[][] result = ArrayUtils.countOccurrences(new int[]{1, 2, 2, 3, 3, 3});
        assertEquals(3, result.length);
    }

    @Test
    void testKthSmallest() {
        assertEquals(1, ArrayUtils.kthSmallest(new int[]{3, 1, 4, 1, 5, 9, 2, 6}, 1));
        assertEquals(2, ArrayUtils.kthSmallest(new int[]{3, 1, 4, 1, 5, 9, 2, 6}, 3));
        assertEquals(9, ArrayUtils.kthSmallest(new int[]{3, 1, 4, 1, 5, 9, 2, 6}, 8));
    }

    @Test
    void testFindSubarray() {
        assertEquals(2, ArrayUtils.findSubarray(new int[]{1, 2, 3, 4, 5}, new int[]{3, 4}));
        assertEquals(-1, ArrayUtils.findSubarray(new int[]{1, 2, 3}, new int[]{4, 5}));
        assertEquals(0, ArrayUtils.findSubarray(new int[]{1, 2, 3}, new int[]{}));
    }

    @Test
    void testMergeSortedArrays() {
        assertArrayEquals(
            new int[]{1, 2, 3, 4, 5, 6},
            ArrayUtils.mergeSortedArrays(new int[]{1, 3, 5}, new int[]{2, 4, 6})
        );
    }
}
