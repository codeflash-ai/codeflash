package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Array utility functions.
 */
public class ArrayUtils {

    /**
     * Find all duplicate elements in an array using nested loops.
     *
     * @param arr Input array
     * @return List of duplicate elements
     */
    public static List<Integer> findDuplicates(int[] arr) {
        List<Integer> duplicates = new ArrayList<>();
        if (arr == null || arr.length < 2) {
            return duplicates;
        }

        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[i] == arr[j] && !duplicates.contains(arr[i])) {
                    duplicates.add(arr[i]);
                }
            }
        }
        return duplicates;
    }

    /**
     * Remove duplicates from array using nested loops.
     *
     * @param arr Input array
     * @return Array without duplicates
     */
    public static int[] removeDuplicates(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        List<Integer> unique = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            boolean found = false;
            for (int j = 0; j < unique.size(); j++) {
                if (unique.get(j) == arr[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique.add(arr[i]);
            }
        }

        int[] result = new int[unique.size()];
        for (int i = 0; i < unique.size(); i++) {
            result[i] = unique.get(i);
        }
        return result;
    }

    /**
     * Linear search through array.
     *
     * @param arr Array to search
     * @param target Value to find
     * @return Index of target, or -1 if not found
     */
    public static int linearSearch(int[] arr, int target) {
        if (arr == null) {
            return -1;
        }

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find intersection of two arrays using nested loops.
     *
     * @param arr1 First array
     * @param arr2 Second array
     * @return Array of common elements
     */
    public static int[] findIntersection(int[] arr1, int[] arr2) {
        if (arr1 == null || arr2 == null) {
            return new int[0];
        }

        List<Integer> intersection = new ArrayList<>();
        for (int i = 0; i < arr1.length; i++) {
            for (int j = 0; j < arr2.length; j++) {
                if (arr1[i] == arr2[j] && !intersection.contains(arr1[i])) {
                    intersection.add(arr1[i]);
                }
            }
        }

        int[] result = new int[intersection.size()];
        for (int i = 0; i < intersection.size(); i++) {
            result[i] = intersection.get(i);
        }
        return result;
    }

    /**
     * Find union of two arrays using nested loops.
     *
     * @param arr1 First array
     * @param arr2 Second array
     * @return Array of all unique elements from both arrays
     */
    public static int[] findUnion(int[] arr1, int[] arr2) {
        List<Integer> union = new ArrayList<>();

        if (arr1 != null) {
            for (int i = 0; i < arr1.length; i++) {
                if (!union.contains(arr1[i])) {
                    union.add(arr1[i]);
                }
            }
        }

        if (arr2 != null) {
            for (int i = 0; i < arr2.length; i++) {
                if (!union.contains(arr2[i])) {
                    union.add(arr2[i]);
                }
            }
        }

        int[] result = new int[union.size()];
        for (int i = 0; i < union.size(); i++) {
            result[i] = union.get(i);
        }
        return result;
    }

    /**
     * Reverse an array.
     *
     * @param arr Array to reverse
     * @return Reversed array
     */
    public static int[] reverseArray(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[arr.length - 1 - i];
        }
        return result;
    }

    /**
     * Rotate array to the right by k positions.
     *
     * @param arr Array to rotate
     * @param k Number of positions to rotate
     * @return Rotated array
     */
    public static int[] rotateRight(int[] arr, int k) {
        if (arr == null || arr.length == 0 || k == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }

        k = k % result.length;

        for (int rotation = 0; rotation < k; rotation++) {
            int last = result[result.length - 1];
            for (int i = result.length - 1; i > 0; i--) {
                result[i] = result[i - 1];
            }
            result[0] = last;
        }

        return result;
    }

    /**
     * Count occurrences of each element using nested loops.
     *
     * @param arr Input array
     * @return 2D array where [i][0] is element and [i][1] is count
     */
    public static int[][] countOccurrences(int[] arr) {
        if (arr == null || arr.length == 0) {
            return new int[0][0];
        }

        List<int[]> counts = new ArrayList<>();

        for (int i = 0; i < arr.length; i++) {
            boolean found = false;
            for (int j = 0; j < counts.size(); j++) {
                if (counts.get(j)[0] == arr[i]) {
                    counts.get(j)[1]++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                counts.add(new int[]{arr[i], 1});
            }
        }

        int[][] result = new int[counts.size()][2];
        for (int i = 0; i < counts.size(); i++) {
            result[i] = counts.get(i);
        }
        return result;
    }

    /**
     * Find the k-th smallest element using repeated minimum finding.
     *
     * @param arr Input array
     * @param k Position (1-indexed)
     * @return k-th smallest element
     */
    public static int kthSmallest(int[] arr, int k) {
        if (arr == null || arr.length == 0 || k <= 0 || k > arr.length) {
            throw new IllegalArgumentException("Invalid input");
        }

        int[] copy = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            copy[i] = arr[i];
        }

        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < copy.length; j++) {
                if (copy[j] < copy[minIdx]) {
                    minIdx = j;
                }
            }
            int temp = copy[i];
            copy[i] = copy[minIdx];
            copy[minIdx] = temp;
        }

        return copy[k - 1];
    }

    /**
     * Check if array contains a subarray using brute force.
     *
     * @param arr Main array
     * @param subArr Subarray to find
     * @return Starting index of subarray, or -1 if not found
     */
    public static int findSubarray(int[] arr, int[] subArr) {
        if (arr == null || subArr == null || subArr.length > arr.length) {
            return -1;
        }

        if (subArr.length == 0) {
            return 0;
        }

        for (int i = 0; i <= arr.length - subArr.length; i++) {
            boolean match = true;
            for (int j = 0; j < subArr.length; j++) {
                if (arr[i + j] != subArr[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Merge two sorted arrays.
     *
     * @param arr1 First sorted array
     * @param arr2 Second sorted array
     * @return Merged sorted array
     */
    public static int[] mergeSortedArrays(int[] arr1, int[] arr2) {
        if (arr1 == null) arr1 = new int[0];
        if (arr2 == null) arr2 = new int[0];

        int[] result = new int[arr1.length + arr2.length];
        int i = 0, j = 0, k = 0;

        while (i < arr1.length && j < arr2.length) {
            if (arr1[i] <= arr2[j]) {
                result[k] = arr1[i];
                i++;
            } else {
                result[k] = arr2[j];
                j++;
            }
            k++;
        }

        while (i < arr1.length) {
            result[k] = arr1[i];
            i++;
            k++;
        }

        while (j < arr2.length) {
            result[k] = arr2[j];
            j++;
            k++;
        }

        return result;
    }
}
