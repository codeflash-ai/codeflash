package com.example;

/**
 * Sorting algorithms.
 */
public class BubbleSort {

    /**
     * Sort an array using bubble sort algorithm.
     *
     * @param arr Array to sort
     * @return New sorted array (ascending order)
     */
    public static int[] bubbleSort(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        // Use System.arraycopy for faster bulk copy
        System.arraycopy(arr, 0, result, 0, arr.length);

        int n = result.length;

        // Optimized bubble sort: track last swap to reduce the inner loop boundary
        while (n > 1) {
            int newN = 0;
            for (int j = 1; j < n; j++) {
                if (result[j - 1] > result[j]) {
                    int temp = result[j - 1];
                    result[j - 1] = result[j];
                    result[j] = temp;
                    newN = j;
                }
            }
            n = newN;
        }

        return result;
    }

    /**
     * Sort an array in descending order using bubble sort.
     *
     * @param arr Array to sort
     * @return New sorted array (descending order)
     */
    public static int[] bubbleSortDescending(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }

        int n = result.length;

        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (result[j] < result[j + 1]) {
                    int temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }

        return result;
    }

    /**
     * Sort an array using insertion sort algorithm.
     *
     * @param arr Array to sort
     * @return New sorted array
     */
    public static int[] insertionSort(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }

        int n = result.length;

        for (int i = 1; i < n; i++) {
            int key = result[i];
            int j = i - 1;

            while (j >= 0 && result[j] > key) {
                result[j + 1] = result[j];
                j = j - 1;
            }
            result[j + 1] = key;
        }

        return result;
    }

    /**
     * Sort an array using selection sort algorithm.
     *
     * @param arr Array to sort
     * @return New sorted array
     */
    public static int[] selectionSort(int[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }

        int n = result.length;

        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (result[j] < result[minIdx]) {
                    minIdx = j;
                }
            }

            int temp = result[minIdx];
            result[minIdx] = result[i];
            result[i] = temp;
        }

        return result;
    }

    /**
     * Check if an array is sorted in ascending order.
     *
     * @param arr Array to check
     * @return true if sorted in ascending order
     */
    public static boolean isSorted(int[] arr) {
        if (arr == null || arr.length <= 1) {
            return true;
        }

        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                return false;
            }
        }
        return true;
    }
}
