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
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }

        int n = result.length;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                if (result[j] > result[j + 1]) {
                    int temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }

        return result;
    }

}
