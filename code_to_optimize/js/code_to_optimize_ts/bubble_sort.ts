/**
 * Bubble sort implementation - intentionally inefficient for optimization testing.
 */

/**
 * Sort an array using bubble sort algorithm.
 * @param arr - The array to sort
 * @returns A new sorted array
 */
export function bubbleSort<T>(arr: T[]): T[] {
    const result = [...arr];
    const n = result.length;

    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (result[j] > result[j + 1]) {
                // Swap elements
                const temp = result[j];
                result[j] = result[j + 1];
                result[j + 1] = temp;
            }
        }
    }

    return result;
}

/**
 * Sort an array in descending order using bubble sort.
 * @param arr - The array to sort
 * @returns A new sorted array (descending)
 */
export function bubbleSortDescending<T>(arr: T[]): T[] {
    const result = [...arr];
    const n = result.length;

    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (result[j] < result[j + 1]) {
                // Swap elements
                const temp = result[j];
                result[j] = result[j + 1];
                result[j + 1] = temp;
            }
        }
    }

    return result;
}

/**
 * Check if an array is sorted in ascending order.
 * @param arr - The array to check
 * @returns True if the array is sorted in ascending order
 */
export function isSorted<T>(arr: T[]): boolean {
    for (let i = 0; i < arr.length - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}
