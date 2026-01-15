/**
 * Bubble sort implementation - intentionally inefficient for optimization testing.
 */

/**
 * Sort an array using bubble sort algorithm.
 * @param {number[]} arr - The array to sort
 * @returns {number[]} - The sorted array
 */
function bubbleSort(arr) {
    const n = arr.length;
    const result = [...arr]; // Create a copy to avoid mutation

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
 * Sort an array in descending order.
 * @param {number[]} arr - The array to sort
 * @returns {number[]} - The sorted array in descending order
 */
function bubbleSortDescending(arr) {
    const n = arr.length;
    const result = [...arr];

    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (result[j] < result[j + 1]) {
                const temp = result[j];
                result[j] = result[j + 1];
                result[j + 1] = temp;
            }
        }
    }

    return result;
}

module.exports = { bubbleSort, bubbleSortDescending };
