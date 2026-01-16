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
    // Create a copy to avoid mutation
    const result = [...arr];

    // Optimized bubble: shrink the inner loop to the last swap position
    // and exit early if no swaps occur in a pass.
    let end = n - 1;
    while (end > 0) {
        let lastSwap = -1;
        for (let j = 0; j < end; j++) {
            if (result[j] > result[j + 1]) {
                // Swap elements
                const temp = result[j];
                result[j] = result[j + 1];
                result[j + 1] = temp;
                lastSwap = j;
            }
        }
        if (lastSwap === -1) break;
        end = lastSwap;
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
